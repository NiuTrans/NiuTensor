/* NiuTrans.Tensor - an open-source tensor library
* Copyright (C) 2017, Natural Language Processing Lab, Northestern University.
* All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

/*
* $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-04-24
*/

#include "../../XDevice.h"
#include "../../XUtility.h"
#include "../../XTensor.h"
#include "../utilities/SetAscendingOrder.h"
#include "TopK.h"
#include "TopK.cuh"
#include "Sort.cuh"
#define WORKERSNUM 64

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* heap item */
template <typename T>
struct CudaHeapNode
{
    /* node index */
    int index;

    /* value of the node */
    T value;

    __device__ CudaHeapNode() {};

    __device__ CudaHeapNode(int i, T v)
    {
        index = i;
        value = v;
    };
};

/* heap (device code) */
template<HeapType hType, typename T>
class CudaXHeap
{
public:
    /* number of the items the heap keeps */
    int size;

    /* number of the items that are already in the heap */
    int count;

    /* items */
    CudaHeapNode<T> * items;

    /* value for the top-most item*/
    T topValue;

public:
    /* constructor */
    __device__ CudaXHeap(int mySize, CudaHeapNode<T> * myItems)
    {
        size = mySize;
        count = 0;
        items = myItems;
        topValue = 0;
    }
    /* constructor */
    __device__ CudaXHeap(int mySize, int myCount, CudaHeapNode<T> * myItems)
    {
        size = mySize;
        count = myCount;
        items = myItems;
        topValue = items[0].value;
    }
    /* compare node i and node j */
    __device__ bool Compare(int i, int j)
    {
        if (hType == MIN_HEAP)
            return items[i].value < items[j].value;
        else
            return items[j].value < items[i].value;
    }

    /* swap */
    __device__ void Swap(int i, int j)
    {
        int tmpIndex = items[i].index;
        T tmpValue = items[i].value;
        items[i] = items[j];
        items[j].index = tmpIndex;
        items[j].value = tmpValue;
    }

    /* replace the top-most item and update the heap */
    __device__ void ReplaceTop(CudaHeapNode<T> node)
    {
        items[0] = node;
        Down(0);
        topValue = items[0].value;
    }

    /* replace the top-most item and update the heap */
    __device__ void ReplaceTop(int index, T value)
    {
        items[0].index = index;
        items[0].value = value;
        Down(0);
        topValue = items[0].value;
    }

    /* push an item into the heap */
    __device__ void Push(CudaHeapNode<T> node)
    {
        items[count] = node;
        Up(count);
        count++;
        topValue = items[0].value;
    }

    /* push an item into the heap */
    __device__ void Push(int index, T value)
    {
        items[count].index = index;
        items[count].value = value;
        Up(count);
        count++;
        topValue = items[0].value;
    }

    /* move item k down the tree */
    __device__ void Down(int k)
    {
        int i = k;
        int i2 = i + i;
        while (i2 + 1 < count) {
            int l = i2 + 1;
            int r = i2 + 2;
            int m = (Compare(l, r) || r >= count) ? l : r;
            if (Compare(i, m))
                break;
            Swap(i, m);
            i = m;
            i2 = m << 1;
        }
    }

    /* move item k up the tree */
    __device__ void Up(int k)
    {
        int i = k;
        int parent = (i - 1) >> 1;
        while (i > 0 && !Compare(parent, i)) {
            Swap(parent, i);
            i = parent;
            parent = (i - 1) >> 1;
        }
    }
};

/*
get the top-k items
>> input - the input data array
>> stride - number of items we go over when we move to the next item along a given dimension
>> strideNum - size of the given dimension
>> blockNum - number of data blocks
>> k - as it is
>> minValue - min value of an item
>> output - the output data array
>> index - the output index array
*/
template<class T> __global__
void KernelTopK(T * input, int stride, int strideNum, int blockNum, int k, T minValue, T * output, int * index)
{
    __shared__ CudaHeapNode<T> heapData[(SHARED_MEMORY_SIZE) / sizeof(CudaHeapNode<T>)];

    /* worker index */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* index of the data arry along the given dimension */
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= strideNum || i >= blockDim.x || j >= stride * blockNum)
        return;

    int blockIndex = j / stride;
    int offsetInBlock = j % stride;
    T * d = input + stride * strideNum * blockIndex + offsetInBlock;

    CudaXHeap<MIN_HEAP, T> heap(k, heapData + k * (threadIdx.y * blockDim.x + threadIdx.x));
    __syncthreads();

    /* go over the data array and build the heap */
    int indexOffset = blockDim.x;
    int dataOffset = stride * blockDim.x;

    if (i + (heap.size - 1) * indexOffset < strideNum) {
        int p = i;
        int q = i * stride;
        for (int m = 0; m < heap.size; m++) {
            heap.Push(p, d[q]);
            p += indexOffset;
            q += dataOffset;
        }

        for (; p < strideNum; p += indexOffset, q += dataOffset) {
            T v = d[q];
            if (v > heap.topValue) {
                heap.ReplaceTop(p, v);
            }
        }
    }
    else {
        for (int p = i, q = i * stride; p < strideNum; p += indexOffset, q += dataOffset) {
            heap.Push(p, d[q]);
        }
    }

    /* fill the heap if no enough items are processed */
    while (heap.count < heap.size) {
        heap.Push(-1, minValue);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        CudaXHeap<MIN_HEAP, T> heapFinal(k, k, heapData + k * threadIdx.y * blockDim.x);

        /* 
	merge the result over the workers.
        This can be improved by parallel merging 
	*/
        if (blockDim.x > 1) {
            for (int p = 1; p < blockDim.x && p < strideNum; p++) {
                CudaHeapNode<T> * hd = heapData + k * (threadIdx.y * blockDim.x + p);
                for (int q = 0; q < k; q++) {
                    if (hd[q].value > heapFinal.topValue)
                        heapFinal.ReplaceTop(hd[q]);
                }
            }
        }

        int offset = stride * k * blockIndex + offsetInBlock;
        T * dOutput = output + offset;
        int * indexOutput = index + offset;

        /* pop for the final result */
        for (int q = k - 1; q >= 0; q--) {
            dOutput[stride * q] = heapFinal.items[0].value;
            indexOutput[stride * q] = heapFinal.items[0].index;
            heapFinal.items[0] = heapFinal.items[heapFinal.count - 1];
            heapFinal.count--;
            heapFinal.Down(0);
        }
    }
}

/*
get the top-k items
>> input - the input data array
>> stride - number of items we go over when we move to the next item along a given dimension
>> strideNum - size of the given dimension
>> blockNum - number of data blocks
>> k - as it is
>> minValue - min value of an item
>> output - the output data array
>> index - the output index array
*/
template<class T> __global__
void KernelTopK2(T * input, int stride, int strideNum, int blockNum, int k, T minValue, T * output, int * index)
{
    __shared__ CudaHeapNode<T> heapData[(SHARED_MEMORY_SIZE) / sizeof(CudaHeapNode<T>)];

    /* worker index */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* index of the data arry along the given dimension */
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= strideNum || i >= blockDim.x || j >= stride * blockNum)
        return;

    int blockIndex = j / stride;
    int offsetInBlock = j % stride;
    T * d = input + stride * strideNum * blockIndex + offsetInBlock;

    CudaXHeap<MIN_HEAP, T> heap(k, heapData + k * (threadIdx.y * blockDim.x + threadIdx.x));
    __syncthreads();

    /* go over the data array and build the heap */
    int indexOffset = blockDim.x;
    int dataOffset = stride * blockDim.x;

    if (i + (heap.size - 1) * indexOffset < strideNum) {
        int p = i;
        int q = i * stride;
        for (int m = 0; m < heap.size; m++) {
            heap.Push(p, d[q]);
            p += indexOffset;
            q += dataOffset;
        }

        for (; p < strideNum; p += indexOffset, q += dataOffset) {
            T v = d[q];
            if (v > heap.topValue) {
                heap.ReplaceTop(p, v);
            }
        }
    }
    else {
        for (int p = i, q = i * stride; p < strideNum; p += indexOffset, q += dataOffset) {
            heap.Push(p, d[q]);
        }
    }

    /* fill the heap if no enough items are processed */
    while (heap.count < heap.size) {
        heap.Push(-1, minValue);
    }

    __syncthreads();

    /* parallel merging */
    int heapOffset = threadIdx.y * blockDim.x;
    CudaHeapNode<T> * heapLocalData = heapData + k * (heapOffset + i);
    CudaXHeap<MIN_HEAP, T> heapLocal(k, k, heapLocalData);
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && i + s < strideNum) {
            CudaHeapNode<T> * hd = heapLocalData + k * s;
            for (int q = 0; q < k; q++) {
                if (hd[q].value > heapLocal.topValue)
                    heapLocal.ReplaceTop(hd[q]);
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        CudaXHeap<MIN_HEAP, T> heapFinal(k, k, heapData + k * heapOffset);
        int offset = stride * k * blockIndex + offsetInBlock;
        T * dOutput = output + offset;
        int * indexOutput = index + offset;

        /* pop for the final result */
        for (int q = k - 1; q >= 0; q--) {
            dOutput[stride * q] = heapFinal.items[0].value;
            indexOutput[stride * q] = heapFinal.items[0].index;
            heapFinal.items[0] = heapFinal.items[heapFinal.count - 1];
            heapFinal.count--;
            heapFinal.Down(0);
        }
    }
}

/*
get the top-k items
>> input - the input data array
>> stride - number of items we go over when we move to the next item along a given dimension
>> strideNum - size of the given dimension
>> blockNum - number of data blocks
>> k - as it is
>> minValue - min value of an item
>> output - the output data array
>> index - the output index array
*/
template<class T> __global__
void KernelTopK3(T * input, int stride, int strideNum, int blockNum, int k, T minValue, T * output, int * index)
{
    __shared__ CudaHeapNode<T> heapData[(SHARED_MEMORY_SIZE - 512 * sizeof(T)) / sizeof(CudaHeapNode<T>)];
    __shared__ T eachHeapMaxValue[512];
    /*optimization k size the parameter must more than half of k*/
    int parameter = 0;

    /* worker index */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* index of the data arry along the given dimension */
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= strideNum || i >= blockDim.x || j >= stride * blockNum)
        return;

    int blockIndex = j / stride;
    int offsetInBlock = j % stride;
    T * d = input + stride * strideNum * blockIndex + offsetInBlock;

    CudaXHeap<MIN_HEAP, T> heap(k - parameter, heapData + k * (threadIdx.y * blockDim.x + threadIdx.x));
    __syncthreads();

    /* go over the data array and build the heap */
    int indexOffset = blockDim.x;
    int dataOffset = stride * blockDim.x;

    if (i + (heap.size - 1) * indexOffset < strideNum) {
        int p = i;
        int q = i * stride;
        for (int m = 0; m < heap.size; m++) {
            heap.Push(p, d[q]);
            p += indexOffset;
            q += dataOffset;
        }

        for (; p < strideNum; p += indexOffset, q += dataOffset) {
            T v = d[q];
            if (v > heap.topValue) {
                heap.ReplaceTop(p, v);
            }
        }
    }
    else {
        for (int p = i, q = i * stride; p < strideNum; p += indexOffset, q += dataOffset) {
            heap.Push(p, d[q]);
        }
    }
    /* fill the heap if no enough items are processed */
    while (heap.count < heap.size) {
        heap.Push(-1, minValue);
    }
    __syncthreads();

    /* to merge the heap use another way */
    T minData = minValue;
    int heapLimit = heap.count / 2;
    if (heapLimit % 2 == 0 && heapLimit != 0) heapLimit -= 1;
    for (int counter = heap.count - 1; counter >= heapLimit; --counter) {
        if (minData < heap.items[counter].value)
            minData = heap.items[counter].value;
    }
    eachHeapMaxValue[threadIdx.y * blockDim.x + threadIdx.x] = minData;

    //need more optimation
    if (i == 0) {
        int threadLimit = threadIdx.y  * blockDim.x + min(blockDim.x,strideNum);
        CudaXHeap<MIN_HEAP, T> chooseHeap(k, heapData + k * ((blockDim.x * blockDim.y) + threadIdx.y));
        int counter = threadIdx.y * blockDim.x;
        for (; counter < threadIdx.y * blockDim.x + min(k, blockDim.x); ++counter) {
            chooseHeap.Push(counter, eachHeapMaxValue[counter]);
        }
        for (; counter < threadLimit; ++counter) {
            if (eachHeapMaxValue[counter]>chooseHeap.items[0].value) {
                chooseHeap.ReplaceTop(counter, eachHeapMaxValue[counter]);
            }
        }
        int heapNum = chooseHeap.count;
        CudaXHeap<MIN_HEAP, T>  ansHeapData(k, k - parameter, heapData + k * chooseHeap.items[0].index);
        int miss = parameter;
        for (counter = 1; counter < heapNum; ++counter) {
            chooseHeap.items[0] = chooseHeap.items[chooseHeap.count - 1];
            chooseHeap.count--;
            chooseHeap.Down(0);
            CudaHeapNode<T> * cmpHeapData = heapData + k * (chooseHeap.items[0].index);
            int cmpHeapLimit = 0;
            if (counter + heapLimit <= k - parameter && heapNum == k){
                cmpHeapLimit = heapLimit;
            }
            /* take the max data from the minHeap,so start search from the leaf node */
            for (int iterator = k - 1 - parameter; iterator >= cmpHeapLimit; --iterator){
                if (miss > 0){
                    ansHeapData.Push(cmpHeapData[iterator].index, cmpHeapData[iterator].value);
                    miss--;
                }
                else if (ansHeapData.items[0].value < cmpHeapData[iterator].value){
                    ansHeapData.ReplaceTop(cmpHeapData[iterator].index, cmpHeapData[iterator].value);
                }
            }
        }
        int offset = stride * k * blockIndex + offsetInBlock;
        T * dOutput = output + offset;
        int * indexOutput = index + offset;
        for (int q = 0; q < k; ++q){
            dOutput[stride * q] = ansHeapData.items[q].value;
            indexOutput[stride * q] = ansHeapData.items[q].index;
        }
    }
}


__device__ __forceinline__ 
unsigned getLaneMaskLe() 
{
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
    return mask;
}

__device__ __forceinline__ 
int getLaneId() 
{
    int laneId;
    asm("mov.s32 %0, %laneid;" : "=r"(laneId));
    return laneId;
}

__device__ 
unsigned convert(float v)
{
    unsigned x = __float_as_int(v);
    unsigned mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;
    return (x ^ mask);
}

__device__ 
float convert(unsigned int v)
{
    float x = __uint_as_float(v);
    return x;
}

__device__ 
float deconvert(unsigned int v) 
{
    unsigned int mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;
    return __int_as_float(v ^ mask);
}

__global__ 
void convert2uintV2(float* input, unsigned int *output, int stride, int strideNum, int blockNum, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int blockIndex = idy / stride;
    int offsetInBlock = idy% stride;
#pragma unroll
    for (int i = idx * stride + stride * strideNum * blockIndex + offsetInBlock;
        i < stride * strideNum * blockIndex + offsetInBlock + stride * strideNum && i < size;
        i += stride * blockDim.x){
        output[i] = convert(input[i]);
    }
}

__global__ 
void deconvert2floatV2(unsigned int * input, float *output, int stride, int strideNum, int blockNum, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    //int strideNum = (int)strideNumSize;
    //if (flag) strideNum = strideNumSize[idy];
    int blockIndex = idy / stride;
    int offsetInBlock = idy% stride;
#pragma unroll
    for (int i = idx * stride + stride * strideNum * blockIndex + offsetInBlock;
        i < stride * strideNum * blockIndex + offsetInBlock + stride * strideNum && i < size;
        i += stride * blockDim.x){
        output[i] = deconvert(input[i]);
    }
}

__device__ 
void radixCount(unsigned int *data, int limit, int *posCount, unsigned int mask, int maskDesire, unsigned int desire, int stride, int strideNum, int blockNum)
{

    /*the idx th thread in one vector */
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    /* the idy th vector in one tensor */
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int blockIndex = idy / stride;
    int offsetInBlock = idy% stride;
    for (int j = idx*stride + stride * strideNum * blockIndex + offsetInBlock;
        j<  stride * strideNum * blockIndex + offsetInBlock + stride*strideNum && j<limit;
        j += stride * WORKERSNUM) {
        if ((data[j] & maskDesire) == desire) {
            if (data[j] & mask) {
                posCount[(idy % (512 / WORKERSNUM))*blockDim.x + idx]++;
            }
        }
    }
}

/* We can use this way to check thread status in a warp fastly,
   note that the theard number need be 32 times */
__device__ 
void gpuCheckWarp(int *smem, bool in, int *carry, int *index)
{
    int vote = __ballot_sync(0xffffffff, in);
    *index = __popc(getLaneMaskLe() & vote);
    *carry = __popc(vote);
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warp = idx / 32; 
    int warpNum = blockDim.x / 32;

    if (getLaneId() == 0) {
        /* save each warp carry */
        smem[warp + warpNum * threadIdx.y] = *carry; 
    }
    __syncthreads();
    /* use one thread to count the carry for globe the warp */
    if (idx == 0) {
        for (int i = 1 + warpNum * threadIdx.y; i < warpNum * (threadIdx.y + 1); ++i) {
            smem[i] += smem[i - 1];
        }
    }
    __syncthreads();
    if (warp % warpNum) {
        *index += smem[warpNum * threadIdx.y + warp - 1];
    }
    *carry = smem[warpNum * threadIdx.y + warpNum - 1];
}

/*
collect the data bigger than pattern as ans return
*/
__device__ 
void collectNumber(unsigned int *data, int stride, int strideNum, int limit, 
                    unsigned int pattern, float *ans, int *ansIndex, int k)
{
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int blockIndex = idy / stride;
    int offsetInBlock = idy % stride;

    /* for count each warp's tmp carry */
    __shared__ int smem[32]; 
    int carry;
    int index;
    int vectorLimit = stride * strideNum * blockIndex + offsetInBlock + stride * strideNum;
    int alibnStrideNum = strideNum;
    if (alibnStrideNum % blockDim.x) alibnStrideNum = alibnStrideNum + blockDim.x - (alibnStrideNum % blockDim.x);
    int vectorAlibnLimit = stride * strideNum * blockIndex + offsetInBlock + stride * alibnStrideNum;
    int ansArrayIndex = stride * k * blockIndex + offsetInBlock;

    int ansSize = 0;
    __syncthreads();

#pragma unroll
    for (int i = idx * stride + stride * strideNum * blockIndex + offsetInBlock;
        i < vectorAlibnLimit; i += stride * WORKERSNUM){

        bool hasTopk = false;
        if (i < vectorLimit&&data[i] > pattern){
            hasTopk = true;
        }
        gpuCheckWarp(smem, hasTopk, &carry, &index);
        if (carry > 0) {
            if (hasTopk) {
                ans[ansArrayIndex + (index - 1) * stride] = deconvert(data[i]);
                ansIndex[ansArrayIndex + (index - 1) * stride] = i - stride * strideNum * blockIndex;
            }
            ansArrayIndex += carry * stride;
            ansSize += carry;
        }
        __syncthreads();
    }
    if (ansSize < k){
        int ramindNum = k - ansSize;
#pragma unroll
        for (int i = idx * stride + stride * strideNum * blockIndex + offsetInBlock; i < vectorAlibnLimit; i += stride * WORKERSNUM) {
            bool hasTopk = false;
            if (i < vectorLimit && data[i] == pattern) {
                hasTopk = true;
            }

            gpuCheckWarp(smem, hasTopk, &carry, &index);

            if (carry>0) {
                int checkTmpIndex = ansArrayIndex + (index - 1) * stride;
                /* for don't pointer boundary overflow, for instance, 
                   if there need one index,but two index fits, wo should filter the bigger index */
                if (hasTopk && checkTmpIndex <stride * k * blockIndex + offsetInBlock + stride * k) {
                    ans[checkTmpIndex] = deconvert(pattern);
                    ansIndex[checkTmpIndex] = i - stride * strideNum * blockIndex;
                }
                ramindNum -= carry;
                ansArrayIndex += carry * stride;
                if (ramindNum <= 0) break;
            }
            __syncthreads();
        }
    }
}

/*
This is an old way,we use one thread to collect number and this way is very slow,so we drop it 
*/
__device__ 
void collectNumberOld(unsigned int *data, int n, int k, unsigned int pattern, unsigned int *ans, int *indexNum, int stride, int strideNum)
{
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int blockIndex = idy / stride;
    int offsetInBlock = idy % stride;
    int cot = 0;
    for (int i = stride * strideNum * blockIndex + offsetInBlock, j = 0; j < strideNum; j++, i += stride) {
        if (data[i] > pattern) {
            ans[cot] = data[i];
            indexNum[cot++] = j;
        }
    }
    /* if the cot < k ,so the left value must be desire */
    if (cot < k) {
        for (int i = cot; i < k; ++i) {
            ans[i] = pattern;
        }
        /* count the remain index and the data value must equal pattern */
        for (int i = stride * strideNum * blockIndex + offsetInBlock, j = 0; j < strideNum; j++, i += stride) {
            if (data[i] == pattern) {
                indexNum[cot++] = j;
                if (cot == k) break;
            }
        }
    }
}

/*
When k is very big, we can't use share memory to calculate, so we use radix select algorithm
*/
template<class T> __global__
void KernelTopKRadixSelect(unsigned int * input, int stride, int strideNum, 
                           int blockNum, int k, T minValue, T * output, int* index, int limit)
{
    /* the idx th thread in one vector */
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    /* the idy th vector in one tensor */
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    //use optimization or not
    //int strideNum =(int)strideNumSize;
    //if (isOptimization) strideNum = strideNumSize[idy];

    if (idy >= stride *blockNum) return;

    int maskDesire = 0;
    unsigned int mask = 0x80000000;
    unsigned int desire = 0;
    __shared__ int posCount[32 * 32];
    int tmpK = k;
    int flag = 1;
#pragma unroll
    for (int i = 0; i < 32; i++){
        /* we need to clean the shared memory every loop */

        posCount[idx + blockDim.x*(idy % (512 / WORKERSNUM))] = 0;
        if (flag)
            radixCount(input, stride*strideNum*blockNum, posCount, mask, maskDesire, desire, stride, strideNum, blockNum);
        __syncthreads();
        int sumCount = 0;
#pragma unroll
        for (int j = 0; j < WORKERSNUM; j++) {
            sumCount += posCount[(idy % (512 / WORKERSNUM))*blockDim.x + j];
        }
        __syncthreads();

        if (tmpK<sumCount) {
            /* this position should be 1 */
            desire = mask^desire;
        }
        else {
            /* zoom out the k size,this position should be 0 */
            tmpK = tmpK - sumCount;
            if (tmpK == 0){
                desire = (~(maskDesire >> 1)) | desire;
                /* avoid Synchronize deadlock ,can't use break,so we use flag */
                //break;
                flag = 0;
            }
        }
        maskDesire = mask^maskDesire;
        mask = mask >> 1;
    }
    __syncthreads();

   /* old way to collect number */
   /*
   if (idx == 0)
    {
        unsigned int* uintOutput = new unsigned int;
        int* tmpIndex = new int;
        //*******************something worng***************************
        cudaMalloc((void **)&uintOutput, sizeof(unsigned int)* k);
        cudaMalloc((void **)&tmpIndex, sizeof(unsigned int)*k);
        //*************************************************************
        collectNumberOld(input, limit, k, desire, uintOutput, tmpIndex, stride, strideNum);
        int blockIndex = idy / stride;
        int offsetInBlock = idy% stride;

        for (int i = stride * k * blockIndex + offsetInBlock, j = 0; j < k; j++, i += stride)
        {
            //for(int i = )
            output[i] = deconvert(uintOutput[j]);
            index[i] = tmpIndex[j];
        }
    }
    __syncthreads();
    */

    collectNumber(input, stride, strideNum, limit, desire, output, index, k);
}

/*
get the top-k items along a given dimension
>> a - input tensor
>> b - output tensor (top-k result)
>> index - index of the top-k items
>> dim - the dimension along which the sorting is performed
>> k - how many items returned after sorting
*/
void _CudaTopK(const XTensor * a, XTensor * b, XTensor * index, int dim, int k)
{
    CheckNTErrors((a->unitSize == b->unitSize), "Unmatched input tensors!");
    CheckNTErrors((a->order == b->order), "Unmatched input tensors!");
    CheckNTErrors((index == NULL || a->order == index->order), "Unmatched input tensors!");
    CheckNTErrors((index->dataType == X_INT), "Wrong data type!");
    CheckNTErrors((b->dimSize[dim] == k), "A too large K");

    int stride = 1;
    int blockNum = 1;
    int strideNumA = a->dimSize[dim];
    for (int i = 0; i < dim; i++)
        blockNum *= a->dimSize[i];

    for (int i = dim + 1; i < a->order; i++)
        stride *= a->dimSize[i];

    int workerNum = blockNum < 16 ? 64 : 32; 
    /* adjust the thread num according size of k for fitting the share memory size */
    if (k< 6) workerNum = 512;
    else if (k < 11) workerNum = 256;
    else if (k < 22) workerNum = 128;
    else if (k < 44) workerNum = 64;
    else workerNum = 32;
 
    int cudaGrids[3];
    int cudaBlocks[3];

    GDevs.GetCudaThread2D(a->devID,
        workerNum, stride * blockNum, MAX_INT,
        cudaGrids, cudaBlocks);

    int devIDBackup = 0;
    ProtectCudaDev(a->devID, devIDBackup);

    /* we run the kernel if the heaps can fit into the shared memory */
    cudaGrids[1] *= cudaBlocks[1];
    cudaBlocks[1] = 1;
    if ((cudaBlocks[0] * cudaBlocks[1] + 1) * k * (a->unitSize + sizeof(int)) + (512 * sizeof(int))< SHARED_MEMORY_SIZE) {
        if (a->dataType == DEFAULT_DTYPE) {
            KernelTopK3<DTYPE> <<<dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >>>
                                 ((DTYPE*)a->data, stride, strideNumA, blockNum, k, DTYPE_MIN,
                                 (DTYPE*)b->data, (int*)index->data);
        }
        else {
            ShowNTErrors("TODO!");
        }

    }
    /* we resort to sorting if the data cannot fit inside the shared memory */
    else {
        //int dimSize[MAX_TENSOR_DIM_NUM];
        //memcpy(dimSize, a->dimSize, sizeof(int) * a->order);
        //dimSize[0] = -dimSize[0];
        //XTensor * indexA = new XTensor(a->order, dimSize, X_INT, 1.0F, a->devID, a->mem);
        //indexA->data = a->mem != NULL ? a->mem->AllocBuf(a->devID, a->unitNum * sizeof(int)) : XMemAlloc(a->devID, a->unitNum * sizeof(int));

        /* make the index tensor */
        //SetAscendingOrder(*indexA, dim);

        //_CudaSortBig(a, b, indexA, index, dim, k);

        //if (a->mem != NULL)
        //    a->mem->ReleaseBuf(a->devID, a->unitNum * sizeof(int));
        //delete indexA;
        int workerNum = WORKERSNUM;

        GDevs.GetCudaThread2D(a->devID,
            workerNum, stride * blockNum, MAX_INT,
            cudaGrids, cudaBlocks);
        if (a->dataType == DEFAULT_DTYPE) {
            unsigned int* goutput = (unsigned int *)a->data;
            /* two way all almost the same time to convert data*/
            convert2uintV2 <<<dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >>> ((float*)a->data, goutput, stride, strideNumA, blockNum, strideNumA*blockNum*stride);
            //convert2uintV2 << <dim3(1, stride * blockNum), dim3(512,1) >> >((float*)a->data, goutput, stride, strideNumA, blockNum, strideNumA*blockNum*stride);

            KernelTopKRadixSelect<DTYPE> <<<dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >>> (goutput, stride, strideNumA, blockNum, k, DTYPE_MIN, (DTYPE *)b->data, (int *)index->data, stride * strideNumA * blockNum);
            deconvert2floatV2 <<<dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >>> ((unsigned int *)a->data, (float *)goutput, stride, strideNumA, blockNum, strideNumA*blockNum*stride);
        }
    }

    BacktoCudaDev(a->devID, devIDBackup);
}

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)