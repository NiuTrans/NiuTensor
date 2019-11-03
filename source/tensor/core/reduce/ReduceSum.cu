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
#include "ReduceSum.cuh"

namespace nts{ // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/*
use PTX code to reduce float data
*/
__device__ __forceinline__  
float shflDownReduceSum(float input)
{
    float output;
    asm volatile(
        "{"
        ".reg .f32 r0;"
        "shfl.sync.down.b32  r0, %1, 0x10, 0x1f,0xffffffff;"
        "add.f32        %1, r0, %1;"
        "shfl.sync.down.b32  r0, %1, 0x8, 0xf,0xffffffff;"
        "add.f32        %1, r0, %1;"
        "shfl.sync.down.b32  r0, %1, 0x4, 0x7,0xffffffff;"
        "add.f32        %1, r0, %1;"
        "shfl.sync.down.b32  r0, %1, 0x2, 0x3,0xffffffff;"
        "add.f32        %1, r0, %1;"
        "shfl.sync.down.b32  r0, %1, 0x1, 0x1,0xffffffff;"
        "add.f32        %0, r0, %1;"
        "}"
        : "=f"(output) : "f"(input));
    return output;
}

/*
use PTX code to reduce int data
*/
__device__ __forceinline__  
int shflDownReduceSum(int input)
{
    int output;
    asm volatile(
        "{"
        ".reg .s32 r0;"
        "shfl.sync.down.b32  r0, %1, 0x10, 0x1f,0xffffffff;"
        "add.s32        %1, r0, %1;"
        "shfl.sync.down.b32  r0, %1, 0x8, 0xf,0xffffffff;"
        "add.s32        %1, r0, %1;"
        "shfl.sync.down.b32  r0, %1, 0x4, 0x7,0xffffffff;"
        "add.s32        %1, r0, %1;"
        "shfl.sync.down.b32  r0, %1, 0x2, 0x3,0xffffffff;"
        "add.s32        %1, r0, %1;"
        "shfl.sync.down.b32  r0, %1, 0x1, 0x1,0xffffffff;"
        "add.s32        %0, r0, %1;"
        "}"
        : "=r"(output) : "r"(input));
    return output;
}


/* 
reduce a tensor to another that keeps the sum along a dimension  - slow version
Given a block of data, we go over each dimension i in the stride and we have
sum_i = sum_{0<=j<strideNum} exp(input_{i,j} - shift) if isExp == true;
      = sum_{0<=j<strideNum} input_{i,j} - shift if isExp == false;
where we can view the block as a matrix and input_{i,j} represent the item at the
crossing of the i-th columne and the j-th row.
>> input - the input array (representing a tensor)
>> output - the sum over each block. NOTE: output is also an array
>> stride - stride that we need to move to the next item
>> strideNum - how many strides we need to finish the reduce
>> reducedStrideNum - the number of strides after reducation 
>> blockSize - size of the block (i.e., stride * strideNum)
>> blockNum - how many blocks
>> shift - the bias imposed on the input
>> power - power of the item in the array
>> isExp - specify if we perform exp() on the input
*/
 __global__
void KernelReduceSum(DTYPE * input, DTYPE * output, 
                     int stride, int strideNum, int reducedStrideNum, 
                     int blockSize, int blockNum, 
                     DTYPE * shift, DTYPE power, bool isExp)
{
    __shared__ DTYPE iData[MAX_CUDA_THREAD_NUM_PER_BLOCK * MIN_CUDA_SHARED_MEM_COL_SIZE/2];
    __shared__ DTYPE bias[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    int idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x*blockDim.x + threadIdx.x;

    if(i >= stride * blockNum)
        return;

    if(threadIdx.x == 0)
        bias[threadIdx.y] = shift != NULL ? shift[i] : 0;

    __syncthreads();

    int k = i / stride;
    int iOffset = i % stride;
    bool isValid = (i < stride * blockNum && j < strideNum);

    DTYPE value =  isValid ? input[blockSize * k + stride * j + iOffset] - bias[threadIdx.y] : 0;

    if(power != (DTYPE)1.0){
        if(power == (DTYPE)2.0)
            value = value * value;
        else if(power == (DTYPE)0.5)
            value = sqrt(value);
        else
            value = pow(value, power);
    }

    if(isExp && isValid)
        value = exp(value);

    /* load data into the shared mem */
    iData[threadIdx.y * blockDim.x + threadIdx.x] = value;

    __syncthreads();

    /* do reduction in shared mem */
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1){
        if (threadIdx.x < s)
            iData[idx] += iData[idx + s];

        __syncthreads();
    }
    /* write result for this block to the output array */
    if (threadIdx.x == 0 && blockIdx.x < reducedStrideNum) 
        output[(k * reducedStrideNum + blockIdx.x) * stride + iOffset] = iData[threadIdx.y * blockDim.x];
}

 /* 
reduce a tensor to another that keeps the sum along a dimension  - slow version
This is for float16 reduction.
Given a block of data, we go over each dimension i in the stride and we have
sum_i = sum_{0<=j<strideNum} exp(input_{i,j} - shift) if isExp == true;
      = sum_{0<=j<strideNum} input_{i,j} - shift if isExp == false;
where we can view the block as a matrix and input_{i,j} represent the item at the
crossing of the i-th columne and the j-th row.
>> input - the input array (representing a tensor)
>> output - the sum over each block. NOTE: output is also an array
>> stride - stride that we need to move to the next item
>> strideNum - how many strides we need to finish the reduce
>> reducedStrideNum - the number of strides after reducation 
>> blockSize - size of the block (i.e., stride * strideNum)
>> blockNum - how many blocks
>> shift - the bias imposed on the input
>> power - power of the item in the array
>> isExp - specify if we perform exp() on the input
*/
 __global__
void KernelReduceSum(__half * input, __half * output, 
                     int stride, int strideNum, int reducedStrideNum, 
                     int blockSize, int blockNum, 
                     __half * shift, __half power, bool isExp)
{
    int idx = threadIdx.x * blockDim.y + threadIdx.y;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;

    if(i >= stride * blockNum)
        return;

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    __shared__ __half iData[MAX_CUDA_THREAD_NUM_PER_BLOCK * MIN_CUDA_SHARED_MEM_COL_SIZE/2];
    __shared__ __half bias[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    if(threadIdx.y == 0)
        bias[threadIdx.x] = shift != NULL ? shift[i] : __half(0);
#else
    __shared__ DTYPE iData[MAX_CUDA_THREAD_NUM_PER_BLOCK * MIN_CUDA_SHARED_MEM_COL_SIZE/2];
    __shared__ DTYPE bias[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    if(threadIdx.y == 0)
        bias[threadIdx.x] = shift != NULL ? __half(shift[i]) : __half(0);
#endif

    __syncthreads();

    int k = i / stride;
    int iOffset = i % stride;
    bool isValid = (i < stride * blockNum && j < strideNum);

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    __half value = isValid ? __hsub(input[blockSize * k + stride * j + iOffset], bias[threadIdx.x]) : __half(0);
    DTYPE power2 = __half2float(power);

    if(power2 != (DTYPE)1.0){
        if(power2 == (DTYPE)2.0)
            value = __hmul(value, value);
        else if(power2 == (DTYPE)0.5)
            value = hsqrt(value);
    }

    if(isExp && isValid)
        value = hexp(value);
#else
    DTYPE value =  isValid ? __half2float(input[blockSize * k + stride * j + iOffset]) - __half2float(bias[threadIdx.x]) : 0;
    DTYPE power2 = __half2float(power);

    if(power2 != (DTYPE)1.0){
        if(power2 == (DTYPE)2.0)
            value = value * value;
        else if(power2 == (DTYPE)0.5)
            value = sqrt(value);
        else
            value = pow(value, power2);
    }

    if(isExp && isValid)
        value = exp(value);
#endif

    /* load data into the shared mem */
    iData[threadIdx.x * blockDim.y + threadIdx.y] = value;

    __syncthreads();

    /* do reduction in shared mem */
    for (unsigned int s = blockDim.y/2; s > 0; s >>= 1){
        if (threadIdx.y < s)
            iData[idx] += iData[idx + s];

        __syncthreads();
    }

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    /* write result for this block to the output array */
    if (threadIdx.y == 0 && blockIdx.y < reducedStrideNum) 
        output[(k * reducedStrideNum + blockIdx.y) * stride + iOffset] = iData[threadIdx.x * blockDim.y];
#else
    /* write result for this block to the output array */
    if (threadIdx.y == 0 && blockIdx.y < reducedStrideNum) 
        output[(k * reducedStrideNum + blockIdx.y) * stride + iOffset] = __half(iData[threadIdx.x * blockDim.y]);
#endif

}

/* 
reduce a tensor to another that keeps the sum along a dimension  - fast version
>> input - the input array (representing a tensor)
>> output - the sum over each block. NOTE: output is also an array
>> stride - stride that we need to move to the next item
>> strideNum - how many strides we need to finish the reduce
>> reducedStrideNum - the number of strides after reducation 
>> blockSize - size of the block (i.e., stride * strideNum)
>> blockNum - how many blocks
>> shift - the bias imposed on the input
>> power - power of the item in the array
>> isExp - specify if we perform exp() on the input
*/
template <unsigned int goodSize> __global__
void KernelReduceSumFast(DTYPE * input, DTYPE * output, 
                         int stride, int strideNum, int reducedStrideNum, 
                         int blockSize, int blockNum,
                         DTYPE * shift, DTYPE power, bool isExp)
{
    __shared__ DTYPE iData[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ DTYPE bias[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    unsigned int tid = threadIdx.x;
    unsigned int j = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;

    if(i >= stride * blockNum)
        return;

    if (threadIdx.x == 0)
        bias[threadIdx.y] = shift != NULL ? shift[i] : 0;

    __syncthreads();

    /* first level reduction */
    int k = i / stride;
    int iOffset = i % stride;

    bool isValid = j < strideNum;
    bool isValid2 = j + blockDim.x < strideNum;

    DTYPE * data =  iData + threadIdx.y * blockDim.x;
    DTYPE * inputData = input  + k * blockSize;
    DTYPE value  = isValid ? inputData[j * stride + iOffset] - bias[threadIdx.y]: 0;
    DTYPE value2 = isValid2 ? inputData[(j + blockDim.x) * stride + iOffset] - bias[threadIdx.y]: 0;
    
    if(power != (DTYPE)1.0){
        if(power == (DTYPE)2.0){
            value = value * value;
            value2 = value2 * value2;
        }
        else if(power == (DTYPE)0.5){
            value = sqrt(value);
            value2 = sqrt(value2);
        }
        else{
            value = pow(value, power);
            value2 = pow(value2, power);
        }
    }

    if(isExp){
        if(isValid)
            value = exp(value);
        if(isValid2)
            value2 = exp(value2);
    }

    value = value + value2;

    __syncthreads();
    
    value = shflDownReduceSum(value);
    if ((tid & 0x1f) == 0) 
        data[tid / 32] = value;

    __syncthreads();
    
    if (tid < 32){
        if (tid < blockDim.x / 32)
            value = data[tid];
        else
            value = 0;
        value = shflDownReduceSum(value);

        if (tid == 0 && blockIdx.x < reducedStrideNum) {
            output[(k * reducedStrideNum + blockIdx.x) * stride + iOffset] = value;
        }
    }
}

/* 
reduce a tensor to another that keeps the sum along a dimension  - fast version
This is for float16 reduction
>> input - the input array (representing a tensor)
>> output - the sum over each block. NOTE: output is also an array
>> stride - stride that we need to move to the next item
>> strideNum - how many strides we need to finish the reduce
>> reducedStrideNum - the number of strides after reducation 
>> blockSize - size of the block (i.e., stride * strideNum)
>> blockNum - how many blocks
>> shift - the bias imposed on the input
>> power - power of the item in the array
>> isExp - specify if we perform exp() on the input
*/
template <unsigned int goodSize> __global__
void KernelReduceSumFast(__half * input, __half * output, 
                         int stride, int strideNum, int reducedStrideNum, 
                         int blockSize, int blockNum,
                         __half * shift, __half power, bool isExp)
{
    unsigned int tid = threadIdx.y;
    unsigned int j = blockIdx.y * (blockDim.y * 2) + threadIdx.y;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i >= stride * blockNum)
        return;

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    __shared__ __half iData[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ __half bias[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    if(threadIdx.y == 0)
        bias[threadIdx.x] = shift != NULL ? shift[i] : __float2half(0);
#else
    __shared__ DTYPE iData[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ DTYPE bias[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    if(threadIdx.y == 0)
        bias[threadIdx.x] = shift != NULL ? __half2float(shift[i]) : 0;
#endif

    __syncthreads();

    /* first level reduction */
    int k = i / stride;
    int iOffset = i % stride;
    bool isValid = j < strideNum;
    bool isValid2 = j + blockDim.y < strideNum;

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    __half * data =  iData + threadIdx.x * blockDim.y;
    __half * inputData = input  + k * blockSize;
    __half value  = isValid ? __hsub(inputData[j * stride + iOffset], bias[threadIdx.x]) : __float2half(0);
    __half value2 = isValid2 ? __hsub(inputData[(j + blockDim.y) * stride + iOffset], bias[threadIdx.x]) : __float2half(0);

    DTYPE powerf = __half2float(power);

    if(powerf != (DTYPE)1.0){
        if(powerf == (DTYPE)2.0){
            value = __hmul(value, value);
            value2 = __hmul(value2, value2);
        }
        else if(powerf == (DTYPE)0.5){
            value = hsqrt(value);
            value2 = hsqrt(value2);
        }
    }

    if(isExp){
        if(isValid)
            value = hexp(value);
        if(isValid2)
            value2 = hexp(value2);
    }

#else
    DTYPE * data =  iData + threadIdx.x * blockDim.y;
    __half * inputData = input  + k * blockSize;
    DTYPE value  = isValid ? __half2float(inputData[j * stride + iOffset]) - __half2float(bias[threadIdx.x]): 0;
    DTYPE value2 = isValid2 ? __half2float(inputData[(j + blockDim.y) * stride + iOffset]) - __half2float(bias[threadIdx.x]): 0;

    DTYPE powerf = __half2float(power);

    if(powerf != (DTYPE)1.0){
        if(powerf == (DTYPE)2.0){
            value = value * value;
            value2 = value2 *value2;
        }
        else if(powerf == (DTYPE)0.5){
            value = sqrt(value);
            value2 = sqrt(value2);
        }
        else{
            value = pow(value, powerf);
            value2 = pow(value2, powerf);
        }
    }

    if(isExp){
        if(isValid)
            value = exp(value);
        if(isValid2)
            value2 = exp(value2);
    }
#endif

    /* load data into the shared mem */
    data[tid] = value + value2;

    __syncthreads();

    /* unroll the warp */
    if(goodSize >= 512) {if(tid < 256) {data[tid] += data[tid + 256];} __syncthreads();}
    if(goodSize >= 256) {if(tid < 128) {data[tid] += data[tid + 128];} __syncthreads();}
    if(goodSize >= 128) {if(tid <  64) {data[tid] += data[tid +  64];} __syncthreads();}
    if(goodSize >= 64)  {if(tid <  32) {data[tid] += data[tid +  32];} __syncthreads();}
    if(goodSize >= 32)  {if(tid <  16) {data[tid] += data[tid +  16];} __syncthreads();}
    if(goodSize >= 16)  {if(tid <   8) {data[tid] += data[tid +   8];} __syncthreads();}
    if(goodSize >=  8)  {if(tid <   4) {data[tid] += data[tid +   4];} __syncthreads();}
    if(goodSize >=  4)  {if(tid <   2) {data[tid] += data[tid +   2];} __syncthreads();}
    if(goodSize >=  2)  {if(tid <   1) {data[tid] += data[tid +   1];} __syncthreads();}

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    /* write result for this block to the output array */
    if(threadIdx.y == 0 && blockIdx.y < reducedStrideNum) 
        output[(k * reducedStrideNum + blockIdx.y) * stride  + iOffset] = data[0];
#else
    /* write result for this block to the output array */
    if(threadIdx.y == 0 && blockIdx.y < reducedStrideNum) 
        output[(k * reducedStrideNum + blockIdx.y) * stride  + iOffset] = __float2half(data[0]);
#endif
}

/*
if data storage is discontinuius ,use this way to reduce 
*/
__global__ 
void KernelReduceSumDiscontinuousStorage(DTYPE * input, DTYPE * output, int stride, int strideNum,
                                         int blockNum, DTYPE * shift, DTYPE power, bool isExp)
{
    __shared__ DTYPE bias[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int blockIndex = idx / stride;
    int offsetInBlock = idx % stride; 
    if (idx >= stride * blockNum)
        return;
    bias[idx % blockDim.x] = shift != NULL ? shift[idx] : 0;
    DTYPE ans = 0;

#pragma unroll
    for (int i = stride * strideNum * blockIndex + offsetInBlock;
        i < stride * strideNum * blockIndex + offsetInBlock + stride * strideNum;
        i += stride){
        DTYPE value = input[i];
        value = value - bias[idx % blockDim.x];
        if (power != (DTYPE)1.0) {
            if (power == (DTYPE)2.0) {
                value = value * value;
            }
            else if (power == (DTYPE)0.5) {
                value = sqrt(value);
            }
            else {
                value = pow(value, power);
            }
        }
        if (isExp) {
            value = exp(value);
        }
        ans += value;
    }
    output[idx] = ans;
}

__global__
void KernelReduceSumOp(DTYPE * input, DTYPE * output,
                       int stride, int strideNum, int reducedStrideNum,
                       int blockSize, int blockNum,
                       DTYPE * shift, DTYPE power, bool isExp)
{
    __shared__ DTYPE iData[MAX_CUDA_THREAD_NUM_PER_BLOCK / 32];
    __shared__ DTYPE bias[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    unsigned int tid = threadIdx.y;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= stride * blockNum)
        return;

    if (threadIdx.y == 0)
        bias[threadIdx.x] = shift != NULL ? shift[i] : 0;

    __syncthreads();

    /* first level reduction */
    int k = i / stride;
    int iOffset = i % stride;

    DTYPE threadSum = 0;

    DTYPE * data = iData + threadIdx.x * blockDim.y;
    DTYPE * inputData = input + k * blockSize;
    for (int it = j; it < strideNum; it += blockDim.y){
        DTYPE value = inputData[it * stride + iOffset] - bias[threadIdx.x];
        if (power != (DTYPE)1.0) {
            if (power == (DTYPE)2.0) {
                value = value * value;
            }
            else if (power == (DTYPE)0.5) {
                value = sqrt(value);
            }
            else {
                value = pow(value, power);
            }
        }
        if (isExp) value = exp(value);
        threadSum += value;
    }
    __syncthreads();
    threadSum = shflDownReduceSum(threadSum);
    if ((tid & 0x1f) == 0) { data[tid / 32] = threadSum; }
    __syncthreads();
    if (tid < 32){
        if (tid < blockDim.y / 32)
            threadSum = data[tid];
        else 
            threadSum = 0;
        threadSum = shflDownReduceSum(threadSum);
        if (tid == 0 && blockIdx.y < reducedStrideNum)
            output[(k * reducedStrideNum + blockIdx.y) * stride + iOffset] = threadSum;
    }

}

__global__
void KernelReduceSumOpLessBlocks(DTYPE * input, DTYPE * output,
                                 int strideNum, int blockNum,
                                 DTYPE * shift, DTYPE power, bool isExp)
{
    __shared__ DTYPE bias[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    int idx = threadIdx.x % 32;
    int idy = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    if (idx == 0)
        bias[threadIdx.x / 32] = shift != NULL ? shift[idy] : 0;

    int startIndex = idy * strideNum;
    DTYPE threadSum = 0;
    for (int i = idx; i < strideNum; i += 32) {
        DTYPE value = input[startIndex + i] - bias[threadIdx.x / 32];
        if (power != (DTYPE)1.0) {
            if (power == (DTYPE)2.0) {
                value = value * value;
            }
            else if (power == (DTYPE)0.5) {
                value = sqrt(value);
            }
            else {
                value = pow(value, power);
            }
        }
        if (isExp) value = exp(value);
        threadSum += value;
    }
    threadSum = shflDownReduceSum(threadSum);
    if (idx == 0)
        output[idy] = threadSum;
}

/*
according the GPU's sm number allocation warp num
*/
inline void continuousStorageThreadAllocation(dim3& grid, dim3& block, long long vectorNum, int vectorSize)
{
    int warpNum = 4;
    if (vectorNum < 20 * 8) {
        warpNum = 8;
        if (vectorNum < 20 * 4) {
            warpNum = 16;
            if (warpNum < 20 * 2)
                warpNum = 32;
        }
    }
    int minWarpNum = vectorSize / 32;
    if (vectorSize % 32 != 0) minWarpNum++;
    warpNum = min(warpNum, minWarpNum);

    grid.x = (unsigned int)vectorNum;
    grid.y = 1;
    grid.z = 1;
    block.x = 1;
    block.y = warpNum * 32;
    block.z = 1;
}

/* 
this situation we use block.x * grid.x deal one vector for continuous read
*/
void discontinuousStorageNoShareMemThreadAllocation(dim3* grid, dim3* block, int stride, int blockNum)
{
    block->x = 512;
    block->y = 1;
    if ((stride * blockNum) % 512 == 0)
        grid->x = (stride * blockNum) / 512;
    else
        grid->x = (stride * blockNum) / 512 + 1;
    grid->y = 1;
}

/*
adjust threads.x number then we can use warp optimization
*/
void adjustThreadForUseWarpOptimization(dim3* blocks, dim3* threads)
{
    if (threads->y > 1){
        blocks->y *= threads->y;
        threads->y = 1;
    }
    if (threads->x < 32)
        threads->x = 32;
}

/* 
sum the items along a dimension of the tensor (cuda version). 
For a 1-dimensional data array a,
sum = \sum_i (a_i - shift)^power if isExp == false
sum = \sum_i exp((a_i - shift)^power) if isExp == true
>> input - the input tensor
>> output - the output tensor
>> dim - which dimension to reduce
>> shift - the bias on the input
>> power - we perform pow(item_i, power) on each item
>> ieExp - specify if the exp() is performed
*/
void _CudaReduceSum(const XTensor * input, XTensor * output, int dim, const XTensor * shift, DTYPE power, bool isExp)
{
    CheckNTErrors(input && output, "Empty input or output tensors!");
    CheckNTErrors(input->order == output->order + 1, "Incorrect tensor sizes!");
    CheckNTErrors(input->order > dim && dim >= 0, "Illegal dimension to reduce!");
    CheckNTErrors(input->dataType == output->dataType, "Unmatched data types!");
    CheckNTErrors(shift == NULL || output->unitNum == shift->unitNum, "Incorrect shift tensor size!");

    for(int i = 0; i < input->order; i++){
        if(i < dim){
            CheckNTErrors(input->dimSize[i] == output->dimSize[i], "Unmatched tensors!");
        }
        else if(i > dim){
            CheckNTErrors(input->dimSize[i] == output->dimSize[i - 1], "Unmatched tensors!");
        }
    }

    if(input->dataType == X_FLOAT16)
        CheckNTErrors(power == 0 || power == 0.5 || power == 1.0 || power == 2.0, "TODO!");

    int cudaGridSize[3];
    int cudaBlockSize[3];
    int iter = 0;
    int stride = 1;
    int strideNum = input->dimSize[dim];
    int blockSize = 1;
    int blockNum = 1;

    for (int i = 0; i < input->order; i++) {
        if (i < dim)
            blockNum *= input->dimSize[i];
        else if (i > dim)
            stride *= input->dimSize[i];
    }
    blockSize = stride * strideNum;

    int devID = input->devID;
    int devIDBackup;
    ProtectCudaDev(devID, devIDBackup);

    DTYPE * sp = shift != NULL ? (DTYPE*)shift->data : NULL;

    if (stride == 1 && blockNum >= 10) {
        dim3 grids;
        dim3 blocks;
        continuousStorageThreadAllocation(grids, blocks, (long long)blockNum, strideNum);
        if (blocks.y >= 128)
            KernelReduceSumOp <<<grids, blocks>>> ((DTYPE *)input->data, (DTYPE*)output->data, stride, 
                                                    strideNum, grids.y, blockSize, blockNum, sp, power, isExp);
        else {
            if (blockNum % 4 != 0) 
                blockNum = (int)(blockNum / 4) + 1;
            else 
                blockNum = blockNum / 4;
            KernelReduceSumOpLessBlocks <<<blockNum, 128>>> ((DTYPE *)input->data, (DTYPE*)output->data, 
                                                              strideNum, blockNum, sp, power, isExp);
        }
    }
    else if (stride != 1 && stride * blockNum > 4096) {
        //GDevs->GetGridAndBlockSize2D(devID, stride * blockNum, strideNum,MAX_INT, cudaGridSize, cudaBlockSize);
        //unsigned int* goutput = (unsigned int *)input->data;
        //convert2uintV2 << <dim3(cudaGridSize[0], cudaGridSize[1]), dim3(cudaBlockSize[0], cudaBlockSize[1]) >> > ((float*)input->data, goutput, stride, strideNum, blockNum, strideNum*blockNum*stride);
        dim3 grid, block;
        discontinuousStorageNoShareMemThreadAllocation(&grid, &block, stride, blockNum);
        KernelReduceSumDiscontinuousStorage <<<grid, block>>> ((DTYPE *)input->data, (DTYPE*)output->data, stride, 
                                                                strideNum, blockNum,sp, power, isExp);
    }
    else {
        XMem * mem = input->mem;

        GDevs.GetCudaThread2D(devID, strideNum, stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);

        int bufSize = input->unitSize * cudaGridSize[0] * stride * blockNum * 2;
        DTYPE * buf  = mem != NULL ? (DTYPE*)mem->AllocBuf(mem->devID, bufSize) : (DTYPE*)XMemAlloc(devID, bufSize);
        DTYPE * buf1 = buf;
        DTYPE * buf2 = buf + cudaGridSize[0] * stride * blockNum;
        do {
            if (input->dataType == DEFAULT_DTYPE) {
                DTYPE * iData = NULL;
                DTYPE * oData = NULL;
                if (iter == 0) {
                    iData = (DTYPE*)input->data;
                    oData = buf1;
                }
                else if (iter % 2 == 1) {
                    iData = buf1;
                    oData = buf2;
                }
                else {
                    iData = buf2;
                    oData = buf1;
                }
                /* unroll the reduction procedure. The code is messy but it is faster. */
                if (strideNum <= 32) {
                    GDevs.GetCudaThread2D(devID, strideNum, stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                    dim3 blocks(cudaGridSize[0], cudaGridSize[1]), threads(cudaBlockSize[0], cudaBlockSize[1]);
                    if (cudaGridSize[0] == 1)
                        oData = (DTYPE*)output->data;
                    KernelReduceSum <<<blocks, threads>>> (iData, oData, stride, strideNum, blocks.x, 
                                                           blockSize, blockNum, sp, power, isExp);
                }
                else if (strideNum < 128) {
                    GDevs.GetCudaThread2D(devID, MAX(strideNum / 2 + 1, 64), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                    dim3 blocks(cudaGridSize[0], cudaGridSize[1]), threads(cudaBlockSize[0], cudaBlockSize[1]);
                    if (cudaGridSize[0] == 1)
                        oData = (DTYPE*)output->data;
                    CheckNTErrors((cudaBlockSize[0] >= 64), "Incorrect thread number when calling the cuda kernel!");
                    adjustThreadForUseWarpOptimization(&blocks, &threads);
                    KernelReduceSumFast<64> <<<blocks, threads>>> (iData, oData, stride, strideNum, blocks.x, 
                                                                   blockSize, blockNum, sp, power, isExp);
                }
                else if (strideNum < 256) {
                    GDevs.GetCudaThread2D(devID, MAX(strideNum / 2 + 1, 128), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                    dim3 blocks(cudaGridSize[0], cudaGridSize[1]), threads(cudaBlockSize[0], cudaBlockSize[1]);
                    if (cudaGridSize[0] == 1)
                        oData = (DTYPE*)output->data;
                    CheckNTErrors((cudaBlockSize[0] >= 128), "Incorrect thread number when calling the cuda kernel!");
                    adjustThreadForUseWarpOptimization(&blocks, &threads);
                    KernelReduceSumFast<128> <<<blocks, threads>>> (iData, oData, stride, strideNum, blocks.x, 
                                                                    blockSize, blockNum, sp, power, isExp);
                }
                else if (strideNum < 512) {
                    GDevs.GetCudaThread2D(devID, MAX(strideNum / 2 + 1, 256), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                    dim3 blocks(cudaGridSize[0], cudaGridSize[1]), threads(cudaBlockSize[0], cudaBlockSize[1]);
                    if (cudaGridSize[0] == 1)
                        oData = (DTYPE*)output->data;
                    CheckNTErrors((cudaBlockSize[0] >= 256), "Incorrect thread number when calling the cuda kernel!");
                    adjustThreadForUseWarpOptimization(&blocks, &threads);
                    KernelReduceSumFast<256> <<<blocks, threads>>> (iData, oData, stride, strideNum, blocks.x, 
                                                                    blockSize, blockNum, sp, power, isExp);
                }
                else {
                    GDevs.GetCudaThread2D(devID, MAX(strideNum / 2 + 1, 512), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                    dim3 blocks(cudaGridSize[0], cudaGridSize[1]), threads(cudaBlockSize[0], cudaBlockSize[1]);
                    if (cudaGridSize[0] == 1)
                        oData = (DTYPE*)output->data;
                    CheckNTErrors((cudaBlockSize[0] >= 512), "Incorrect thread number when calling the cuda kernel!");
                    adjustThreadForUseWarpOptimization(&blocks, &threads);
                    KernelReduceSumFast<512> <<<blocks, threads>>> (iData, oData, stride, strideNum, blocks.x, 
                                                                    blockSize, blockNum, sp, power, isExp);
                }
            }
            else if (input->dataType == X_FLOAT16) {
                __half * buf1ft16 = (__half *)buf1;
                __half * buf2ft16 = (__half *)buf2;
                __half * spft16 = (__half *)sp;
                unsigned short power2 = FloatToFloat16(power);
                __half * powerft16p = (__half*)&power2;
                __half * iData = NULL;
                __half * oData = NULL;
                if (iter == 0) {
                    iData = (__half*)input->data;
                    oData = buf1ft16;
                }
                else if (iter % 2 == 1) {
                    iData = buf1ft16;
                    oData = buf2ft16;
                }
                else {
                    iData = buf2ft16;
                    oData = buf1ft16;
                }

                /* unroll the reduction procedure. The code is messy but it is faster. */
                if (strideNum < 32) {
                    GDevs.GetCudaThread2D(devID, strideNum, stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                    dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);
                    if (cudaGridSize[0] == 1)
                        oData = (__half*)output->data;
                    KernelReduceSum <<<blocks, threads>>> (iData, oData, stride, strideNum, blocks.y, 
                                                           blockSize, blockNum, spft16, *powerft16p, isExp);
                }
                else if (strideNum < 128) {
                    GDevs.GetCudaThread2D(devID, MAX(strideNum / 2 + 1, 64), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                    dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);
                    if (cudaGridSize[0] == 1)
                        oData = (__half*)output->data;
                    CheckNTErrors((cudaBlockSize[0] >= 64), "Incorrect thread number when calling the cuda kernel!");
                    KernelReduceSumFast<64> <<<blocks, threads>>> (iData, oData, stride, strideNum, blocks.y, 
                                                                   blockSize, blockNum, spft16, *powerft16p, isExp);
                }
                else if (strideNum < 256) {
                    GDevs.GetCudaThread2D(devID, MAX(strideNum / 2 + 1, 128), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                    dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);
                    if (cudaGridSize[0] == 1)
                        oData = (__half*)output->data;
                    CheckNTErrors((cudaBlockSize[0] >= 128), "Incorrect thread number when calling the cuda kernel!");
                    KernelReduceSumFast<128> <<<blocks, threads>>> (iData, oData, stride, strideNum, blocks.y, 
                                                                    blockSize, blockNum, spft16, *powerft16p, isExp);
                }
                else if (strideNum < 512) {
                    GDevs.GetCudaThread2D(devID, MAX(strideNum / 2 + 1, 256), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                    dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);
                    if (cudaGridSize[0] == 1)
                        oData = (__half*)output->data;
                    CheckNTErrors((cudaBlockSize[0] >= 256), "Incorrect thread number when calling the cuda kernel!");
                    KernelReduceSumFast<256> <<<blocks, threads>>> (iData, oData, stride, strideNum, blocks.y, 
                                                                    blockSize, blockNum, spft16, *powerft16p, isExp);
                }
                else {
                    GDevs.GetCudaThread2D(devID, MAX(strideNum / 2 + 1, 512), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
                    dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);
                    if (cudaGridSize[0] == 1)
                        oData = (__half*)output->data;
                    CheckNTErrors((cudaBlockSize[0] >= 512), "Incorrect thread number when calling the cuda kernel!");
                    KernelReduceSumFast<512> <<<blocks, threads>>> (iData, oData, stride, strideNum, blocks.y, 
                                                                    blockSize, blockNum, spft16, *powerft16p, isExp);
                }
            }

            strideNum = cudaGridSize[0];
            blockSize = cudaGridSize[0];
            sp = NULL;
            power = (DTYPE)1.0;
            isExp = false;

            iter++;

        } while (strideNum > 1);
        

        if (mem != NULL)
            mem->ReleaseBuf(mem->devID, bufSize);
        else
            XMemFree(devID, buf);
    }

    BacktoCudaDev(devID, devIDBackup);
}

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)