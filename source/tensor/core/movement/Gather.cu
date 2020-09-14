/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2017, Natural Language Processing Lab, Northeastern University.
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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-11-27
 */

#include "Gather.cuh"
#include "CopyBlocksSelected.cuh"
#include "../../XDevice.h"
#include "../../XUtility.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/*
gather indexed sub-tensors(cuda version)

>> sData - the data pointer of the source tensor
>> tData - the data pointer of the target tensor
>> sIndex - the index of the source tensor
>> indexSize - the size of the srcIndex
>> stride - stride of a data block
*/
template <class T>
__global__
void KernelGather(T * sData, T * tData, int * sIndex, int indexSize, int stride)
{
    __shared__ T * sp[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ T * tp[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    /* block id */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* offset in each block */
    int offset = blockDim.y * blockIdx.y + threadIdx.y;

    if(i >= indexSize || offset >= stride)
        return;

    if(threadIdx.y == 0){
        sp[threadIdx.x] = sData + sIndex[i] * stride;
        tp[threadIdx.x] = tData + i * stride;
    }

    __syncthreads();

    T * s = sp[threadIdx.x];
    T * t = tp[threadIdx.x];

    t[offset] = s[offset];
}

/*
gather indexed sub-tensors(cuda version)

>> sData - the data pointer of the source tensor
>> tData - the data pointer of the target tensor
>> sIndex - the index of the source tensor
>> indexSize - the size of the srcIndex
>> stride - stride of a data block
>> strideNum - strideNum of a data block
>> blockNum - block size of data
*/
template <class T>
__global__
void KernelGather(T * sData, T * tData, int * sIndex, int stride, int strideNum, int blockNum, int srcStrideNum)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int blockIndex = idy / stride;
    int offsetInBlock = idy % stride;

    int size = stride * strideNum * blockNum;  

#pragma unroll
    for (int i = idx * stride + stride * strideNum * blockIndex + offsetInBlock;
        i < stride * strideNum * blockIndex + offsetInBlock + stride * strideNum && i < size;
        i += stride * blockDim.x) {
        tData[i] = sData[sIndex[i] * stride + stride * srcStrideNum * blockIndex + offsetInBlock];
    }
}

/*
gather indexed sub-tensors(cuda version)

>> s - the source tensor
>> t - the target tensor
>> srcIndex - the tensor to save the index of the source tensor
*/
void _CudaGather(const XTensor * s, XTensor * t, XTensor * srcIndex)
{
    int devID = s->devID;
    XMem * mem = s->mem;

    int stride = s->GetDim(1);
    int indexSize = srcIndex->unitNum;

    int cudaGrids[3];
    int cudaBlocks[3];

    int devIDBackup;
    ProtectCudaDev(devID, devIDBackup);

    GDevs.GetCudaThread2D(devID, indexSize, stride, MAX_INT, cudaGrids, cudaBlocks);

    dim3 blocks(cudaGrids[0], cudaGrids[1]);
    dim3 threads(cudaBlocks[0], cudaBlocks[1]);

    int * sIndex = NULL;
    
    if (srcIndex->devID < 0) {
        int * sIndexData = (int*)srcIndex->data;
        for (int i = 0; i < indexSize; i++) {
            int srcIndexValue = sIndexData[i] * stride;
            CheckNTErrors(srcIndexValue < s->unitNum, "Wrong index!");
        }

        sIndex = mem != NULL ? 
                  (int*)mem->AllocBuf(mem->devID, sizeof(int) * indexSize) : 
                  (int*)XMemAlloc(mem->devID, sizeof(int) * indexSize);
        XMemCopy(sIndex, devID, srcIndex, -1, sizeof(int) * indexSize);
    }
    else {
        int * sIndexData = new int[sizeof(int) * indexSize];
        XMemCopy(sIndexData, -1, srcIndex->data, srcIndex->devID, sizeof(int) * indexSize);
        for (int i = 0; i < indexSize; i++) {
            int srcIndexValue = sIndexData[i] * stride;
            CheckNTErrors(srcIndexValue < s->unitNum, "Wrong index!");
        }

        sIndex = (int *)srcIndex->data;

        delete[] sIndexData;
    }

    if (s->dataType == X_FLOAT && t->dataType == X_FLOAT) {
        DTYPE * sData = (float*)s->data;
        DTYPE * tData = (float*)t->data;
        KernelGather<<<blocks, threads>>>(sData, tData, sIndex, indexSize, stride);
    }
    else if (s->dataType == X_FLOAT16 && t->dataType == X_FLOAT16) {
        half * sData = (half*)s->data;
        half * tData = (half*)t->data;
        KernelGather<<<blocks, threads>>>(sData, tData, sIndex, indexSize, stride);
    }
    else if (s->dataType == X_INT && t->dataType == X_INT) {
        int * sData = (int*)s->data;
        int * tData = (int*)t->data;
        KernelGather<<<blocks, threads>>>(sData, tData, sIndex, indexSize, stride);
    }
    else {
        ShowNTErrors("Unsupported dataType!");
    }

    if (srcIndex->devID < 0) {
        if(mem != NULL)
            mem->ReleaseBuf(mem->devID, sizeof(int) * indexSize);
        else
            XMemFree(mem->devID, sIndex);
    }

    BacktoCudaDev(devID, devIDBackup);
}

/*
gather indexed sub-tensors(cuda version)

>> s - the source tensor
>> t - the target tensor
>> srcIndex - the tensor to save the index of the source tensor
>> dim - the leading dimension to define "sub-tensors"
*/
void _CudaGather(const XTensor * s, XTensor * t, XTensor * srcIndex, int dim)
{
    int devID = srcIndex->devID;
    XMem * mem = s->mem;

    int stride = 1;
    int blockNum = 1;
    int indexSize = srcIndex->unitNum;
    int strideNum = srcIndex->dimSize[dim];
    int srcStrideNum = s->dimSize[dim];
    for (int i = 0; i < dim; i++)
        blockNum *= srcIndex->dimSize[i];
    for (int i = dim + 1; i < srcIndex->order; i++)
        stride *= srcIndex->dimSize[i];

    int * sIndex = NULL;
    if (srcIndex->devID < 0) {
        int * sIndexData = (int*)srcIndex->data;
        for (int i = 0; i < indexSize; i++) {
            int srcIndexValue = sIndexData[i] * stride;
            CheckNTErrors(srcIndexValue < s->unitNum, "Wrong index!");
        }

        sIndex = mem != NULL ?
                  (int*)mem->AllocBuf(mem->devID, sizeof(int) * indexSize) :
                  (int*)XMemAlloc(mem->devID, sizeof(int) * indexSize);
        XMemCopy(sIndex, devID, srcIndex, -1, sizeof(int) * indexSize);
    }
    else {
        int * sIndexData = new int[sizeof(int) * indexSize];
        XMemCopy(sIndexData, -1, srcIndex->data, srcIndex->devID, sizeof(int) * indexSize);
        for (int i = 0; i < indexSize; i++) {
            int srcIndexValue = sIndexData[i] * stride;
            CheckNTErrors(srcIndexValue < s->unitNum, "Wrong index!");
        }

        sIndex = (int *)srcIndex->data;
	   delete[] sIndexData;
    }

    int cudaGrids[3];
    int cudaBlocks[3];
    GDevs.GetCudaThread2D(devID, max(32, strideNum), stride*blockNum, MAX_INT, cudaGrids, cudaBlocks);
    if(s->dataType == X_FLOAT){
        KernelGather << <dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >> > ((float *)s->data, (float *)t->data, sIndex, stride, strideNum, blockNum, srcStrideNum);
    }
    else if(s->dataType == X_FLOAT16){
        KernelGather << <dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >> > ((half *)s->data, (half *)t->data, sIndex, stride, strideNum, blockNum, srcStrideNum);
    }
    else {
        ShowNTErrors("Unsupported dataType!");
    }
}
#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)