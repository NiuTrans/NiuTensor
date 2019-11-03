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
__global__
void KernelGather(DTYPE * sData, DTYPE * tData, int * sIndex, int indexSize, int stride)
{
    __shared__ DTYPE * sp[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ DTYPE * tp[MAX_CUDA_THREAD_NUM_PER_BLOCK];

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

    DTYPE * s = sp[threadIdx.x];
    DTYPE * t = tp[threadIdx.x];

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
__global__
void KernelGather(DTYPE * sData, DTYPE * tData, int * sIndex, int stride, int strideNum, int blockNum)
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
        tData[i] = sData[sIndex[i]];
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

    DTYPE * sData = (DTYPE*)s->data;
    DTYPE * tData = (DTYPE*)t->data;

    int * sIndex = NULL;
    
    if (srcIndex->devID < 0) {
        sIndex = mem != NULL ? 
                  (int*)mem->AllocBuf(mem->devID, sizeof(int) * indexSize) : 
                  (int*)XMemAlloc(mem->devID, sizeof(int) * indexSize);
        XMemCopy(sIndex, devID, srcIndex, -1, sizeof(int) * indexSize);
    }
    else
        sIndex = (int *)srcIndex->data;

    KernelGather<<<blocks, threads >>>(sData, tData, sIndex, indexSize, stride);

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
    for (int i = 0; i < dim; i++)
        blockNum *= srcIndex->dimSize[i];
    for (int i = dim + 1; i < srcIndex->order; i++)
        stride *= srcIndex->dimSize[i];

    int * sIndex = NULL;
    if (srcIndex->devID < 0) {
        sIndex = mem != NULL ?
            (int*)mem->AllocBuf(mem->devID, sizeof(int) * indexSize) :
            (int*)XMemAlloc(mem->devID, sizeof(int) * indexSize);
        XMemCopy(sIndex, devID, srcIndex, -1, sizeof(int) * indexSize);
    }
    else
        sIndex = (int *)srcIndex->data;

    int cudaGrids[3];
    int cudaBlocks[3];
    GDevs.GetCudaThread2D(devID, max(32, strideNum), stride*blockNum, MAX_INT, cudaGrids, cudaBlocks);

    KernelGather << <dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >> > ((DTYPE *)s->data, (DTYPE *)t->data, sIndex, stride, strideNum, blockNum);
}
#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)