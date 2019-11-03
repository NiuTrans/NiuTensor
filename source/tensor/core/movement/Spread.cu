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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-09-25
 */

#include "../../XTensor.h"
#include "../../XDevice.h"
#include "../../XUtility.h"
#include "Spread.cuh"
#include "CopyValues.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* 
This is core assignment for spread function.

>> sData - the data pointer of the source tensor
>> cData - the data pointer of collection tensor
>> blockNum - the number of data blocks
>> blockSizeSrc - the size of source data block
>> blockSizeColl - the size of source data block
>> stride - the stride of a data block
*/
__global__
void KernelSpread(DTYPE * sData, DTYPE * cData,  int blockNum, 
                  int blockSizeSrc, int blockSizeColl, int stride)
{
    /* block id */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* offset in each block */
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    
    if(i >= blockNum || j >= stride)
        return;

    DTYPE * s = sData + blockSizeSrc * i;
    DTYPE * c = cData + blockSizeColl * i;

    s[j] = c[j];
}

/* 
This is core assignment for spread function.

>> sData - the data pointer of the source tensor
>> cData - the data pointer of collection tensor
>> blockNum - number of data blocks
>> blockSizeSrc - size of source data block
>> blockSizeColl - size of source data block
>> stride - stride of a data block
>> subtensorNum - number of sub-tensors
>> srcIndex - index of the source sub-tensor
>> colIndex - index of the sub-tensor in the collection tensor
*/
__global__
void KernelSpreadFuzed(DTYPE * sData, DTYPE * cData,  int blockNum, 
                       int blockSizeSrc, int blockSizeColl, int stride,
                       int subtensorNum,
                       int * srcIndex, int * colIndex)
{
    __shared__ DTYPE * sp[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ DTYPE * cp[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    /* block id */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* offset in each block */
    int offset = blockDim.y * blockIdx.y + threadIdx.y;

    int blockId = i % blockNum;
    int subtensorId = i / blockNum;
    
    if(subtensorId >= subtensorNum || offset >= stride)
        return;

    if(threadIdx.y == 0){
        sp[threadIdx.x] = sData + srcIndex[subtensorId] * stride;
        cp[threadIdx.x] = cData + colIndex[subtensorId] * stride;
    }

    __syncthreads();

    DTYPE * s = sp[threadIdx.x] + blockSizeSrc * blockId;
    DTYPE * c = cp[threadIdx.x] + blockSizeColl * blockId;

    s[offset] = c[offset];
}

/*
spread a collection tensor to source tensor (cuda version).
This is a inverse operation compared to gather.

>> source - the source tensor whose data would be modified
>> collection - the collection whose data would be spread to source tensor
>> dim - the leading dimension to define "sub-tensors"
         e.g., for a tensor of size (3, 2, 4) and dim = 2, 
         we have 4 sub-tensors of size (3, 2)
>> srcIndex - index of the source sub-tensors
>> indexSize - length of srcIndex (and collIndex)
>> collIndex - index of the gathered sub-tensors
*/
void _CudaSpread(XTensor * source, XTensor * collection, int dim, 
                 int * srcIndex, int indexSize, int * collIndex)
{
    int order = source->order;

    CheckNTErrors(source->dataType == DEFAULT_DTYPE, "TODO!");
    CheckNTErrors(dim >= 0 && dim < order, "Illegal dimension!");
    
    int blockSizeSrc = 1;
    int blockSizeColl = 1;
    int blockNum = 1;
    int stride = 1;

    for (int i = dim + 1; i < order; i++) {
        stride *= source->GetDim(i);
    }
    
    blockSizeSrc = stride * source->GetDim(dim);
    blockSizeColl = stride * collection->GetDim(dim);
    blockNum = source->unitNum / blockSizeSrc;

    int cudaGrids[3];
    int cudaBlocks[3];

    GDevs.GetCudaThread2D(source->devID, blockNum, stride, MAX_INT, cudaGrids, cudaBlocks);

    dim3 blocks(cudaGrids[0], cudaGrids[1]);
    dim3 threads(cudaBlocks[0], cudaBlocks[1]);

    int devIDBackup;
    ProtectCudaDev(source->devID, devIDBackup);
    
    if(indexSize < 4){
        GDevs.GetCudaThread2D(source->devID, blockNum, stride, MAX_INT, cudaGrids, cudaBlocks);

        dim3 blocks(cudaGrids[0], cudaGrids[1]);
        dim3 threads(cudaBlocks[0], cudaBlocks[1]);
    
        DTYPE * sData = (DTYPE*)source->data;
        DTYPE * cData = (DTYPE*)collection->data;
        for(int i = 0; i < indexSize; i++) {
            int src = srcIndex[i];
            int tgt = collIndex[i];
            DTYPE * s = sData + src * stride;
            DTYPE * c = cData + tgt * stride;

            KernelSpread<<<blocks, threads >>>(s, c, blockNum, blockSizeSrc, blockSizeColl, stride);
        }
    }
    else{
        GDevs.GetCudaThread2D(source->devID, blockNum * indexSize, stride, MAX_INT, cudaGrids, cudaBlocks);

        dim3 blocks(cudaGrids[0], cudaGrids[1]);
        dim3 threads(cudaBlocks[0], cudaBlocks[1]);

        DTYPE * s = (DTYPE*)source->data;
        DTYPE * c = (DTYPE*)collection->data;

        XMem * mem = source->mem;
        int * si = mem != NULL ? 
                   (int*)mem->AllocBuf(mem->devID, sizeof(int) * indexSize * 2) : 
                   (int*)XMemAlloc(mem->devID, sizeof(int) * indexSize * 2);
        int * ci = si + indexSize;

        XMemCopy(si, mem->devID, srcIndex, -1, sizeof(int) * indexSize);
        XMemCopy(ci, mem->devID, collIndex, -1, sizeof(int) * indexSize);

        KernelSpreadFuzed<<<blocks, threads >>>(s, c, blockNum, blockSizeSrc, blockSizeColl,
                                                stride, indexSize, si, ci);

        if(mem != NULL)
            mem->ReleaseBuf(mem->devID, sizeof(int) * indexSize * 2);
        else
            XMemFree(mem->devID, si);
    }
    
    BacktoCudaDev(source->devID, devIDBackup);
}

/*
spread a collection tensor to source tensor (kernel version).
And this is a special spread function for backward computation of CopyIndexed function.

>> sData - the data pointer of the source tensor
>> cData - the data pointer of collection tensor
>> sIndex - index of the source sub-tensor
>> cIndex - index of the sub-tensor in the collection tensor
>> blockNum - number of data blocks
>> blockSizeSrc - size of source data block
>> blockSizeColl - size of source data block
>> stride - stride of a data block
>> indexSize - number of indexs
>> copyNum - number of the sub-tensors we copy for each source index
*/
__global__
void KernelSpreadForCopyIndexed(DTYPE * sData, DTYPE * cData, int * sIndex, int * cIndex,
                                int blockNum, int blockSizeSrc, int blockSizeColl,
                                int stride, int indexSize, int copyNum)
{
    __shared__ DTYPE * sp[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ DTYPE * cp[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    /* block id */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* offset in each block */
    int offset = blockDim.y * blockIdx.y + threadIdx.y;

    int realIndexSize = indexSize * copyNum;

    int realBlockNum = i / realIndexSize;
    int tmp = i % realIndexSize;
    int realIndex = tmp / copyNum;
    int realCopyNum = tmp % copyNum;

    if (realBlockNum >= blockNum || offset >= stride || realIndex >= indexSize || realCopyNum >= copyNum)
        return;

    //if(i >= blockNum * indexSize * copyNum || offset >= stride)
    //    return;

    int realSrcIndex = sIndex[realIndex] + realCopyNum;
    int realCollIndex = cIndex[realIndex] + realCopyNum;

    //int realSrcIndex = sIndex[realIndex / copyNum] + realIndex % copyNum;
    //int realCollIndex = cIndex[realIndex / copyNum] + realIndex % copyNum;

    if(threadIdx.y == 0){
        sp[threadIdx.x] = sData + realBlockNum * blockSizeSrc + realSrcIndex * stride;
        cp[threadIdx.x] = cData + realBlockNum * blockSizeColl + realCollIndex * stride;
    }

    __syncthreads();

    DTYPE * s = sp[threadIdx.x];
    DTYPE * c = cp[threadIdx.x];

    atomicAdd(s + offset, c[offset]);

}

/*
spread a collection tensor to source tensor.
And this is a special spread function for backward computation of CopyIndexed function.

>> s - the source tensor whose data would be modified
>> c - the collection whose data would be spread to source tensor
>> dim - the leading dimension to define "sub-tensors"
         e.g., for a tensor of size (3, 2, 4) and dim = 2, 
         we have 4 sub-tensors of size (3, 2)
>> srcIndex - the tensor to save the index of the source sub-tensors
>> collIndex - the tensor to save the index of the collection sub-tensors
>> copyNum - number of the sub-tensors we copy for each source index, 
             e.g., for srcIndex = [1,4] and copyNum = 2,
             we actually copy the source sub-tensors 1, 2, 4, 5
*/
void _CudaSpreadForCopyIndexed(XTensor * s, XTensor * c, int dim, 
                               XTensor * srcIndex, XTensor * collIndex, 
                               int copyNum)
{
    int devID = s->devID;
    int order = s->order;
    int indexSize = srcIndex->unitNum;

    int blockNum = 1;
    int stride = 1;
    int blockSizeSrc = 1;
    int blockSizeTgt = 1;

    for (int i = 0; i < dim; i++)
        blockNum *= s->GetDim(i);
    
    for (int i = dim + 1; i < order; i++)
        stride *= s->GetDim(i);

    blockSizeSrc = stride * s->GetDim(dim);
    blockSizeTgt = stride * c->GetDim(dim);

    int cudaGrids[3];
    int cudaBlocks[3];

    int devIDBackup;
    ProtectCudaDev(devID, devIDBackup);

    GDevs.GetCudaThread2D(devID, blockNum * indexSize * copyNum, stride, MAX_INT, cudaGrids, cudaBlocks);

    dim3 blocks(cudaGrids[0], cudaGrids[1]);
    dim3 threads(cudaBlocks[0], cudaBlocks[1]);

    DTYPE * sData = (DTYPE*)s->data;
    DTYPE * cData = (DTYPE*)c->data;

    int * sIndex = (int *)srcIndex->data;
    int * cIndex = (int *)collIndex->data;

    KernelSpreadForCopyIndexed<<<blocks, threads >>>(sData, cData, sIndex, cIndex, 
                                                     blockNum, blockSizeSrc, blockSizeTgt,
                                                     stride, indexSize, copyNum);

    BacktoCudaDev(devID, devIDBackup);
}

/* 
This is core assignment for backward computation of gather function.
Care of the operator "+=" instead of "=".

>> sData - the data pointer of the source tensor
>> cData - the data pointer of collection tensor
>> srcIndex - index of the source sub-tensor
>> indexSize - the number of index
>> stride - stride of a data block
*/
__global__
void KernelSpreadForGather(DTYPE * sData, DTYPE * cData, int * srcIndex, 
                           int indexSize, int stride)
{
    __shared__ DTYPE * sp[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ DTYPE * cp[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    /* block id */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* offset in each block */
    int offset = blockDim.y * blockIdx.y + threadIdx.y;

    if(i >= indexSize || offset >= stride)
        return;

    if (threadIdx.y == 0) {
        sp[threadIdx.x] = sData + srcIndex[i] * stride;
        cp[threadIdx.x] = cData + i * stride;
    }

    __syncthreads();

    DTYPE * s = sp[threadIdx.x];
    DTYPE * c = cp[threadIdx.x];

    //DTYPE * s = sData + srcIndex[i] * stride;
    //DTYPE * c = cData + i * stride;

    atomicAdd(s + offset, c[offset]);
}

/*
spread a collection tensor to source tensor (cuda version).
And this is a special spread function for backward computation of gather function.

>> source - the source tensor whose data would be modified
>> collection - the collection whose data would be spread to source tensor
>> srcIndex - index of the source sub-tensors
*/
void _CudaSpreadForGather(XTensor * source, XTensor * collection, XTensor * srcIndex)
{
    int devID = source->devID;
    XMem * mem = source->mem;

    int stride = source->GetDim(1);
    int indexSize = srcIndex->unitNum;

    int cudaGrids[3];
    int cudaBlocks[3];

    int devIDBackup;
    ProtectCudaDev(source->devID, devIDBackup);

    DTYPE * sData = (DTYPE*)source->data;
    DTYPE * cData = (DTYPE*)collection->data;
    int * sIndex = NULL;

    GDevs.GetCudaThread2D(devID, indexSize, stride, MAX_INT, cudaGrids, cudaBlocks);

    dim3 blocks(cudaGrids[0], cudaGrids[1]);
    dim3 threads(cudaBlocks[0], cudaBlocks[1]);

    if (srcIndex->devID < 0) {
        sIndex = mem != NULL ? 
                (int*)mem->AllocBuf(mem->devID, sizeof(int) * indexSize) : 
                (int*)XMemAlloc(devID, sizeof(int) * indexSize);
        XMemCopy(sIndex, devID, srcIndex->data, -1, sizeof(int) * indexSize);
    }
    else
        sIndex = (int *)srcIndex->data;

    KernelSpreadForGather<<<blocks, threads >>>(sData, cData, sIndex, indexSize, stride);

    if (srcIndex->devID < 0) {
        if(mem != NULL)
            mem->ReleaseBuf(mem->devID, sizeof(int) * indexSize);
        else
            XMemFree(devID, sIndex);
    }

    BacktoCudaDev(source->devID, devIDBackup);
}

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)