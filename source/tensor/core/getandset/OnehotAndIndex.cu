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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-07-31
 */

#include "OnehotAndIndex.cuh"
#include "../../XDevice.h"


namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* 
convert onehot tensor to index tensor (kernel version) 

>> onehotData - the data pointer of the onehot tensor
>> indexData - the data pointer of the index tensor
>> blockNum - the number of block
>> stride - stride of a data block
*/
__global__
void KernelOnehotToIndex(int * onehotData, int * indexData, int blockNum, int stride)
{
    /* block id */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* offset in each block */
    int offset = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= blockNum || offset >= stride)
        return;

    int * od = onehotData + i * stride;
    int * id = indexData + i;

    if (od[offset] != 0)
        *id = offset;
}

/* 
convert onehot tensor to index tensor (cuda version) 

>> onehot - onehot tensor, which value is 0 or 1
>> index - index tensor, which value is an integer num
>> size - the last dimension size of the onehot tensor
*/
void _CudaOnehotToIndex(const XTensor * onehot, XTensor * index, int size)
{
    int devID = onehot->devID;

    int blockNum = index->unitNum;
    int stride = size;

    int cudaGrids[3];
    int cudaBlocks[3];

    int devIDBackup;
    ProtectCudaDev(devID, devIDBackup);

    GDevs.GetCudaThread2D(devID, blockNum, stride, MAX_INT, cudaGrids, cudaBlocks);

    dim3 blocks(cudaGrids[0], cudaGrids[1]);
    dim3 threads(cudaBlocks[0], cudaBlocks[1]);

    int * onehotData = (int *)onehot->data;
    int * indexData = (int *)index->data;

    KernelOnehotToIndex<<<blocks, threads >>>(onehotData, indexData, blockNum, stride);

    BacktoCudaDev(devID, devIDBackup);
}

/* 
convert index tensor to onehot tensor (kernel version) 

>> onehotData - the data pointer of the onehot tensor
>> indexData - the data pointer of the index tensor
>> blockNum - the number of block
>> stride - stride of a data block
*/
__global__
void KernelIndexToOnehot(DTYPE * onehotData, int * indexData, int blockNum, int stride, float confidence, float lowconfidence)
{
    /* block id */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* offset in each block */
    int offset = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= blockNum || offset >= stride)
        return;

    DTYPE * od = onehotData + i * stride;

    int id = indexData[i];

    if (offset == id)
        od[offset] = confidence;
    //else
    //    od[offset] = lowconfidence;
}

/* 
convert index tensor to onehot tensor (cuda version) 

>> index - index tensor, which value is an integer num
>> onehot - onehot tensor, which value is 0 or 1
>> size - the last dimension size of the onehot tensor
*/
void _CudaIndexToOnehot(const XTensor * index, XTensor * onehot, 
                        int size, float confidence, float lowconfidence)
{
    int devID = onehot->devID;

    int blockNum = index->unitNum;
    int stride = size;

    int cudaGrids[3];
    int cudaBlocks[3];

    int devIDBackup;
    ProtectCudaDev(devID, devIDBackup);

    GDevs.GetCudaThread2D(devID, blockNum, stride, MAX_INT, cudaGrids, cudaBlocks);

    dim3 blocks(cudaGrids[0], cudaGrids[1]);
    dim3 threads(cudaBlocks[0], cudaBlocks[1]);

    DTYPE * onehotData = (DTYPE *)onehot->data;
    int * indexData = (int *)index->data;

    KernelIndexToOnehot<<<blocks, threads >>>(onehotData, indexData, blockNum, stride, confidence, lowconfidence);

    BacktoCudaDev(devID, devIDBackup);
}

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)