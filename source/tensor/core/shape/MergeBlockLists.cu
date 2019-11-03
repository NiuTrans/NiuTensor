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
#include "MergeBlockLists.h"
#include "MergeBlockLists.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA
/*
copy a number of blocks (of different sizes) to target positions
>> sourceList - list of data arrays to copy from
>> sourceBlockSizes - the size of the block_i
>> sourceBlockNum - number of blocks to merge
>> targetList - list of data arrays to copy to
*/
__global__
void KernelCopyBlockLists(DTYPE * sourceList[], int * sourceBlockSizes, int sourceBlockNum, DTYPE * targetList[])
{
    __shared__ int iBlockSizes[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ DTYPE * iSourceList[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ DTYPE * iTargetList[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    /* entry index in the block */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* block index */
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (j >= sourceBlockNum)
        return;

    if (threadIdx.x == 0) {
        iBlockSizes[threadIdx.y] = sourceBlockSizes[j];
        iSourceList[threadIdx.y] = sourceList[j];
        iTargetList[threadIdx.y] = targetList[j];
    }

    __syncthreads();

    if (i < iBlockSizes[threadIdx.y])
        iTargetList[threadIdx.y][i] = iSourceList[threadIdx.y][i];
}

/*
merge data by blocks (cuda version)
>> sourceList - list of data arrays (heads of the blocks) to copy from
>> blockSizes - size of the blocks
>> blockNum - number of blocks
>> target - target data array
>> myMem - the memory pool
*/
void _CudaMergeBlockLists(const StrList* sourceList, int * blockSizes, int blockNum, void * target, XMem * myMem)
{
    CheckNTErrors((myMem != NULL), "No memory pool!");
    CheckNTErrors((myMem->devID >= 0), "Wrong device to run!");

    int newBlockListSize = sourceList->count * blockNum;

    int minBlockSize = MAX_INT;
    int maxBlockSize = -MAX_INT;
    int realMaxBlockSize = 1;
    DTYPE ** sourceArrays = new DTYPE*[newBlockListSize];
    DTYPE ** targetArrays = new DTYPE*[newBlockListSize];
    int * sizes = new int[newBlockListSize];
    int * offsets = new int[sourceList->count];
    memset(offsets, 0, sizeof(int) * sourceList->count);

    int totalOffset = 0;
    for (int k = 0; k < blockNum; k++) {
        for (int i = 0; i < sourceList->count; i++) {
            CheckNTErrors((blockSizes[i] % sizeof(DTYPE) == 0), "Unsupported block size!");
            int j = k * sourceList->count + i;
            sizes[j] = blockSizes[i] / sizeof(DTYPE);
            sourceArrays[j] = (DTYPE*)sourceList->GetItem(i) + offsets[i];
            targetArrays[j] = (DTYPE*)target + totalOffset;
            offsets[i] += sizes[i];
            totalOffset += sizes[i];

            if (minBlockSize > blockSizes[i])
                minBlockSize = blockSizes[i];
            if (maxBlockSize < blockSizes[i])
                maxBlockSize = blockSizes[i];
        }
    }

    CheckNTErrors((minBlockSize % sizeof(DTYPE) == 0), "Unsupported block size!");
    CheckNTErrors((maxBlockSize % sizeof(DTYPE) == 0), "Unsupported block size!");
    realMaxBlockSize = maxBlockSize / sizeof(DTYPE);

    int devIDBackup;
    ProtectCudaDev(myMem->devID, devIDBackup);

    int cudaGridSizes[3];
    int cudaBlockSizes[3];

    GDevs.GetCudaThread2D(myMem->devID, realMaxBlockSize, newBlockListSize, MAX_INT,
                          cudaGridSizes, cudaBlockSizes);

    myMem->SetPinBuf();
    int * sizesGPU = (int*)myMem->AllocBuf(myMem->devID, sizeof(int) * newBlockListSize, 256);

    DTYPE ** sourceArraysGPU = (DTYPE**)myMem->AllocBuf(myMem->devID, sizeof(DTYPE*) * newBlockListSize, 256);

    DTYPE ** targetArraysGPU = (DTYPE**)myMem->AllocBuf(myMem->devID, sizeof(DTYPE*) * newBlockListSize, 256);

    XMemCopy(sizesGPU, myMem->devID, sizes, -1, sizeof(int) * newBlockListSize);
    XMemCopy(sourceArraysGPU, myMem->devID, sourceArrays, -1, sizeof(DTYPE*) * newBlockListSize);
    XMemCopy(targetArraysGPU, myMem->devID, targetArrays, -1, sizeof(DTYPE*) * newBlockListSize);

    KernelCopyBlockLists << <dim3(cudaGridSizes[0], cudaGridSizes[1]), dim3(cudaBlockSizes[0], cudaBlockSizes[1]) >> >
                            (sourceArraysGPU, sizesGPU, newBlockListSize, targetArraysGPU);

    myMem->BackToPinBuf();

    delete[] sourceArrays;
    delete[] targetArrays;
    delete[] sizes;
    delete[] offsets;

    BacktoCudaDev(myMem->devID, devIDBackup);
}
#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)