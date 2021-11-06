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
* $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-04-24
*/

#include "CopyBlocks.h"
#include "CopyBlocksSelected.cuh"
#include "../../XUtility.h"
#include "../../XDevice.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/*
copy a number of blocks from source positions to target positions
>> source - data array (head of the blocks) to copy from
>> blockSize - size of block
>> sourceBlocks - source positions of the copy
>> blockNum - number of blocks
>> target - target data array
>> targetBlocks - target positions of the copy
*/
template <class T>
__global__
void KernelCopyBlocksSelected(T * source, int blockSize, int * sourceBlocks, int blockNum, T * target, int * targetBlocks)
{
    /* block index */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* entry index in the block */
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (j >= blockNum)
        return;

    /* target position */
    int srcIndex = sourceBlocks[j];
    int tgtIndex = targetBlocks[j];

    T * s = source + blockSize * srcIndex;
    T * t = target + blockSize * tgtIndex;

    if (i < blockSize)
        t[i] = s[i];
}

/*
copy a number of blocks from source positions to target positions (cuda version)
>> source - data array (head of the blocks) to copy from
>> blockSize - size of block
>> sourceBlocks - source positions of the copy
>> blockNum - number of blocks
>> target - target data array
>> targetBlocks - target positions of the copy
>> myMem - memory pool
*/
void _CudaCopyBlocksSelected(void * source, int unitSize, int blockSize, int * sourceBlocks, int blockNum, void * target, int * targetBlocks, XMem * myMem, int devID)
{
    CheckNTErrors(devID >= 0, "Wrong device to run!");
    CheckNTErrors((blockSize % unitSize == 0), "Unsupported block size!");

    int devIDBackup;
    ProtectCudaDev(devID, devIDBackup);

    /* copy the index to the GPU memory */
    /*int * sourceBlocksTMP = myMem != NULL ? 
                           (int*)myMem->AllocBuf(myMem->devID, blockNum * sizeof(int)) : 
                           (int *)XMemAlloc(devID, blockNum * sizeof(int));
    int * targetBlocksTMP = myMem != NULL ? 
                           (int*)myMem->AllocBuf(myMem->devID, blockNum * sizeof(int)) : 
                           (int *)XMemAlloc(devID, blockNum * sizeof(int));*/
    int * sourceBlocksTMP;
    int * targetBlocksTMP;
    if (myMem != NULL) {
        myMem->LockBuf();
        sourceBlocksTMP = (int*)myMem->AllocBuf(myMem->devID, blockNum * sizeof(int));
        targetBlocksTMP = (int*)myMem->AllocBuf(myMem->devID, blockNum * sizeof(int));
    }
    else {
        sourceBlocksTMP = (int *)XMemAlloc(devID, blockNum * sizeof(int));
        targetBlocksTMP = (int *)XMemAlloc(devID, blockNum * sizeof(int));
    }
    
    XMemCopy(sourceBlocksTMP, devID, sourceBlocks, -1, blockNum * sizeof(int));
    XMemCopy(targetBlocksTMP, devID, targetBlocks, -1, blockNum * sizeof(int));

    int cudaGrids[3];
    int cudaBlocks[3];

    int bSize = blockSize / unitSize;
    GDevs.GetCudaThread2D(devID, bSize, blockNum, MAX_INT, cudaGrids, cudaBlocks);
    if (unitSize == 4)
        KernelCopyBlocksSelected <<<dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1])>>>
                                  ((float*)source, bSize, sourceBlocksTMP, blockNum, (float*)target, targetBlocksTMP);
    else if (unitSize == 2)
        KernelCopyBlocksSelected <<<dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1])>>>
                                  ((half*)source, bSize, sourceBlocksTMP, blockNum, (half*)target, targetBlocksTMP);
    else
        ShowNTErrors("Unsupported unit size!");

    if (myMem != NULL) {
        myMem->ReleaseBuf(myMem->devID, blockNum * sizeof(int));
        myMem->ReleaseBuf(myMem->devID, blockNum * sizeof(int));
        myMem->UnlockBuf();
    }
    else {
        XMemFree(devID, sourceBlocksTMP);
        XMemFree(devID, targetBlocksTMP);
    }

    BacktoCudaDev(devID, devIDBackup);
}

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)