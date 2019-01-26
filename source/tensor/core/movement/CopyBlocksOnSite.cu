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

#include "CopyBlocksOnSite.h"
#include "CopyBlocksOnSite.cuh"
#include "../../XDevice.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/*
copy a number of blocks to target positions
NOTE that this version makes more use of the 2d threads in cuda
>> source - data array (head of the blocks) to copy from
>> blockSize - size of block
>> blockNum - number of blocks
>> target - target data array
>> targetBlocks - target positions of the copy
*/
template<class T>
__global__
void KernelCopyBlocks(T * source, int blockSize, int blockNum, T * target, int * targetBlocks)
{
    /* entry index in the block */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* block index */
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= blockSize || j >= blockNum)
        return;

    T * s = source + blockSize * j;
    T * t = target + blockSize * targetBlocks[j];

    t[i] = s[i];
}

/*
copy a number of blocks to target positions
NOTE that this version makes more use of the 2d threads in cuda
>> source - data array (head of the blocks) to copy from
>> blockSize - size of block
>> blockNum - number of blocks
>> target - target data array
>> targetBlocks - target positions of the copy
*/
template<class T>
__global__
void KernelCopyBlocksV2(T * source, int blockSize, int blockNum, int totalSize, T * target, int * targetBlocks)
{
    /* entry index in the block */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= totalSize)
        return;

    int targetBlockID = targetBlocks[i / blockSize];
    int targetOffset  = i % blockSize;

    target[blockSize * targetBlockID + targetOffset] = source[i];
}

/*
copy a number of blocks to target positions (cuda version)
>> source - data array (head of the blocks) to copy from
>> blockSize - size of block
>> blockNum - number of blocks
>> target - target data array
>> targetBlocks - target positions of the copy (on the device)
>> devID - device id
*/
void _CudaCopyBlocks(void * source, int blockSize, int blockNum, void * target, int * targetBlocks, int devID)
{
    CheckNTErrors(devID >= 0, "Wrong device to run!");
    int cudaGrids[3];
    int cudaBlocks[3];

    int devIDBackup;
    ProtectCudaDev(devID, devIDBackup);

    if(blockSize % sizeof(float) == 0){
        int bSize = blockSize / sizeof(float);
        GDevs.GetCudaThread(devID, bSize * blockNum, cudaGrids, cudaBlocks);
        KernelCopyBlocksV2<float> <<<dim3(cudaGrids[0]), dim3(cudaBlocks[0]) >>>
                                   ((float*)source, bSize, blockNum, bSize * blockNum, (float*)target, targetBlocks);
        //GDevs.GetCudaThread2D(devID, bSize, blockNum, MAX_INT, cudaGrids, cudaBlocks);
        //KernelCopyBlocks<float> <<<dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >>>
        //                         ((float*)source, bSize, blockNum, (float*)target, targetBlocks);
    }
    else{
        ShowNTErrors("Unsupported block size!");
    }

    BacktoCudaDev(devID, devIDBackup);
}
#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)
