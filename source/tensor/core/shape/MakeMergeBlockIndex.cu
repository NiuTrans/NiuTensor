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
#include "../../XTensor.h"
#include "MakeMergeBlockIndex.h"
#include "MakeMergeBlockIndex.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/*
set target data block index for the data movement in split (device code)
>> blockIndex - index of the blocks
>> blockNum - number of the blocks
>> blockNumInMerge - size of the dimension along which we perform the merging operation
>> splitSizeInGrid - size of each data array to merge
>> gridSize - number of blocks in a grid (here grid is a higher level orgnization upon blocks)
>> gridNum - number of grids
>> mem - the memory pool
*/
__global__
void KernelMakeMergeBlockIndex(int * blockIndex, int blockNum, int blockNumInMerge,
                               int splitSizeInGrid, int gridSize, int gridNum)
{
    /* block index */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* grid index */
    int k = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= blockNum || k >= gridNum)
        return;

    int j = blockNumInMerge * (i % splitSizeInGrid) + int(i / splitSizeInGrid);

    /* i = source block index, j = target block index and k = (target) grid index */
    blockIndex[i + gridSize * k] = j + gridSize * k;
}


/*
set target data block index for the data movement in split
>> devID - id of the GPU device
>> blockIndex - index of the blocks
>> blockNum - number of the blocks
>> blockNumInMerge - size of the dimension along which we perform the merging operation
>> splitSizeInGrid - size of each data array to merge
>> gridSize - number of blocks in a grid (here grid is a higher level orgnization upon blocks)
>> gridNum - number of grids
>> mem - the memory pool
*/
void _CudaMakeMergeBlockIndex(int devID,
                              int * blockIndex, int blockNum, int blockNumInMerge,
                              int splitSizeInGrid, int gridSize, int gridNum)
{
    int cudaGrids[3];
    int cudaBlocks[3];

    GDevs.GetCudaThread2D(devID, blockNum, gridNum, MAX_INT, cudaGrids, cudaBlocks);

    int devIDBackup;
    ProtectCudaDev(devID, devIDBackup);

    KernelMakeMergeBlockIndex << <dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >> >
                                 (blockIndex, blockNum, blockNumInMerge, splitSizeInGrid, gridSize, gridNum);

    BacktoCudaDev(devID, devIDBackup);
}
#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)