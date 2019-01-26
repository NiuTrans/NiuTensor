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

#include "../../XTensor.h"
#include "MakeMergeBlockIndex.h"
#include "MakeMergeBlockIndex.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
set target data block index for the data movement in merge
>> blockIndex - index of the blocks
>> blockNum - number of the blocks
>> blockNumInMerge - size of the dimension along which we perform the merging operation
>> splitSizeInGrid - size of each data array to merge
>> gridSize - number of blocks in a grid (here grid is a higher level orgnization upon blocks)
>> gridNum - number of grids
>> devID - device id
*/
void _MakeMergeBlockIndex(int * blockIndex, int blockNum, int blockNumInMerge,
                          int splitSizeInGrid, int gridSize, int gridNum, int devID)
{
    if (devID >= 0) {
#ifdef USE_CUDA
        _CudaMakeMergeBlockIndex(devID, blockIndex, blockNum, blockNumInMerge, splitSizeInGrid, gridSize, gridNum);
#else
        ShowNTErrors("Please specify USE_CUDA and recompile the code!");
#endif
    }
    else {
        for (int k = 0; k < gridNum; k++) {
            for (int i = 0; i < blockNum; i++) {
                int j = blockNumInMerge * (i % splitSizeInGrid) + int(i / splitSizeInGrid);

                /* i = source block index, j = target block index and k = (target) grid index */
                blockIndex[i + gridSize * k] = j + gridSize * k;
            }
        }
    }
}
} // namespace nts(NiuTrans.Tensor)