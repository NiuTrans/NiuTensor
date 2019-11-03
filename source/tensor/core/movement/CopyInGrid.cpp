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
#include "../shape/IsSameShaped.h"
#include "CopyInGrid.h"
#include "CopyBlocksInGrid.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
copy a number of blocks in grid, 
i.e., reorder the data blocks in the same memory piece
>> s - source tensor
>> t - target tensor
>> index - index[offset_k + j] means the source block id for the j-th target block
in the k-th grid
>> blockDim - leading dimension of blocks
>> blockNumInGrid - number of blocks in each grid
>> isIndexOnDev - indicates whether the index is on the device already
*/
void _CopyInGrid(const XTensor * s, XTensor * t, int * index, int blockDim, int blockNumInGrid, bool isIndexOnDev)
{
    CheckNTErrors((_IsSameShaped(s, t)), "Unmatched tensors!");

    int blockSize = 1;
    int blockNum = blockNumInGrid;
    int gridNum = 1;
    for (int i = blockDim; i < s->order; i++)
        blockSize *= s->dimSize[i];

    CheckNTErrors((s->unitNum % (blockSize * blockNum) == 0), "Illegal block number!");
    gridNum = s->unitNum / (blockSize * blockNum);

    _CopyBlocksInGrid(s->data, blockSize, blockNum, gridNum, t->data, index, s->unitSize, isIndexOnDev, s->mem);
}

} // namespace nts(NiuTrans.Tensor)