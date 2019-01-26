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
#include "CopyBlocksInGrid.h"
#include "../../XUtility.h"
#include "CopyBlocksInGrid.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
copy a number of blocks in grid
>> source - pointer to the source data array
>> blockSize - size of a data block
>> blockNum - number of the blocks (in a grid)
>> gridNum - number of the grids.
Note that a grid may have a number of blocks
>> target - pointer to the target data array
>> index - source block id for each target block
>> myMem - the memory pool
>> isIndexOnDev - indicates whether the index is on the device already
*/
void _CopyBlocksInGrid(void * source, int blockSize, int blockNum, int gridNum, void * target,
                       int * index, int unitSize, bool isIndexOnDev, XMem * myMem)
{
    CheckNTErrors((unitSize == sizeof(int)), "TODO!");

    if (myMem != NULL && myMem->devID >= 0) {
#ifdef USE_CUDA
        int * indexGPU = index;
        if (!isIndexOnDev) {
            indexGPU = (int*)myMem->AllocBuf(myMem->devID, blockNum * gridNum * sizeof(int));
            XMemCopy(indexGPU, myMem->devID, index, -1, blockNum * gridNum * sizeof(int));
        }

        _CudaCopyBlocksInGrid(source, blockSize, blockNum, gridNum, target, indexGPU, unitSize, myMem);

        if (!isIndexOnDev)
            myMem->ReleaseBuf(myMem->devID, blockNum * gridNum * sizeof(int));
#else
        ShowNTErrors("Plesae specify USE_CUDA and recompile the code!");
#endif
    }
    else if(myMem != NULL){
        void * buf = XMemAlloc(myMem->devID, blockSize * blockNum * unitSize);
        for (int k = 0; k < gridNum; k++) {
            int offset = k * blockSize * blockNum;
            for (int i = 0; i < blockNum; i++) {
                int b = index[k * blockNum + i];
                if (b >= 0 && b < blockNum) {
                    int * t = (int*)buf + blockSize * i;
                    int * s = (int*)source + offset + blockSize * b;
                    for (int j = 0; j < blockSize; j++)
                        t[j] = s[j];
                }
            }
            XMemCopy((int*)target + offset, myMem->devID,
                buf, myMem->devID,
                blockSize * blockNum * unitSize);
        }
        XMemFree(myMem->devID, buf);
    }
    else {
        ShowNTErrors("TODO!");
    }
}

} // namespace nts(NiuTrans.Tensor)