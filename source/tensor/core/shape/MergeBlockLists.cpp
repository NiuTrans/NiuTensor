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
#include "../../XUtility.h"
#include "MergeBlockLists.h"
#include "MergeBlockLists.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
merge data by blocks
>> sourceList - list of source data array
>> blockSizes - list of the block size for each source data array
>> blockNum - number of blocks kept in each data array
>> target - target data array
>> myMem - memory pool
*/
void _MergeBlockLists(const StrList* sourceList, int * blockSizes, int blockNum, void * target, XMem * myMem)
{
    if (myMem != NULL && myMem->devID >= 0) {
#ifdef USE_CUDA
        _CudaMergeBlockLists(sourceList, blockSizes, blockNum, target, myMem);
#else
        ShowNTErrors("Plesae specify USE_CUDA and recompile the code!");
#endif
    }
    else {
        int devID = myMem != NULL ? myMem->devID : -1;

        char ** dataArrays = new char*[sourceList->count];
        int * offsets = new int[sourceList->count];
        for (int i = 0; i < sourceList->count; i++) {
            dataArrays[i] = (char*)sourceList->GetItem(i);
            offsets[i] = 0;
        }

        int size = 0;
        for (int i = 0; i < blockNum; i++) {
            for (int j = 0; j < sourceList->count; j++) {
                XMemCopy((char*)target + size, devID,
                    (char*)dataArrays[j] + offsets[j], devID, blockSizes[j]);
                offsets[j] += blockSizes[j];
                size += blockSizes[j];
            }
        }

        delete[] dataArrays;
        delete[] offsets;
    }
}

} // namespace nts(NiuTrans.Tensor)