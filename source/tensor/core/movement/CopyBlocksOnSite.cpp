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
#include "CopyBlocksOnSite.h"
#include "CopyBlocksOnSite.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
copy a number of blocks to target positions. Here we assume that
all the data has been on the device (CPU/GPU) already.
>> source - data array (head of the blocks) to copy from
>> blockSize - size of block
>> blockNum - number of blocks
>> target - target data array
>> targetBlocks - target positions of the copy
>> devID - device id
*/
void _CopyBlocksOnSite(void * source, int blockSize, int blockNum, void * target, int * targetBlocks, int devID)
{
    if (devID >= 0) {
#ifdef USE_CUDA
        _CudaCopyBlocks(source, blockSize, blockNum, target, targetBlocks, devID);
#else
        ShowNTErrors("Plesae specify USE_CUDA and recompile the code!");
#endif
    }
    else {
        /* 
        The following code should be fine with GPUs, but too many
        kernel calls would slow down the system. We prefer to use
        one kernel to do block copy in batch (kernel fusion). 
        */
        if(blockSize == sizeof(int)){
            for (int i = 0, b = 0; i < blockNum; i++, b += blockSize) {
                *(int*)((char*)target + targetBlocks[i] * blockSize) = 
                *(int*)((char*)source + b);
            }
        }
        else{
            for (int i = 0, b = 0; i < blockNum; i++, b += blockSize) {
                XMemCopy((char*)target + targetBlocks[i] * blockSize, devID,
                         (char*)source + b, devID, blockSize);
            }
        }
    }
}
} // namespace nts(NiuTrans.Tensor)
