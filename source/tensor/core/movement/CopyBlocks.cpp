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
#include "CopyBlocks.h"
#include "CopyBlocksOnSite.h"
#include "CopyBlocksSelected.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
copy a number of blocks to target positions
>> source - data array (head of the blocks) to copy from
>> blockSize - size of block
>> blockNum - number of blocks
>> target - target data array
>> targetBlocks - target positions of the copy
>> myMem - the memory pool
>> devID - device id
*/
void _CopyBlocks(void * source, int blockSize, int blockNum, void * target, int * targetBlocks, XMem * myMem, int devID)
{
    if (myMem != NULL)
        devID = myMem->devID;
    
    if (devID >= 0) {
#ifdef USE_CUDA
        /* copy the index from host to device */
        int * targetBlocksTMP = myMem != NULL ?
                               (int*)myMem->AllocBuf(devID, blockNum * sizeof(int)):
                               (int*)XMemAlloc(devID, blockNum * sizeof(int));
        XMemCopy(targetBlocksTMP, devID, targetBlocks, -1, blockNum * sizeof(int));

        _CopyBlocksOnSite(source, blockSize, blockNum, target, targetBlocksTMP, devID);

        if(myMem != NULL)
            myMem->ReleaseBuf(myMem->devID, blockNum * sizeof(int));
        else
            XMemFree(devID, targetBlocksTMP);
#else
        ShowNTErrors("Plesae specify USE_CUDA and recompile the code!");
#endif
    }
    else {
        _CopyBlocksOnSite(source, blockSize, blockNum, target, targetBlocks, devID);
    }
}

/*
copy a number of blocks source source positions to target positions
>> source - data array (head of the blocks) to copy from
>> blockSize - size of block
>> srcBlocks - source positions of the copy
>> blockNum - number of blocks (lenth of srcBlocks and tgtBlocks)
>> target - target data array
>> targetBlocks - target positions of the copy
>> myMem - the memory pool
>> devID - device id
*/
void _CopyBlocks(void * source, int blockSize, int * sourceBlocks, int blockNum, void * target, int * targetBlocks, XMem * myMem, int devID)
{
    if (myMem != NULL)
        devID = myMem->devID;

    if (devID >= 0) {
#ifdef USE_CUDA
        _CudaCopyBlocksSelected(source, blockSize, sourceBlocks, blockNum, target, targetBlocks, myMem, devID);
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
        for (int i = 0; i < blockNum; i++) {
            XMemCopy((char*)target + targetBlocks[i] * blockSize, devID,
                     (char*)source + sourceBlocks[i] * blockSize, devID, blockSize);
        }
    }
}

} // namespace nts(NiuTrans.Tensor)
