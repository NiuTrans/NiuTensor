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
#include "MakeSplitBlockIndex.h"
#include "MakeSplitBlockIndex.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/*
set target data block index for the data movement in split (device code)
>> blockIndex - block index
>> splitNum - number of splits
>> blockSplitSize - size of the splitted block
>> blockNum - number of data blocks
*/
__global__
void KernelMakeSplitBlockIndex(int * blockIndex, int splitNum, int blockSplitSize, int blockNum)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= blockNum)
        return;

    int j = (i % splitNum) * blockSplitSize + i / splitNum;

    /* i = source block index, j = target block index */
    blockIndex[i] = j;
}

/*
set target data block index for the data movement in split
>> devID - device id
>> blockIndex - block index
>> splitNum - number of splits
>> blockSplitSize - size of the splitted block
>> blockNum - number of data blocks
*/
void _CudaMakeSplitBlockIndex(int devID, int * blockIndex, int splitNum, int blockSplitSize, int blockNum)
{
    int cudaGrids[3];
    int cudaBlocks[3];

    GDevs.GetCudaThread(devID, blockNum, cudaGrids, cudaBlocks);

    int devIDBackup;
    ProtectCudaDev(devID, devIDBackup);

    KernelMakeSplitBlockIndex << <dim3(cudaGrids[0]), dim3(cudaBlocks[0]) >> >
                                 (blockIndex, splitNum, blockSplitSize, blockNum);

    BacktoCudaDev(devID, devIDBackup);
}

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)