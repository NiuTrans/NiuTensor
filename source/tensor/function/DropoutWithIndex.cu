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
* $Created by: Jiang Yufan (email: jiangyufan2018@outlook.com) 2019-03-20
*/

#include "DropoutWithIndex.cuh"
#include "../XDevice.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA
__global__

/*
This is a special implementation of "dropout" to reduce memory with maskIndex.

>> tData - the data pointer of the target tensor
>> sIndex - mask index
>> size - the size of the sIndex
*/
void KernelDropoutWithIndex1D(DTYPE * tData, int * sIndex, int size)
{
    /* block id */
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    DTYPE * t = tData;
    
    if (i < size) {

        int id = sIndex[i];

        t[id] = DTYPE(0.0F);
    }
    
}

/*
This is a special implementation of "dropout" to reduce memory with maskIndex.

>> x - input tensor
>> maskIndex - mask index tensor
>> c - output tensor
*/
void _CudaDropoutWithIndex(const XTensor * x, XTensor * maskIndex, XTensor * c)
{
    int devID = c->devID;

    int blockNum = maskIndex->unitNum;

    int cudaGrids[3];
    int cudaBlocks[3];

    int devIDBackup;
    ProtectCudaDev(devID, devIDBackup);

    GDevs.GetCudaThread(devID, blockNum, cudaGrids, cudaBlocks);

    dim3 blocks(cudaGrids[0]);
    dim3 threads(cudaBlocks[0]);

    DTYPE * tData = (DTYPE*)c->data;
    int * sIndex = NULL;

    sIndex = (int *)maskIndex->data;

    KernelDropoutWithIndex1D <<<blocks, threads >>>(tData, sIndex, blockNum);

    BacktoCudaDev(devID, devIDBackup);
}

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)