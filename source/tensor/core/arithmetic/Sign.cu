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
* $Created by: LI Yinqiao (li.yin.qiao.2012@hotmail.com) 2018-7-11
*/

#include "../../XDevice.h"
#include "../../XTensor.h"
#include "Sign.h"
#include "Sign.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA
/*
set each entry to its sign value (CUDA Kernel)
>> a - pointer to input data array
>> b - pointer to output data array
>> size - size of the data array
*/
__global__
void KernelSign(DTYPE * a, DTYPE * b, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size) {
        if (a[i] > 0)
            b[i] = 1.0F;
        else if (a[i] == 0)
            b[i] = 0.0F;
        else
            b[i] = -1.0F;
    }
}

/*
set each entry to its sign value with float16 data type value (CUDA Kernel)
This is for float16 computation
>> a - pointer to input data array
>> b - pointer to output data array
>> size - size of the data array
*/
__global__
void KernelSign(__half * a, __half * b, int size)
{
    return;
}

/*
set each entry to its sign value
>> a - input tensor we are processing
>> b - output tensor we are processing
*/
void _CudaSign(const XTensor * a, XTensor * b)
{
    CheckNTErrors((XTensor::IsSameShaped(a, b)), "Input tensors should have the same type!");
    CheckNTErrors((a->isSparse == false), "TODO!");

    int gridSize[3];
    int blockSize[3];

    GDevs.GetCudaThread(a->devID, a->unitNum, gridSize, blockSize);

    dim3 blocks(gridSize[0]);
    dim3 threads(blockSize[0]);

    int devIDBackup;
    ProtectCudaDev(a->devID, devIDBackup);

    if (a->dataType == DEFAULT_DTYPE) {
        KernelSign << <blocks, threads >> >((DTYPE*)a->data, (DTYPE*)b->data, a->unitNum);
    }
    else if (a->dataType == X_FLOAT16) {
        KernelSign << <blocks, threads >> >((__half*)a->data, (__half*)b->data, a->unitNum);
    }
    else {
        ShowNTErrors("TODO!");
    }

    BacktoCudaDev(a->devID, devIDBackup);
}

#endif // USE_CUDA
} // namespace nts(NiuTrans.Tensor)
