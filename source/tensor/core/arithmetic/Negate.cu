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
#include "Negate.h"
#include "Negate.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA
/*
set each entry to its negtive value (CUDA Kernel)
>> a - pointer to the input data array
>> b - pointer to the output data array
>> size - size of the data array
*/
__global__
void KernelNegate(DTYPE * a, DTYPE * b, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size)
        b[i] = -a[i];
}

/*
set each entry to its negtive value (CUDA Kernel)
This is for float16 computation
>> a - pointer to the input data array
>> b - pointer to the output data array
>> size - size of the data array
*/
__global__
void KernelNegate(__half * a, __half * b, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
        if (i < size)
            b[i] = __hsub(__float2half(0), a[i]);
#else
        if (i < size)
            b[i] = __float2half(-__half2float(a[i]));
#endif
}

/*
set each entry to its negtive value
>> a - input tensor
>> b - output tensor
*/
void _CudaNegate(const XTensor * a, XTensor * b)
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
        KernelNegate << <blocks, threads >> >((DTYPE*)a->data, (DTYPE*)b->data, a->unitNum);
    }
    else if (a->dataType == X_FLOAT16) {
        KernelNegate << <blocks, threads >> >((__half*)a->data, (__half*)b->data, a->unitNum);
    }
    else {
        ShowNTErrors("TODO!");
    }

    BacktoCudaDev(a->devID, devIDBackup);
}

#endif // USE_CUDA
} // namespace nts(NiuTrans.Tensor)
