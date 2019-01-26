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
#include "../movement/CopyValues.cuh"
#include "Power.h"
#include "Power.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/*
set all entries to its root (CUDA Kernel)
>> a - input data array
>> b - output data array
>> size - size of the data array
*/
__global__
void KernelSqrtV2(DTYPE * a, DTYPE * b, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size)
        b[i] = sqrt(a[i]);
}

/*
set all entries to its root (CUDA Kernel)
>> a - input data array
>> b - output data array
>> size - size of the data array
*/
__global__
void KernelSqrtV2(__half * a, __half * b, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    if (i < size)
        b[i] = hsqrt(a[i]);
#else
    if (i < size)
        b[i] = __float2half(sqrt(__half2float(a[i])));
#endif
}


/*
get power(d[i], p)
>> a - input data array
>> b - output data array
>> p - power
>> size - size of the data array
*/
__global__
void KernelPower(DTYPE * a, DTYPE * b, DTYPE p, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size) {
        DTYPE v = a[i];
        if (p < 0 && v == 0)
            b[i] = 1e20;
        else
            b[i] = pow(a[i], p);
    }
}

/*
get power(d[i], p)
>> a - input data array
>> b - output data array
>> p - power
>> size - size of the data array
*/
__global__
void KernelPower(__half * a, __half * b, __half p, int size)
{
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
#else
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        float v = __half2float(a[i]);
        if (__half2float(p) < 0 && v == 0)
            b[i] = __float2half(1e20);
        else
            b[i] = __float2half(pow(__half2float(a[i]), __half2float(p)));
    }
#endif
}

/* get the power of the entries */
void _CudaPower(const XTensor * a, XTensor * b, DTYPE p)
{
    CheckNTErrors((XTensor::IsSameShaped(a, b)), "Input tensors should have the same type!");
    
    int gridSize[3];
    int blockSize[3];

    GDevs.GetCudaThread(a->devID, a->unitNum, gridSize, blockSize);

    dim3 blocks(gridSize[0]);
    dim3 threads(blockSize[0]);

    int devIDBackup;
    ProtectCudaDev(a->devID, devIDBackup);

    if (a->dataType == DEFAULT_DTYPE) {
        if (p == (DTYPE)0.5) {
            KernelSqrtV2 << <blocks, threads >> >((DTYPE*)a->data, (DTYPE*)b->data, a->unitNum);
        }
        else if (p == (DTYPE)1.0) {
            _CudaCopyValues(a, b);
        }
        else if (p != (DTYPE)1.0) {
            KernelPower << <blocks, threads >> >((DTYPE*)a->data, (DTYPE*)b->data, p, a->unitNum);
        }
    }
    else if (a->dataType == X_FLOAT16) {
        if (p == (DTYPE)0.5) {
            KernelSqrtV2 << <blocks, threads >> >((__half*)a->data, (__half*)b->data, a->unitNum);
        }
        else if (p != (DTYPE)1.0) {
            ShowNTErrors("TODO!");
        }
    }
    else {
        ShowNTErrors("TODO!");
    }

    BacktoCudaDev(a->devID, devIDBackup);
}

#endif // USE_CUDA
} // namespace nts(NiuTrans.Tensor)