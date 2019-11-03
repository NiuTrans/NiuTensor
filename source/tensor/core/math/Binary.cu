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
* $Created by: JIANG Yufan (email: jiangyufan2018@outlook.com) 2019-04-05
*/

#include <math.h>
#include "../../XDevice.h"
#include "../../XUtility.h"
#include "../../XName.h"
#include "../shape/IsSameShaped.h"
#include "Binary.h"
#include "Binary.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA
    
__device__
int BinaryCudaMod(int x, int base)
{
    return x % base;
}

template<class T1, class T2>
__device__
T1 BinaryCudaDescale(T1 x, T2 num)
{
    return x / num;
}

template<class T1, class T2>
__device__
T1 BinaryCudaPower(T1 x, T2 num)
{
    if (num == 0)
        return (T1)1.0;
    else if (num == 0.5)
        return (T1)sqrt((float)x);
    else if (num == 2)
        return (T1)(x * x);
    else {
        if (x == 0 && num < 0)
            return (T1)1e20F;
        else
            return (T1)pow((float)x, (float)num);
    }
}

template<class T1, class T2>
__device__
T1 BinaryCudaScale(T1 x, T2 num)
{
    return x * num;
}

template<class T1, class T2>
__device__
T1 BinaryCudaShift(T1 x, T2 num)
{
    return x + num;
}

#define SIMPLE_BINARY_FUNCTION_GPU(funcName, origFunc)                              \
template<class T1, class T2>                                                        \
__global__                                                                          \
void Kernel##funcName(T1 * a, T1 * b, int size, T2 num)                             \
{                                                                                   \
    int i = blockDim.x * blockIdx.x + threadIdx.x;                                  \
                                                                                    \
    if (i < size)                                                                   \
        b[i] = (T1)origFunc((T1)a[i], (T2)num);                                     \
}                                                                                   \
                                                                                    \
template<class T>                                                                   \
void _Cuda##funcName(const XTensor * a, XTensor * b, T num)                         \
{                                                                                   \
    CheckNTErrors((_IsSameShaped(a, b)),                                    \
                  "Input tensors should have the same type!");                      \
    CheckNTErrors((a->isSparse == false), "TODO!");                                 \
                                                                                    \
    int gridSize[3];                                                                \
    int blockSize[3];                                                               \
                                                                                    \
    GDevs.GetCudaThread(a->devID, a->unitNum, gridSize, blockSize);                 \
                                                                                    \
    dim3 blocks(gridSize[0]);                                                       \
    dim3 threads(blockSize[0]);                                                     \
                                                                                    \
    int devIDBackup;                                                                \
    ProtectCudaDev(a->devID, devIDBackup);                                          \
                                                                                    \
    if (a->dataType == X_FLOAT) {                                                   \
        Kernel##funcName<<<blocks, threads>>>                                       \
                         ((float*)a->data, (float*)b->data, a->unitNum, (T)num);    \
    }                                                                               \
    else if (a->dataType == X_DOUBLE) {                                             \
        Kernel##funcName<<<blocks, threads>>>                                       \
                         ((double*)a->data, (double*)b->data, a->unitNum, (T)num);  \
    }                                                                               \
    else if (a->dataType == X_INT) {                                                \
        Kernel##funcName<<<blocks, threads>>>                                       \
                         ((int*)a->data, (int*)b->data, a->unitNum, (T)num);        \
    }                                                                               \
    else {                                                                          \
        ShowNTErrors("TODO!");                                                      \
    }                                                                               \
                                                                                    \
    BacktoCudaDev(a->devID, devIDBackup);                                           \
}                                                                                   \
template void _Cuda##funcName<int>(const XTensor*, XTensor*, int);                  \
template void _Cuda##funcName<float>(const XTensor*, XTensor*, float);              \
template void _Cuda##funcName<double>(const XTensor*, XTensor*, double);            

SIMPLE_BINARY_FUNCTION_GPU(Descale, BinaryCudaDescale)
SIMPLE_BINARY_FUNCTION_GPU(Mod, BinaryCudaMod)
SIMPLE_BINARY_FUNCTION_GPU(Power, BinaryCudaPower)
SIMPLE_BINARY_FUNCTION_GPU(Scale, BinaryCudaScale)
SIMPLE_BINARY_FUNCTION_GPU(Shift, BinaryCudaShift)

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)
