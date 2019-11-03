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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-07-31
 */

#include <math.h>
#include "../../XDevice.h"
#include "../../XName.h"
#include "../shape/IsSameShaped.h"
#include "Unary.h"
#include "Unary.cuh"
#include<cuda_runtime.h>

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

template<class T>
__device__
T UnaryCudaCeil(T x)
{
    return (T)ceil((float)x);
}

template<class T>
__device__
T UnaryCudaExp(T x)
{
    return (T)exp((float)x);
}

template<class T>
__device__
T UnaryCudaFabs(T x)
{
    return (T)fabs((float)x);
}

template<class T>
__device__
T UnaryCudaFloor(T x)
{
    return (T)floor((float)x);
}

template<class T>
__device__
T UnaryCudaIsNonZero(T r)
{
    return (r != (T)0.0) ? (T)1.0 : (T)0.0;
}

template<class T>
__device__
T UnaryCudaIsZero(T r)
{
    return (r == (T)0.0) ? (T)1.0 : (T)0.0;
}

template<class T>
__device__
T UnaryCudaLog(T x)
{
    return (T)log((float)x);
}

template<class T>
__device__
T UnaryCudaNegate(T x)
{
    return -x;
}

template<class T>
__device__
T UnaryCudaSign(T r)
{
    if (r > (T)0)
       return 1.0;
    else if (r == (T)0)
       return 0.0;
    else
       return -1.0;
}

template<class T>
__device__
T UnaryCudaSqrt(T x)
{
    return (T)sqrt((float)x);
}

template<class T>
__device__
T UnaryCudaSquare(T x)
{
    return x * x;
}

template<class T>
__device__
T UnaryCudaRound(T r)
{
	return (r > (T)0.0) ? (T)UnaryCudaFloor(r + (T)0.5) : (T)UnaryCudaCeil(r - (T)0.5);
}


template<class T>
__device__
T UnaryCudaSin(T x)
{
    return (T)sin((float)x);
}

template<class T>
__device__
T UnaryCudaCos(T x)
{
    return (T)cos((float)x);
}

template<class T>
__device__
T UnaryCudaTan(T x)
{
    return (T)tan((float)x);
}


#define SIMPLE_UNARY_FUNCTION_GPU(funcName, origFunc)                       \
template<class T>                                                           \
__global__                                                                  \
void Kernel##funcName(T * a, T * b, int size)                               \
{                                                                           \
    int i = blockDim.x * blockIdx.x + threadIdx.x;                          \
                                                                            \
    if (i < size)                                                           \
        b[i] = (T)origFunc(a[i]);                                           \
}                                                                           \
void _Cuda##funcName(const XTensor * a, XTensor * b)                        \
{                                                                           \
    CheckNTErrors((_IsSameShaped(a, b)),                            \
                  "Input tensors should have the same type!");              \
    CheckNTErrors(a->isSparse == false, "TODO!");                           \
                                                                            \
    int gridSize[3];                                                        \
    int blockSize[3];                                                       \
                                                                            \
    GDevs.GetCudaThread(a->devID, a->unitNum, gridSize, blockSize);         \
                                                                            \
    dim3 blocks(gridSize[0]);                                               \
    dim3 threads(blockSize[0]);                                             \
                                                                            \
    int devIDBackup;                                                        \
    ProtectCudaDev(a->devID, devIDBackup);                                  \
                                                                            \
    if (a->dataType == X_FLOAT) {                                           \
        Kernel##funcName<<<blocks, threads>>>                               \
                         ((float*)a->data, (float*)b->data, a->unitNum);    \
    }                                                                       \
    else if (a->dataType == X_DOUBLE) {                                     \
        Kernel##funcName<<<blocks, threads>>>                               \
                         ((double*)a->data, (double*)b->data, a->unitNum);  \
    }                                                                       \
    else if (a->dataType == X_INT) {                                        \
        Kernel##funcName<<<blocks, threads>>>                               \
                         ((int*)a->data, (int*)b->data, a->unitNum);        \
    }                                                                       \
    else {                                                                  \
        ShowNTErrors("TODO!");                                              \
    }                                                                       \
                                                                            \
    BacktoCudaDev(a->devID, devIDBackup);                                   \
}



SIMPLE_UNARY_FUNCTION_GPU(Absolute, UnaryCudaFabs)
SIMPLE_UNARY_FUNCTION_GPU(Ceil, UnaryCudaCeil)
SIMPLE_UNARY_FUNCTION_GPU(Exp, UnaryCudaExp)
SIMPLE_UNARY_FUNCTION_GPU(Floor, UnaryCudaFloor)
SIMPLE_UNARY_FUNCTION_GPU(IsNonZero, UnaryCudaIsNonZero)
SIMPLE_UNARY_FUNCTION_GPU(IsZero, UnaryCudaIsZero)
SIMPLE_UNARY_FUNCTION_GPU(Log, UnaryCudaLog)
SIMPLE_UNARY_FUNCTION_GPU(Negate, UnaryCudaNegate)
SIMPLE_UNARY_FUNCTION_GPU(Round, UnaryCudaRound)
SIMPLE_UNARY_FUNCTION_GPU(Sign, UnaryCudaSign)
SIMPLE_UNARY_FUNCTION_GPU(Sqrt, UnaryCudaSqrt)
SIMPLE_UNARY_FUNCTION_GPU(Square, UnaryCudaSquare)

SIMPLE_UNARY_FUNCTION_GPU(Sin, UnaryCudaSin)
SIMPLE_UNARY_FUNCTION_GPU(Cos, UnaryCudaCos)
SIMPLE_UNARY_FUNCTION_GPU(Tan, UnaryCudaTan)

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)