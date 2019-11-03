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
#include "Compare.h"
#include "Compare.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

__device__
DTYPE cudaIsEqual(DTYPE a, DTYPE b)
{
    return (a == b ? 1.0F : 0.0F);
}

__device__
DTYPE cudaIsNotEqual(DTYPE a, DTYPE b)
{
    return (a != b ? 1.0F : 0.0F);
}

#define SIMPLE_COMPARE_FUNCTION_GPU(funcName, origFunc)                     \
__global__                                                                  \
void Kernel##funcName(DTYPE * a, DTYPE * b, int size, DTYPE number)         \
{                                                                           \
    int i = blockDim.x * blockIdx.x + threadIdx.x;                          \
                                                                            \
    if (i < size)                                                           \
        b[i] = (DTYPE)origFunc(a[i], number);                               \
}                                                                           \
__global__                                                                  \
void Kernel##funcName(__half * a, __half * b, int size, __half number)      \
{                                                                           \
    return;                                                                 \
}                                                                           \
void _Cuda##funcName(const XTensor * a, XTensor * b, DTYPE number)          \
{                                                                           \
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
    if (a->dataType == DEFAULT_DTYPE) {                                     \
        Kernel##funcName<<<blocks, threads>>>                               \
                         ((DTYPE*)a->data, (DTYPE*)b->data,                 \
                           a->unitNum, (DTYPE)number);                      \
    }                                                                       \
    else if (a->dataType == X_FLOAT16) {                                    \
        Kernel##funcName<<<blocks, threads>>>                               \
                         ((__half*)a->data, (__half*)b->data,               \
                           a->unitNum, (__half)number);                     \
    }                                                                       \
    else {                                                                  \
        ShowNTErrors("TODO!");                                              \
    }                                                                       \
                                                                            \
    BacktoCudaDev(a->devID, devIDBackup);                                   \
}                                                                           \

SIMPLE_COMPARE_FUNCTION_GPU(Equal, cudaIsEqual)
SIMPLE_COMPARE_FUNCTION_GPU(NotEqual, cudaIsNotEqual)

#define SIMPLE_MAX_MIN_FUNCTION_GPU(funcName, origFunc)                     \
__global__                                                                  \
void Kernel##funcName(DTYPE * a, DTYPE * b, DTYPE * c, int size)            \
{                                                                           \
    int i = blockDim.x * blockIdx.x + threadIdx.x;                          \
                                                                            \
    if (i < size)                                                           \
        c[i] = (DTYPE)origFunc(a[i], b[i]);                                 \
}                                                                           \
__global__                                                                  \
void Kernel##funcName(__half * a, __half * b, __half * c, int size)         \
{                                                                           \
    return;                                                                 \
}                                                                           \
void _Cuda##funcName(const XTensor * a, const XTensor * b, XTensor * c)     \
{                                                                           \
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
    if (a->dataType == DEFAULT_DTYPE) {                                     \
        Kernel##funcName<<<blocks, threads>>>                               \
                         ((DTYPE*)a->data, (DTYPE*)b->data,                 \
                          (DTYPE*)c->data, a->unitNum);                     \
    }                                                                       \
    else if (a->dataType == X_FLOAT16) {                                    \
        Kernel##funcName<<<blocks, threads>>>                               \
                         ((__half*)a->data, (__half*)b->data,               \
                          (__half*)c->data, a->unitNum);                    \
    }                                                                       \
    else {                                                                  \
        ShowNTErrors("TODO!");                                              \
    }                                                                       \
                                                                            \
    BacktoCudaDev(a->devID, devIDBackup);                                   \
}    

SIMPLE_MAX_MIN_FUNCTION_GPU(Max, max)
SIMPLE_MAX_MIN_FUNCTION_GPU(Min, min)

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)