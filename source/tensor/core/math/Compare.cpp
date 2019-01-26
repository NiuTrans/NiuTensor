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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-12-10
 */

#include "../../XTensor.h"
#include "../../XName.h"
#include "Compare.h"
#include "Compare.cuh"

namespace nts{ // namespace nts(NiuTrans.Tensor)

DTYPE myIsEqual(DTYPE a, DTYPE b)
{
    return (a == b ? 1.0F : 0.0F);
}

DTYPE myIsNotEqual(DTYPE a, DTYPE b)
{
    return (a != b ? 1.0F : 0.0F);
}

#ifdef USE_CUDA
/* define three marco separately, specify the respective function names  (GPU mode) */
#define _SIMPLE_COMPARE_FUNCTION(_funcName, _cudaFuncName, origFunc)        \
void _funcName(const XTensor * a, XTensor * b, DTYPE number)                \
{                                                                           \
    CheckNTErrors((XTensor::IsSameShaped(a, b)),                            \
                  "Input tensors should have the same type!");              \
    CheckNTErrors((a->dataType == DEFAULT_DTYPE), "TODO!");                 \
    /* run it on GPUs */                                                    \
    if (a->devID >= 0) {                                                    \
        _cudaFuncName(a, b, number);                                        \
        return;                                                             \
    }                                                                       \
    DTYPE * d = (DTYPE*)a->data;                                            \
    DTYPE * db = (DTYPE*)b->data;                                           \
    for (int i = 0; i < a->unitNum; i++)                                    \
        db[i] = (DTYPE)origFunc(d[i], number);                              \
}

#define _SIMPLE_COMPARE_FUNCTION_ME(_funcNameMe, _funcName)                 \
void _funcNameMe(XTensor * a, DTYPE number)                                 \
{                                                                           \
    _funcName(a, a, number);                                                \
}        

#define SIMPLE_COMPARE_FUNCTION(funcName, _funcName, operationId)           \
XTensor funcName(const XTensor &a, DTYPE number)                            \
{                                                                           \
    XTensor b(&a);                                                          \
    b.SetTMPFlag();                                                         \
    _funcName(&a, &b, number);                                              \
    return b;                                                               \
}
// I think we needn't to make link.
// XLink::MakeLink(&a, NULL, &b, operationId);

_SIMPLE_COMPARE_FUNCTION(_Equal, _CudaEqual, myIsEqual)
_SIMPLE_COMPARE_FUNCTION_ME(_EqualMe, _Equal)
SIMPLE_COMPARE_FUNCTION(Equal, _Equal, MATH_EQUAL)

_SIMPLE_COMPARE_FUNCTION(_NotEqual, _CudaNotEqual, myIsNotEqual)
_SIMPLE_COMPARE_FUNCTION_ME(_NotEqualMe, _NotEqual)
SIMPLE_COMPARE_FUNCTION(NotEqual, _NotEqual, MATH_NOTEQUAL)

#else
/* define three marco separately, specify the respective function names (CPU mode) */
#define _SIMPLE_COMPARE_FUNCTION(_funcName, origFunc)                       \
void _funcName(const XTensor * a, XTensor * b, DTYPE number)                \
{                                                                           \
    CheckNTErrors((XTensor::IsSameShaped(a, b)),                            \
                  "Input tensors should have the same type!");              \
    CheckNTErrors((a->dataType == DEFAULT_DTYPE), "TODO!");                 \
    DTYPE * d = (DTYPE*)a->data;                                            \
    DTYPE * db = (DTYPE*)b->data;                                           \
    for (int i = 0; i < a->unitNum; i++)                                    \
        db[i] = (DTYPE)origFunc(d[i], number);                              \
}

#define _SIMPLE_COMPARE_FUNCTION_ME(_funcNameMe, _funcName)                 \
void _funcNameMe(XTensor * a, DTYPE number)                                 \
{                                                                           \
    _funcName(a, a, number);                                                \
}        

#define SIMPLE_COMPARE_FUNCTION(funcName, _funcName, operationId)           \
XTensor funcName(const XTensor &a, DTYPE number)                            \
{                                                                           \
    XTensor b(&a);                                                          \
    b.SetTMPFlag();                                                         \
    _funcName(&a, &b, number);                                              \
    return b;                                                               \
}

// I think we needn't to make link.
// XLink::MakeLink(&a, NULL, &b, operationId);

_SIMPLE_COMPARE_FUNCTION(_Equal, myIsEqual)
_SIMPLE_COMPARE_FUNCTION_ME(_EqualMe, _Equal)
SIMPLE_COMPARE_FUNCTION(Equal, _Equal, MATH_EQUAL)

_SIMPLE_COMPARE_FUNCTION(_NotEqual, myIsNotEqual)
_SIMPLE_COMPARE_FUNCTION_ME(_NotEqualMe, _NotEqual)
SIMPLE_COMPARE_FUNCTION(NotEqual, _NotEqual, MATH_NOTEQUAL)

#endif

} // namespace nts(NiuTrans.Tensor)