/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2017, Natural Language Processing Lab, Northeastern University.
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
#include "../../XDevice.h"
#include "../../XName.h"
#include "../shape/IsSameShaped.h"
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

/* define three marco separately, specify the respective function names */
#ifdef USE_CUDA
#define _SIMPLE_COMPARE_FUNCTION(_funcName, _cudaFuncName, origFunc)                 \
void _funcName(const XTensor * a, XTensor * b, DTYPE number)                         \
{                                                                                    \
    CheckNTErrors((_IsSameShaped(a, b)),                                             \
                  "Input tensors should have the same type!");                       \
    CheckNTErrors((a->dataType == DEFAULT_DTYPE), "TODO!");                          \
    /* run it on GPUs */                                                             \
    if (a->devID >= 0) {                                                             \
        _cudaFuncName(a, b, number);                                                 \
        return;                                                                      \
    }                                                                                \
    DTYPE * d = (DTYPE*)a->data;                                                     \
    DTYPE * db = (DTYPE*)b->data;                                                    \
    for (int i = 0; i < a->unitNum; i++)                                             \
        db[i] = (DTYPE)origFunc(d[i], number);                                       \
}     
#else
#define _SIMPLE_COMPARE_FUNCTION(_funcName, origFunc)                                \
void _funcName(const XTensor * a, XTensor * b, DTYPE number)                         \
{                                                                                    \
    CheckNTErrors((_IsSameShaped(a, b)),                                             \
                  "Input tensors should have the same type!");                       \
    CheckNTErrors((a->dataType == DEFAULT_DTYPE), "TODO!");                          \
    /* run it on GPUs */                                                             \
    if (a->devID >= 0) {                                                             \
        ShowNTErrors("No GPU devices support!")                                      \
    }                                                                                \
    DTYPE * d = (DTYPE*)a->data;                                                     \
    DTYPE * db = (DTYPE*)b->data;                                                    \
    for (int i = 0; i < a->unitNum; i++)                                             \
        db[i] = (DTYPE)origFunc(d[i], number);                                       \
}     
#endif
                                                                                     
#define _SIMPLE_COMPARE_FUNCTION_ME(_funcNameMe, _funcName)                          \
void _funcNameMe(XTensor * a, DTYPE number)                                          \
{                                                                                    \
    _funcName(a, a, number);                                                         \
}                                                                                    
                                                                                        
#define SIMPLE_COMPARE_FUNCTION_ME(funcNameMe, _funcName)                            \
void funcNameMe(XTensor & a, DTYPE number)                                           \
{                                                                                    \
    _funcName(&a, &a, number);                                                       \
}                                                                                    
                                                                                     
#define SIMPLE_COMPARE_FUNCTION(funcName, _funcName, operationId)                    \
XTensor funcName(const XTensor &a, DTYPE number)                                     \
{                                                                                    \
    XTensor b(&a);                                                                   \
    b.SetTMPFlag();                                                                  \
    _funcName(&a, &b, number);                                                       \
    return b;                                                                        \
}
                                                                                     
#define SIMPLE_COMPARE_FUNCTION_VOID(funcName, _funcName, operationId)               \
void funcName(const XTensor &a, XTensor &b, DTYPE number)                            \
{                                                                                    \
    if (!b.isInit || !IsSameShaped(a, b)) {                                          \
        InitTensorV2(&b, &a);                                                        \
    }                                                                                \
    _funcName(&a, &b, number);                                                       \
}

// I think we needn't to make link.
// XLink::MakeLink(&a, NULL, &b, operationId);

#ifdef USE_CUDA
_SIMPLE_COMPARE_FUNCTION(_Equal, _CudaEqual, myIsEqual)
_SIMPLE_COMPARE_FUNCTION(_NotEqual, _CudaNotEqual, myIsNotEqual)
#else
_SIMPLE_COMPARE_FUNCTION(_Equal, myIsEqual)
_SIMPLE_COMPARE_FUNCTION(_NotEqual, myIsNotEqual)
#endif

_SIMPLE_COMPARE_FUNCTION_ME(_EqualMe, _Equal)
SIMPLE_COMPARE_FUNCTION_ME(EqualMe, _Equal)
SIMPLE_COMPARE_FUNCTION(Equal, _Equal, MATH_EQUAL)
SIMPLE_COMPARE_FUNCTION_VOID(Equal, _Equal, MATH_EQUAL)

_SIMPLE_COMPARE_FUNCTION_ME(_NotEqualMe, _NotEqual)
SIMPLE_COMPARE_FUNCTION_ME(NotEqualMe, _NotEqual)
SIMPLE_COMPARE_FUNCTION(NotEqual, _NotEqual, MATH_NOTEQUAL)
SIMPLE_COMPARE_FUNCTION_VOID(NotEqual, _NotEqual, MATH_NOTEQUAL)


/* define three marco separately, specify the respective function names */
#ifdef USE_CUDA
#define _SIMPLE_MAX_MIN_FUNCTION(_funcName, _cudaFuncName, origFunc)                 \
void _funcName(const XTensor * a, const XTensor * b,  XTensor * c)                   \
{                                                                                    \
    CheckNTErrors((_IsSameShaped(a, b, c)),                                          \
                  "Input and output tensors should have the same type!");            \
    CheckNTErrors((a->dataType == DEFAULT_DTYPE), "TODO!");                          \
    CheckDev(a->devID, b->devID);                                                    \
    CheckDev(a->devID, c->devID);                                                    \
    /* run it on GPUs */                                                             \
    if (a->devID >= 0) {                                                             \
        _cudaFuncName(a, b, c);                                                      \
        return;                                                                      \
    }                                                                                \
    DTYPE * da = (DTYPE*)a->data;                                                    \
    DTYPE * db = (DTYPE*)b->data;                                                    \
    DTYPE * dc = (DTYPE*)c->data;                                                    \
    for (int i = 0; i < a->unitNum; i++)                                             \
        dc[i] = (DTYPE)origFunc(da[i], db[i]);                                       \
}     
#else
#define _SIMPLE_MAX_MIN_FUNCTION(_funcName, origFunc)                                \
void _funcName(const XTensor * a, const XTensor * b, XTensor *c)                     \
{                                                                                    \
    CheckNTErrors((_IsSameShaped(a, b, c)),                                          \
                  "Input and output tensors should have the same type!");            \
    CheckNTErrors((a->dataType == DEFAULT_DTYPE), "TODO!");                          \
    CheckDev(a->devID, b->devID);                                                    \
    CheckDev(a->devID, c->devID);                                                    \
    /* run it on GPUs */                                                             \
    if (a->devID >= 0) {                                                             \
        ShowNTErrors("No GPU devices support!")                                      \
    }                                                                                \
    DTYPE * da = (DTYPE*)a->data;                                                    \
    DTYPE * db = (DTYPE*)b->data;                                                    \
    DTYPE * dc = (DTYPE*)c->data;                                                    \
    for (int i = 0; i < a->unitNum; i++)                                             \
        dc[i] = (DTYPE)origFunc(da[i], db[i]);                                       \
}     
#endif
                                                                                     
#define _SIMPLE_MAX_MIN_FUNCTION_ME(_funcNameMe, _funcName)                          \
void _funcNameMe(XTensor * a, const XTensor * b)                                     \
{                                                                                    \
    _funcName(a, b, a);                                                              \
}                                                                                    
                                                                                        
#define SIMPLE_MAX_MIN_FUNCTION_ME(funcNameMe, _funcName)                            \
void funcNameMe(XTensor & a, const XTensor & b)                                      \
{                                                                                    \
    _funcName(&a, &b, &a);                                                           \
}                                                                                    
                                                                                     
#define SIMPLE_MAX_MIN_FUNCTION(funcName, _funcName, operationId)                    \
XTensor funcName(const XTensor & a, const XTensor & b)                               \
{                                                                                    \
    XTensor c(&a);                                                                   \
    c.SetTMPFlag();                                                                  \
    _funcName(&a, &b, &c);                                                           \
    return c;                                                                        \
}
                                                                                     
#define SIMPLE_MAX_MIN_FUNCTION_VOID(funcName, _funcName, operationId)               \
void funcName(const XTensor &a, const XTensor &b, XTensor c)                         \
{                                                                                    \
    if (!c.isInit || !_IsSameShaped(&a, &c)) {                                       \
        InitTensor(&c, &a);                                                          \
    }                                                                                \
    _funcName(&a, &b, &c);                                                           \
}

#ifdef USE_CUDA
_SIMPLE_MAX_MIN_FUNCTION(_Equal, _CudaEqual, myIsEqual)
_SIMPLE_MAX_MIN_FUNCTION(_NotEqual, _CudaNotEqual, myIsNotEqual)
_SIMPLE_MAX_MIN_FUNCTION(_Max, _CudaMax, MAX)
_SIMPLE_MAX_MIN_FUNCTION(_Min, _CudaMin, MIN)
#else
_SIMPLE_MAX_MIN_FUNCTION(_Equal, myIsEqual)
_SIMPLE_MAX_MIN_FUNCTION(_NotEqual, myIsNotEqual)
_SIMPLE_MAX_MIN_FUNCTION(_Max, MAX)
_SIMPLE_MAX_MIN_FUNCTION(_Min, MIN)
#endif

_SIMPLE_MAX_MIN_FUNCTION_ME(_EqualMe, _Equal)
SIMPLE_MAX_MIN_FUNCTION_ME(EqualMe, _Equal)
SIMPLE_MAX_MIN_FUNCTION(Equal, _Equal, MATH_EQUAL)
SIMPLE_MAX_MIN_FUNCTION_VOID(Equal, _Equal, MATH_EQUAL)

_SIMPLE_MAX_MIN_FUNCTION_ME(_NotEqualMe, _NotEqual)
SIMPLE_MAX_MIN_FUNCTION_ME(NotEqualMe, _NotEqual)
SIMPLE_MAX_MIN_FUNCTION(NotEqual, _NotEqual, MATH_NOTEQUAL)
SIMPLE_MAX_MIN_FUNCTION_VOID(NotEqual, _NotEqual, MATH_NOTEQUAL)

_SIMPLE_MAX_MIN_FUNCTION_ME(_MaxMe, _Max)
SIMPLE_MAX_MIN_FUNCTION_ME(MaxMe, _Max)
SIMPLE_MAX_MIN_FUNCTION(Max, _Max, MATH_MAX)
SIMPLE_MAX_MIN_FUNCTION_VOID(Max, _Max, MATH_MAX)

_SIMPLE_MAX_MIN_FUNCTION_ME(_MinMe, _Min)
SIMPLE_MAX_MIN_FUNCTION_ME(MinMe, _Min)
SIMPLE_MAX_MIN_FUNCTION(Min, _Min, MATH_MIN)
SIMPLE_MAX_MIN_FUNCTION_VOID(Min, _Min, MATH_MIN)

} // namespace nts(NiuTrans.Tensor)
