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
#include "../../XName.h"
#include "Unary.h"
#include "Unary.cuh"

namespace nts{
    
DTYPE square(DTYPE x)
{
    return x * x;
}

DTYPE round(DTYPE r)
{
	return (r > 0.0) ? (DTYPE)floor(r + 0.5) : (DTYPE)ceil(r - 0.5);
}

DTYPE isnonzero(DTYPE r)
{
    return (r != 0.0) ? (DTYPE)1.0 : (DTYPE)0.0;
}

DTYPE iszero(DTYPE r)
{
    return (r == 0.0) ? (DTYPE)1.0 : (DTYPE)0.0;
}

#ifdef USE_CUDA
/* define three marco separately, specify the respective function names  (GPU mode) */
#define _SIMPLE_UNARY_FUNCTION(_funcName, _cudaFuncName, origFunc)          \
void _funcName(const XTensor * a, XTensor * b)                              \
{                                                                           \
    /* run it on GPUs */                                                    \
    if (a->devID >= 0) {                                                    \
        _cudaFuncName(a, b);                                                \
    return;                                                                 \
    }                                                                       \
    CheckNTErrors((XTensor::IsSameShaped(a, b)),                            \
                  "Input tensors should have the same type!");              \
    CheckNTErrors((a->dataType == DEFAULT_DTYPE), "TODO!");                 \
    DTYPE * d = (DTYPE*)a->data;                                            \
    DTYPE * db = (DTYPE*)b->data;                                           \
    for (int i = 0; i < a->unitNum; i++)                                    \
        db[i] = (DTYPE)origFunc(d[i]);                                      \
}

#define _SIMPLE_UNARY_FUNCTION_ME(_funcNameMe, _funcName)                   \
void _funcNameMe(XTensor * a)                                               \
{                                                                           \
    _funcName(a, a);                                                        \
}        

#define SIMPLE_UNARY_FUNCTION(funcName, _funcName, operationId)             \
XTensor funcName(const XTensor &a)                                          \
{                                                                           \
    XTensor b(&a);                                                          \
    b.SetTMPFlag();                                                         \
    _funcName(&a, &b);                                                      \
    XLink::MakeLink(&a, NULL, &b, operationId);                             \
    return b;                                                               \
}

_SIMPLE_UNARY_FUNCTION(_Absolute, _CudaAbsolute, fabs)
_SIMPLE_UNARY_FUNCTION_ME(_AbsoluteMe, _Absolute)
SIMPLE_UNARY_FUNCTION(Absolute, _Absolute, MATH_ABSOLUTE)

_SIMPLE_UNARY_FUNCTION(_Ceil, _CudaCeil, ceil)
_SIMPLE_UNARY_FUNCTION_ME(_CeilMe, _Ceil)
SIMPLE_UNARY_FUNCTION(Ceil, _Ceil, MATH_CEIL)

_SIMPLE_UNARY_FUNCTION(_Exp, _CudaExp, exp)
_SIMPLE_UNARY_FUNCTION_ME(_ExpMe, _Exp)
SIMPLE_UNARY_FUNCTION(Exp, _Exp, MATH_EXP)

_SIMPLE_UNARY_FUNCTION(_Floor, _CudaFloor, floor)
_SIMPLE_UNARY_FUNCTION_ME(_FloorMe, _Floor)
SIMPLE_UNARY_FUNCTION(Floor, _Floor, MATH_FLOOR)

_SIMPLE_UNARY_FUNCTION(_IsNonZero, _CudaIsNonZero, isnonzero)
_SIMPLE_UNARY_FUNCTION_ME(_IsNonZeroMe, _IsNonZero)
SIMPLE_UNARY_FUNCTION(IsNonZero, _IsNonZero, MATH_ISNONZERO)

_SIMPLE_UNARY_FUNCTION(_IsZero, _CudaIsZero, iszero)
_SIMPLE_UNARY_FUNCTION_ME(_IsZeroMe, _IsZero)
SIMPLE_UNARY_FUNCTION(IsZero, _IsZero, MATH_ISZERO)

_SIMPLE_UNARY_FUNCTION(_Log, _CudaLog, log)
_SIMPLE_UNARY_FUNCTION_ME(_LogMe, _Log)
SIMPLE_UNARY_FUNCTION(Log, _Log, MATH_LOG)

_SIMPLE_UNARY_FUNCTION(_Round, _CudaRound, round)
_SIMPLE_UNARY_FUNCTION_ME(_RoundMe, _Round)
SIMPLE_UNARY_FUNCTION(Round, _Round, MATH_ROUND)

_SIMPLE_UNARY_FUNCTION(_Sqrt, _CudaSqrt, sqrt)
_SIMPLE_UNARY_FUNCTION_ME(_SqrtMe, _Sqrt)
SIMPLE_UNARY_FUNCTION(Sqrt, _Sqrt, MATH_SQRT)

_SIMPLE_UNARY_FUNCTION(_Square, _CudaSquare, square)
_SIMPLE_UNARY_FUNCTION_ME(_SquareMe, _Square)
SIMPLE_UNARY_FUNCTION(Square, _Square, MATH_SQUARE)


_SIMPLE_UNARY_FUNCTION(_Sin, _CudaSin, sin)
_SIMPLE_UNARY_FUNCTION_ME(_SinMe, _Sin)
SIMPLE_UNARY_FUNCTION(Sin, _Sin, MATH_SIN)

_SIMPLE_UNARY_FUNCTION(_Cos, _CudaCos, cos)
_SIMPLE_UNARY_FUNCTION_ME(_CosMe, _Cos)
SIMPLE_UNARY_FUNCTION(Cos, _Cos, MATH_COS)

_SIMPLE_UNARY_FUNCTION(_Tan, _CudaTan, tan)
_SIMPLE_UNARY_FUNCTION_ME(_TanMe, _Tan)
SIMPLE_UNARY_FUNCTION(Tan, _Tan, MATH_TAN)

#else
/* define three marco separately, specify the respective function names (CPU mode) */
#define _SIMPLE_UNARY_FUNCTION(_funcName, origFunc)          \
void _funcName(const XTensor * a, XTensor * b)                              \
{                                                                           \
    CheckNTErrors((XTensor::IsSameShaped(a, b)),                            \
                  "Input tensors should have the same type!");              \
    CheckNTErrors((a->dataType == DEFAULT_DTYPE), "TODO!");                 \
    DTYPE * d = (DTYPE*)a->data;                                            \
    DTYPE * db = (DTYPE*)b->data;                                           \
    for (int i = 0; i < a->unitNum; i++)                                    \
        db[i] = (DTYPE)origFunc(d[i]);                                      \
}

#define _SIMPLE_UNARY_FUNCTION_ME(_funcNameMe, _funcName)                   \
void _funcNameMe(XTensor * a)                                               \
{                                                                           \
    _funcName(a, a);                                                        \
}        

#define SIMPLE_UNARY_FUNCTION(funcName, _funcName, operationId)             \
XTensor funcName(const XTensor &a)                                          \
{                                                                           \
    XTensor b(&a);                                                          \
    b.SetTMPFlag();                                                         \
    _funcName(&a, &b);                                                      \
    XLink::MakeLink(&a, NULL, &b, operationId);                             \
    return b;                                                               \
}

_SIMPLE_UNARY_FUNCTION(_Absolute, fabs)
_SIMPLE_UNARY_FUNCTION_ME(_AbsoluteMe, _Absolute)
SIMPLE_UNARY_FUNCTION(Absolute, _Absolute, MATH_ABSOLUTE)


_SIMPLE_UNARY_FUNCTION(_Ceil, ceil)
_SIMPLE_UNARY_FUNCTION_ME(_CeilMe, _Ceil)
SIMPLE_UNARY_FUNCTION(Ceil, _Ceil, MATH_CEIL)

_SIMPLE_UNARY_FUNCTION(_Exp, exp)
_SIMPLE_UNARY_FUNCTION_ME(_ExpMe, _Exp)
SIMPLE_UNARY_FUNCTION(Exp, _Exp, MATH_EXP)

_SIMPLE_UNARY_FUNCTION(_Floor, floor)
_SIMPLE_UNARY_FUNCTION_ME(_FloorMe, _Floor)
SIMPLE_UNARY_FUNCTION(Floor, _Floor, MATH_FLOOR)

_SIMPLE_UNARY_FUNCTION(_IsNonZero, isnonzero)
_SIMPLE_UNARY_FUNCTION_ME(_IsNonZeroMe, _IsNonZero)
SIMPLE_UNARY_FUNCTION(IsNonZero, _IsNonZero, MATH_ISNONZERO)

_SIMPLE_UNARY_FUNCTION(_IsZero, iszero)
_SIMPLE_UNARY_FUNCTION_ME(_IsZeroMe, _IsZero)
SIMPLE_UNARY_FUNCTION(IsZero, _IsZero, MATH_ISZERO)

_SIMPLE_UNARY_FUNCTION(_Log, log)
_SIMPLE_UNARY_FUNCTION_ME(_LogMe, _Log)
SIMPLE_UNARY_FUNCTION(Log, _Log, MATH_LOG)

_SIMPLE_UNARY_FUNCTION(_Round, round)
_SIMPLE_UNARY_FUNCTION_ME(_RoundMe, _Round)
SIMPLE_UNARY_FUNCTION(Round, _Round, MATH_ROUND)

_SIMPLE_UNARY_FUNCTION(_Sqrt, sqrt)
_SIMPLE_UNARY_FUNCTION_ME(_SqrtMe, _Sqrt)
SIMPLE_UNARY_FUNCTION(Sqrt, _Sqrt, MATH_SQRT)

_SIMPLE_UNARY_FUNCTION(_Square, square)
_SIMPLE_UNARY_FUNCTION_ME(_SquareMe, _Square)
SIMPLE_UNARY_FUNCTION(Square, _Square, MATH_SQUARE)

_SIMPLE_UNARY_FUNCTION(_Sin, sin)
_SIMPLE_UNARY_FUNCTION_ME(_SinMe, _Sin)
SIMPLE_UNARY_FUNCTION(Sin, _Sin, MATH_SIN)

_SIMPLE_UNARY_FUNCTION(_Cos, cos)
_SIMPLE_UNARY_FUNCTION_ME(_CosMe, _Cos)
SIMPLE_UNARY_FUNCTION(Cos, _Cos, MATH_COS)

_SIMPLE_UNARY_FUNCTION(_Tan, tan)
_SIMPLE_UNARY_FUNCTION_ME(_TanMe, _Tan)
SIMPLE_UNARY_FUNCTION(Tan, _Tan, MATH_TAN)

/*_SIMPLE_UNARY_FUNCTION(_Round, round)
_SIMPLE_UNARY_FUNCTION_ME(_RoundMe, _Round)
SIMPLE_UNARY_FUNCTION(Round, _Round, MATH_ROUND)*/
#endif

} // namespace nts(NiuTrans.Tensor)