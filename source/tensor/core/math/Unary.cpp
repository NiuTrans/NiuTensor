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
#include "../shape/IsSameShaped.h"
#include "Unary.h"
#include "Unary.cuh"

namespace nts{
  
template<class T>
T UnaryNegate(T x) {
    return (T)-x;
}

template<class T>
T UnarySquare(T x)
{
    return (T)(x * x);
}

template<class T>
T UnaryRound(T r)
{
	return (r > 0.0) ? (T)floor(r + 0.5) : (T)ceil(r - 0.5);
}

template<class T>
T UnarySign(T r)
{
    if (r > 0.0)
       return (T)1.0;
    else if (r == 0.0)
       return (T)0.0;
    else
       return (T)-1.0;
}

template<class T>
T UnaryIsNonZero(T r)
{
    return (r != 0.0) ? (T)1.0 : (T)0.0;
}

template<class T>
T UnaryIsZero(T r)
{
    return (r == 0.0) ? (T)1.0 : (T)0.0;
}

/* define three marco separately, specify the respective function names */
#ifdef USE_CUDA
#define _SIMPLE_UNARY_FUNCTION(_funcName, _cudaFuncName, origFunc)                   \
void _funcName(const XTensor * a, XTensor * b)                                       \
{                                                                                    \
    /* run it on GPUs */                                                             \
    if (a->devID >= 0) {                                                             \
        _cudaFuncName(a, b);                                                         \
        return;                                                                      \
    }                                                                                \
    CheckNTErrors((_IsSameShaped(a, b)),                                              \
                  "Input tensors should have the same type!");                       \
    if (a->dataType == X_INT) {                                                      \
        int * d = (int*)a->data;                                                     \
        int * db = (int*)b->data;                                                    \
        for (int i = 0; i < a->unitNum; i++)                                         \
            db[i] = (int)origFunc(d[i]);                                             \
    }                                                                                \
    else if (a->dataType == X_FLOAT) {                                               \
        float * d = (float*)a->data;                                                 \
        float * db = (float*)b->data;                                                \
        for (int i = 0; i < a->unitNum; i++)                                         \
            db[i] = (float)origFunc(d[i]);                                           \
    }                                                                                \
    else if (a->dataType == X_DOUBLE) {                                              \
        double * d = (double*)a->data;                                               \
        double * db = (double*)b->data;                                              \
        for (int i = 0; i < a->unitNum; i++)                                         \
            db[i] = (double)origFunc(d[i]);                                          \
    }                                                                                \
    else                                                                             \
        ShowNTErrors("TO DO!");                                                      \
}                                       
#else
#define _SIMPLE_UNARY_FUNCTION(_funcName, origFunc)                                  \
void _funcName(const XTensor * a, XTensor * b)                                       \
{                                                                                    \
    /* run it on GPUs */                                                             \
    if (a->devID >= 0) {                                                             \
        ShowNTErrors("No GPU devices support!")                                      \
    }                                                                                \
    CheckNTErrors((_IsSameShaped(a, b)),                                              \
                  "Input tensors should have the same type!");                       \
    if (a->dataType == X_INT) {                                                      \
        int * d = (int*)a->data;                                                     \
        int * db = (int*)b->data;                                                    \
        for (int i = 0; i < a->unitNum; i++)                                         \
            db[i] = (int)origFunc(d[i]);                                             \
    }                                                                                \
    else if (a->dataType == X_FLOAT) {                                               \
        float * d = (float*)a->data;                                                 \
        float * db = (float*)b->data;                                                \
        for (int i = 0; i < a->unitNum; i++)                                         \
            db[i] = (float)origFunc(d[i]);                                           \
    }                                                                                \
    else if (a->dataType == X_DOUBLE) {                                              \
        double * d = (double*)a->data;                                               \
        double * db = (double*)b->data;                                              \
        for (int i = 0; i < a->unitNum; i++)                                         \
            db[i] = (double)origFunc(d[i]);                                          \
    }                                                                                \
    else                                                                             \
        ShowNTErrors("TO DO!");                                                      \
}
#endif

#define _SIMPLE_UNARY_FUNCTION_ME(_funcNameMe, _funcName)                            \
void _funcNameMe(XTensor * a)                                                        \
{                                                                                    \
    _funcName(a, a);                                                                 \
}        

#define SIMPLE_UNARY_FUNCTION_ME(funcNameMe, _funcName)                              \
void funcNameMe(XTensor & a)                                                         \
{                                                                                    \
    _funcName(&a, &a);                                                               \
}                                                                                    
                                                                                     
#define SIMPLE_UNARY_FUNCTION(funcName, _funcName, operationId)                      \
XTensor funcName(const XTensor & a)                                                  \
{                                                                                    \
    XTensor b(&a);                                                                   \
    b.SetTMPFlag();                                                                  \
    _funcName(&a, &b);                                                               \
    if(a.enableGrad){                                                                \
        XLink::MakeLink(&a, NULL, &b, operationId);                                  \
    }                                                                                \
    return b;                                                                        \
}                                                                                    
                                                                                     
#define SIMPLE_UNARY_FUNCTION_VOID(funcName, _funcName, operationId)                 \
void funcName(const XTensor & a, XTensor & b)                                        \
{                                                                                    \
    if (!b.isInit || !IsSameShaped(a, b)) {                                        \
        InitTensorV2(&b, &a);                                                          \
    }                                                                                \
    _funcName(&a, &b);                                                               \
    if (a.enableGrad) {                                                              \
        XLink::MakeLink(&a, NULL, &b, operationId);                                  \
    }                                                                                \
}

#ifdef USE_CUDA
_SIMPLE_UNARY_FUNCTION(_Absolute, _CudaAbsolute, fabs)
_SIMPLE_UNARY_FUNCTION(_Ceil, _CudaCeil, ceil)
_SIMPLE_UNARY_FUNCTION(_Exp, _CudaExp, exp)
_SIMPLE_UNARY_FUNCTION(_Floor, _CudaFloor, floor)
_SIMPLE_UNARY_FUNCTION(_IsNonZero, _CudaIsNonZero, UnaryIsNonZero)
_SIMPLE_UNARY_FUNCTION(_IsZero, _CudaIsZero, UnaryIsZero)
_SIMPLE_UNARY_FUNCTION(_Log, _CudaLog, log)
_SIMPLE_UNARY_FUNCTION(_Negate, _CudaNegate, UnaryNegate)
_SIMPLE_UNARY_FUNCTION(_Round, _CudaRound, round)
_SIMPLE_UNARY_FUNCTION(_Sign, _CudaSign, UnarySign)
_SIMPLE_UNARY_FUNCTION(_Sqrt, _CudaSqrt, sqrt)
_SIMPLE_UNARY_FUNCTION(_Square, _CudaSquare, UnarySquare)
_SIMPLE_UNARY_FUNCTION(_Sin, _CudaSin, sin)
_SIMPLE_UNARY_FUNCTION(_Cos, _CudaCos, cos)
_SIMPLE_UNARY_FUNCTION(_Tan, _CudaTan, tan)
#else
_SIMPLE_UNARY_FUNCTION(_Absolute, fabs)
_SIMPLE_UNARY_FUNCTION(_Ceil, ceil)
_SIMPLE_UNARY_FUNCTION(_Exp, exp)
_SIMPLE_UNARY_FUNCTION(_Floor, floor)
_SIMPLE_UNARY_FUNCTION(_IsNonZero, UnaryIsNonZero)
_SIMPLE_UNARY_FUNCTION(_IsZero, UnaryIsZero)
_SIMPLE_UNARY_FUNCTION(_Log, log)
_SIMPLE_UNARY_FUNCTION(_Negate, UnaryNegate)
_SIMPLE_UNARY_FUNCTION(_Round, round)
_SIMPLE_UNARY_FUNCTION(_Sign, UnarySign)
_SIMPLE_UNARY_FUNCTION(_Sqrt, sqrt)
_SIMPLE_UNARY_FUNCTION(_Square, UnarySquare)
_SIMPLE_UNARY_FUNCTION(_Sin, sin)
_SIMPLE_UNARY_FUNCTION(_Cos, cos)
_SIMPLE_UNARY_FUNCTION(_Tan, tan)
#endif

_SIMPLE_UNARY_FUNCTION_ME(_AbsoluteMe, _Absolute)
SIMPLE_UNARY_FUNCTION_ME(AbsoluteMe, _Absolute)
SIMPLE_UNARY_FUNCTION(Absolute, _Absolute, MATH_ABSOLUTE)
SIMPLE_UNARY_FUNCTION_VOID(Absolute, _Absolute, MATH_ABSOLUTE)

_SIMPLE_UNARY_FUNCTION_ME(_CeilMe, _Ceil)
SIMPLE_UNARY_FUNCTION_ME(CeilMe, _Ceil)
SIMPLE_UNARY_FUNCTION(Ceil, _Ceil, MATH_CEIL)
SIMPLE_UNARY_FUNCTION_VOID(Ceil, _Ceil, MATH_CEIL)

_SIMPLE_UNARY_FUNCTION_ME(_ExpMe, _Exp)
SIMPLE_UNARY_FUNCTION_ME(ExpMe, _Exp)
SIMPLE_UNARY_FUNCTION(Exp, _Exp, MATH_EXP)
SIMPLE_UNARY_FUNCTION_VOID(Exp, _Exp, MATH_EXP)

_SIMPLE_UNARY_FUNCTION_ME(_FloorMe, _Floor)
SIMPLE_UNARY_FUNCTION_ME(FloorMe, _Floor)
SIMPLE_UNARY_FUNCTION(Floor, _Floor, MATH_FLOOR)
SIMPLE_UNARY_FUNCTION_VOID(Floor, _Floor, MATH_FLOOR)

_SIMPLE_UNARY_FUNCTION_ME(_IsNonZeroMe, _IsNonZero)
SIMPLE_UNARY_FUNCTION_ME(IsNonZeroMe, _IsNonZero)
SIMPLE_UNARY_FUNCTION(IsNonZero, _IsNonZero, MATH_ISNONZERO)
SIMPLE_UNARY_FUNCTION_VOID(IsNonZero, _IsNonZero, MATH_ISNONZERO)

_SIMPLE_UNARY_FUNCTION_ME(_IsZeroMe, _IsZero)
SIMPLE_UNARY_FUNCTION_ME(IsZeroMe, _IsZero)
SIMPLE_UNARY_FUNCTION(IsZero, _IsZero, MATH_ISZERO)
SIMPLE_UNARY_FUNCTION_VOID(IsZero, _IsZero, MATH_ISZERO)

_SIMPLE_UNARY_FUNCTION_ME(_LogMe, _Log)
SIMPLE_UNARY_FUNCTION_ME(LogMe, _Log)
SIMPLE_UNARY_FUNCTION(Log, _Log, MATH_LOG)
SIMPLE_UNARY_FUNCTION_VOID(Log, _Log, MATH_LOG)

_SIMPLE_UNARY_FUNCTION_ME(_NegateMe, _Negate)
SIMPLE_UNARY_FUNCTION_ME(NegateMe, _Negate)
SIMPLE_UNARY_FUNCTION(Negate, _Negate, MATH_NEGATE)
SIMPLE_UNARY_FUNCTION_VOID(Negate, _Negate, MATH_NEGATE)

_SIMPLE_UNARY_FUNCTION_ME(_RoundMe, _Round)
SIMPLE_UNARY_FUNCTION_ME(RoundMe, _Round)
SIMPLE_UNARY_FUNCTION(Round, _Round, MATH_ROUND)
SIMPLE_UNARY_FUNCTION_VOID(Round, _Round, MATH_ROUND)

_SIMPLE_UNARY_FUNCTION_ME(_SignMe, _Sign)
SIMPLE_UNARY_FUNCTION_ME(SignMe, _Sign)
SIMPLE_UNARY_FUNCTION(Sign, _Sign, MATH_SIGN)
SIMPLE_UNARY_FUNCTION_VOID(Sign, _Sign, MATH_SIGN)

_SIMPLE_UNARY_FUNCTION_ME(_SqrtMe, _Sqrt)
SIMPLE_UNARY_FUNCTION_ME(SqrtMe, _Sqrt)
SIMPLE_UNARY_FUNCTION(Sqrt, _Sqrt, MATH_SQRT)
SIMPLE_UNARY_FUNCTION_VOID(Sqrt, _Sqrt, MATH_SQRT)

_SIMPLE_UNARY_FUNCTION_ME(_SquareMe, _Square)
SIMPLE_UNARY_FUNCTION_ME(SquareMe, _Square)
SIMPLE_UNARY_FUNCTION(Square, _Square, MATH_SQUARE)
SIMPLE_UNARY_FUNCTION_VOID(Square, _Square, MATH_SQUARE)

_SIMPLE_UNARY_FUNCTION_ME(_SinMe, _Sin)
SIMPLE_UNARY_FUNCTION_ME(SinMe, _Sin)
SIMPLE_UNARY_FUNCTION(Sin, _Sin, MATH_SIN)
SIMPLE_UNARY_FUNCTION_VOID(Sin, _Sin, MATH_SIN)

_SIMPLE_UNARY_FUNCTION_ME(_CosMe, _Cos)
SIMPLE_UNARY_FUNCTION_ME(CosMe, _Cos)
SIMPLE_UNARY_FUNCTION(Cos, _Cos, MATH_COS)
SIMPLE_UNARY_FUNCTION_VOID(Cos, _Cos, MATH_COS)

_SIMPLE_UNARY_FUNCTION_ME(_TanMe, _Tan)
SIMPLE_UNARY_FUNCTION_ME(TanMe, _Tan)
SIMPLE_UNARY_FUNCTION(Tan, _Tan, MATH_TAN)
SIMPLE_UNARY_FUNCTION_VOID(Tan, _Tan, MATH_TAN)

} // namespace nts(NiuTrans.Tensor)