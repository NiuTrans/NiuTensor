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
#include "../../XName.h"
#include "../shape/IsSameShaped.h"
#include "Binary.h"
#include "Binary.cuh"

namespace nts {

template<class T1, class T2>
T1 BinaryDescale(T1 x, T2 num)
{
    return (T1)(x / num);
}

template<class T1, class T2>
T1 BinaryPower(T1 x, T2 num)
{
    if (num == 0)
        return (T1)1.0;
    else if (num == 0.5)
        return (T1)sqrt(x);
    else if (num == 2)
        return x * x;
    else {
        if (x == 0 && num < 0)
            return (T1)1e20F;
        else
            return (T1)pow(x, num);
    }
}

template<class T1, class T2>
T1 BinaryScale(T1 x, T2 num)
{
    return (T1)(x * num);
}

template<class T1, class T2>
T1 BinaryShift(T1 x, T2 num)
{
    return (T1)(x + num);
}

int BinaryMod(int x, int num)
{
    return x % num;
}

/* define three marco separately, specify the respective function names */
#ifdef USE_CUDA                                                                      
#define _SIMPLE_BINARY_FUNCTION(_funcName, _cudaFuncName, origFunc)                  \
template<class T>                                                                    \
void _funcName(const XTensor * a, XTensor * b, T num)                                \
{                                                                                    \
    /* run it on GPUs */                                                             \
    if (a->devID >= 0) {                                                             \
        _cudaFuncName(a, b, num);                                                    \
        return;                                                                      \
    }                                                                                \
    CheckNTErrors((_IsSameShaped(a, b)),                                             \
                  "Input tensors should have the same data type!");                  \
    if (a->dataType == X_INT) {                                                      \
        int * d = (int*)a->data;                                                     \
        int * db = (int*)b->data;                                                    \
        for (int i = 0; i < a->unitNum; i++)                                         \
            db[i] = (int)origFunc((int)d[i], (T)num);                                \
    }                                                                                \
    else if (a->dataType == X_FLOAT) {                                               \
        float * d = (float*)a->data;                                                 \
        float * db = (float*)b->data;                                                \
        for (int i = 0; i < a->unitNum; i++)                                         \
            db[i] = (float)origFunc((float)d[i], (T)num);                            \
    }                                                                                \
    else if (a->dataType == X_DOUBLE) {                                              \
        double * d = (double*)a->data;                                               \
        double * db = (double*)b->data;                                              \
        for (int i = 0; i < a->unitNum; i++)                                         \
            db[i] = (double)origFunc((double)d[i], (T)num);                          \
    }                                                                                \
    else                                                                             \
        ShowNTErrors("TO DO!");                                                      \
}                                                                                    \
template void _funcName<int>(const XTensor*, XTensor*, int);                         \
template void _funcName<float>(const XTensor*, XTensor*, float);                     \
template void _funcName<double>(const XTensor*, XTensor*, double);                   
#else
#define _SIMPLE_BINARY_FUNCTION(_funcName, origFunc)                                 \
template<class T>                                                                    \
void _funcName(const XTensor * a, XTensor * b, T num)                                \
{                                                                                    \
    /* run it on GPUs */                                                             \
    if (a->devID >= 0) {                                                             \
        ShowNTErrors("No GPU devices support!")                                      \
    }                                                                                \
    CheckNTErrors((_IsSameShaped(a, b)),                                             \
                  "Input tensors should have the same data type!");                  \
    if (a->dataType == X_INT) {                                                      \
        int * d = (int*)a->data;                                                     \
        int * db = (int*)b->data;                                                    \
        for (int i = 0; i < a->unitNum; i++)                                         \
            db[i] = (int)origFunc((int)d[i], (T)num);                                \
    }                                                                                \
    else if (a->dataType == X_FLOAT) {                                               \
        float * d = (float*)a->data;                                                 \
        float * db = (float*)b->data;                                                \
        for (int i = 0; i < a->unitNum; i++)                                         \
            db[i] = (float)origFunc((float)d[i], (T)num);                            \
    }                                                                                \
    else if (a->dataType == X_DOUBLE) {                                              \
        double * d = (double*)a->data;                                               \
        double * db = (double*)b->data;                                              \
        for (int i = 0; i < a->unitNum; i++)                                         \
            db[i] = (double)origFunc((double)d[i], (T)num);                          \
    }                                                                                \
    else                                                                             \
        ShowNTErrors("TO DO!");                                                      \
}                                                                                    \
template void _funcName<int>(const XTensor*, XTensor*, int);                         \
template void _funcName<float>(const XTensor*, XTensor*, float);                     \
template void _funcName<double>(const XTensor*, XTensor*, double);                   
#endif

#define _SIMPLE_BINARY_FUNCTION_ME(_funcNameMe, _funcName)                           \
template<class T>                                                                    \
void _funcNameMe(XTensor * a, T num)                                                 \
{                                                                                    \
    _funcName(a, a, num);                                                            \
}                                                                                    \
template void _funcNameMe<int>(XTensor*, int);                                       \
template void _funcNameMe<float>(XTensor*, float);                                   \
template void _funcNameMe<double>(XTensor*, double);                                                                                    
                                                                                     
#define SIMPLE_BINARY_FUNCTION_ME(funcNameMe, _funcName)                             \
template<class T>                                                                    \
void funcNameMe(XTensor &a, T num)                                                   \
{                                                                                    \
    _funcName(&a, &a, num);                                                          \
}                                                                                    \
template void funcNameMe<int>(XTensor&, int);                                        \
template void funcNameMe<float>(XTensor&, float);                                    \
template void funcNameMe<double>(XTensor&, double);                                                                                    
                                                                                     
#define SIMPLE_BINARY_FUNCTION(funcName, _funcName, operationId)                     \
template<class T>                                                                    \
XTensor funcName(const XTensor &a, T num)                                            \
{                                                                                    \
    XTensor b(&a);                                                                   \
    b.SetTMPFlag();                                                                  \
    _funcName(&a, &b, num);                                                          \
    if(a.enableGrad){                                                                \
        XLink::MakeLink(&a, NULL, &b, operationId);                                  \
        XLink::AddParamToHead(&b, num);                                              \
    }                                                                                \
    return b;                                                                        \
}                                                                                    \
template XTensor funcName<int>(const XTensor&, int);                                 \
template XTensor funcName<float>(const XTensor&, float);                             \
template XTensor funcName<double>(const XTensor&, double);                                                                                    
                                                                                     
#define SIMPLE_BINARY_FUNCTION_VOID(funcName, _funcName, operationId)                \
template<class T>                                                                    \
void funcName(const XTensor &a, XTensor &b, T num)                                   \
{                                                                                    \
    if (!b.isInit || !IsSameShaped(a, b)) {                                          \
        InitTensorV2(&b, &a);                                                        \
    }                                                                                \
    _funcName(&a, &b, num);                                                          \
    if (a.enableGrad) {                                                              \
        XLink::MakeLink(&a, NULL, &b, operationId);                                  \
        XLink::AddParamToHead(&b, num);                                              \
    }                                                                                \
}                                                                                    \
template void funcName<int>(const XTensor&, XTensor&, int);                          \
template void funcName<float>(const XTensor&, XTensor&, float);                      \
template void funcName<double>(const XTensor&, XTensor&, double);                                                                           

#ifdef USE_CUDA
_SIMPLE_BINARY_FUNCTION(_Descale, _CudaDescale, BinaryDescale)
_SIMPLE_BINARY_FUNCTION(_Mod, _CudaMod, BinaryMod)
_SIMPLE_BINARY_FUNCTION(_Power, _CudaPower, BinaryPower)
_SIMPLE_BINARY_FUNCTION(_Scale, _CudaScale, BinaryScale)
_SIMPLE_BINARY_FUNCTION(_Shift, _CudaShift, BinaryShift)
#else
_SIMPLE_BINARY_FUNCTION(_Descale, BinaryDescale)
_SIMPLE_BINARY_FUNCTION(_Mod, BinaryMod)
_SIMPLE_BINARY_FUNCTION(_Power, BinaryPower)
_SIMPLE_BINARY_FUNCTION(_Scale, BinaryScale)
_SIMPLE_BINARY_FUNCTION(_Shift, BinaryShift)
#endif

_SIMPLE_BINARY_FUNCTION_ME(_DescaleMe, _Descale)
SIMPLE_BINARY_FUNCTION_ME(DescaleMe, _Descale)
SIMPLE_BINARY_FUNCTION(Descale, _Descale, MATH_DESCALE)
SIMPLE_BINARY_FUNCTION_VOID(Descale, _Descale, MATH_DESCALE)

_SIMPLE_BINARY_FUNCTION_ME(_ModMe, _Mod)
SIMPLE_BINARY_FUNCTION_ME(ModMe, _Mod)
SIMPLE_BINARY_FUNCTION(Mod, _Mod, MATH_MOD)
SIMPLE_BINARY_FUNCTION_VOID(Mod, _Mod, MATH_MOD)

_SIMPLE_BINARY_FUNCTION_ME(_PowerMe, _Power)
SIMPLE_BINARY_FUNCTION_ME(PowerMe, _Power)
SIMPLE_BINARY_FUNCTION(Power, _Power, MATH_POWER)
SIMPLE_BINARY_FUNCTION_VOID(Power, _Power, MATH_POWER)

_SIMPLE_BINARY_FUNCTION_ME(_ScaleMe, _Scale)
SIMPLE_BINARY_FUNCTION_ME(ScaleMe, _Scale)
SIMPLE_BINARY_FUNCTION(Scale, _Scale, MATH_SCALE)
SIMPLE_BINARY_FUNCTION_VOID(Scale, _Scale, MATH_SCALE)

_SIMPLE_BINARY_FUNCTION_ME(_ShiftMe, _Shift)
SIMPLE_BINARY_FUNCTION_ME(ShiftMe, _Shift)
SIMPLE_BINARY_FUNCTION(Shift, _Shift, MATH_SHIFT)
SIMPLE_BINARY_FUNCTION_VOID(Shift, _Shift, MATH_SHIFT)

} // namespace nts(NiuTrans.Tensor)
