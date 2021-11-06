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
* $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-04-24
*/

#ifndef __NORMALIZE_CUH__
#define __NORMALIZE_CUH__

#include "Normalize.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* 
normalized the data with normal distribution (Kernel code). For an input x,
y = a * (x-mean)/sqrt(variance+\epsilon) + b
where a and b are the scalar and bias respectively, and \epsilon is the adjustment parameter
*/
template<class T> __global__
void KernelNormalizeFloat(T * input, T* output, T* mean, T* var,
                          T* a, T* b, T epsilon,
                          int stride, int strideNum, int blockNum);

template<class T> __global__
void KernelNormalizeHalf(T* input, T* output, T* mean, T* var,
                         T* a, T* b, int stride, int strideNum, int blockNum);

/* 
normalized the data with normal distribution. For an input x,
y = a * (x-mean)/sqrt(variance+\epsilon) + b
where a and b are the scalar and bias respectively, and \epsilon is the adjustment parameter
*/
void _CudaNormalize(const XTensor * input, XTensor * output, int dim,
                    const XTensor * mean, const XTensor * var,
                    const XTensor * a, const XTensor * b, DTYPE epsilon);

/* 
L1-normalized the data with normal distribution (Kernel code). For an input x,
y = a * (x-mean)/distance + b
where a and b are the scalar and bias respectively, and \epsilon is the adjustment parameter
*/
template<class T> __global__
void KernelL1NormalizeFloat(T * input, T* output, T* mean, T* distance,
                            T* a, T* b, int stride, int strideNum, int blockNum);

template<class T> __global__
void KernelL1NormalizeHalf(T* input, T* output, T* mean, T* distance,
                           T* a, T* b, int stride, int strideNum, int blockNum);

/* 
L1-normalized the data with normal distribution. For an input x,
y = a * (x-mean)/distance + b
where a and b are the scalar and bias respectively, and \epsilon is the adjustment parameter
*/
void _CudaL1Normalize(const XTensor * input, XTensor * output, int dim,
                      const XTensor * mean, const XTensor * distance,
                      const XTensor * a, const XTensor * b);

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)

#endif // __NORMALIZE_CUH__

