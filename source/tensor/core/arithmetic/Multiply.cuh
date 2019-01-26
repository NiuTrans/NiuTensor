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

#ifndef __MULTIPLY_CUH__
#define __MULTIPLY_CUH__

#include "Multiply.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* multiplication of two tensors in a element-wise manner c(i) = a(i)*b(i) */
__global__
void KernelMulElementWise(DTYPE * a, DTYPE * b, DTYPE * c, int size);

/* multiplication of two tensors in a element-wise manner c(i) = a(i)*b(i) + \alpha*c(i) */
__global__
void KernelMulElementWiseV2(DTYPE * a, DTYPE * b, DTYPE * c, int size, DTYPE alpha);

/* multiplication of two tensors in a element-wise manner c(i) = a(i)*b(i)+ \alpha*c(i)  */
template<int nonZeroAlpha>__global__
void KernelMulElementWiseTensorDynamic(DTYPE * a, DTYPE * b, DTYPE * c, DTYPE alpha, int stride, int ldSizeA, int ldSizeB, int ldSizeC, int blockNum);

/* element-wise product of two tensors */
void _CudaMultiply(const XTensor * a, const XTensor * b, XTensor * c, DTYPE alpha = 0, int leadingDim = 0);

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)

#endif // __MULTIPLY_CUH__

