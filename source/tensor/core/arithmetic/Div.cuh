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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-08-01
 */

#ifndef __DIV_CUH__
#define __DIV_CUH__

#include "Div.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* division of two tensors in a element-wise manner c(i) = a(i)/b(i) */
__global__
void KernelDivElementWise(DTYPE * a, DTYPE * b, DTYPE * c, int size);

/* division of two tensors in a element-wise manner c(i) = a(i)/b(i) + \alpha*c(i) */
__global__
void KernelDivElementWiseV2(DTYPE * a, DTYPE * b, DTYPE * c, int size, DTYPE alpha);

/* division of two tensors in a element-wise manner c(i) = a(i)/b(i)+ \alpha*c(i)  */
template<int nonZeroAlpha>__global__
void KernelDivElementWiseTensorDynamic(DTYPE * a, DTYPE * b, DTYPE * c, DTYPE alpha, int stride, int ldSizeA, int ldSizeB, int ldSizeC, int blockNum);

/* element-wise division of two tensors */
void _CudaDiv(const XTensor * a, const XTensor * b, XTensor * c, DTYPE alpha = 0, int leadingDim = 0);

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)

#endif // __DIV_CUH__

