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

#ifndef __SUB_CUH__
#define __SUB_CUH__

#include "Sub.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* subtraction of data arrays (CUDA Kernel) */
__global__
void KernelSUB(DTYPE * a, DTYPE * b, DTYPE * c, int size, DTYPE beta = (DTYPE)1.0);

/* tensor subtraction c = a - b * \beta (cuda version) */
void _CudaSub(const XTensor * a, const XTensor * b, XTensor * c = NULL, DTYPE beta = (DTYPE)1.0);

/*  tensor subtraction c = a - b * \beta (cuda version) with an input handle */
void _CudaSubWithHandle(int devID, cublasHandle_t * handle, DTYPE * a, DTYPE * b, DTYPE * c, int size, DTYPE beta = (DTYPE)1.0);

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)

#endif // __SUB_CUH__
