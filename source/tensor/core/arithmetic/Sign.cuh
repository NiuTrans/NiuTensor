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
* $Created by: LI Yinqiao (li.yin.qiao.2012@hotmail.com) 2018-7-11
*/

#ifndef __SIGN_CUH__
#define __SIGN_CUH__

#include "Sign.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* set each entry to its sign value (CUDA Kernel) */
__global__
void KernelSign(DTYPE * a, DTYPE * b, int size);

/* set each entry to its sign value (CUDA Kernel) with float16 data type*/
__global__
void KernelSign(__half * a, __half * b, int size);

/* set each entry to its sign value */
void _CudaSign(const XTensor * a, XTensor * b);

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)

#endif // __SIGN_H__