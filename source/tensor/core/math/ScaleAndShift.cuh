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

#ifndef __SCALEANDSHIFT_CUH__
#define __SCALEANDSHIFT_CUH__

#include "ScaleAndShift.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* scale and shift all tensor entires b = a * scale + shift (CUDA Kernel) */
__global__ 
void KernelScaleAndShift(DTYPE * a, DTYPE * b, int size, DTYPE scale, DTYPE shift);

/* scale and shift all tensor entires b = a * scale + shift (CUDA Kernel) with float16 data type */
__global__ 
void KernelScaleAndShift(__half * a, __half * b, int size, __half scale, __half shift);

/* scale and shift all tensor entires b = a * scale + shift (cuda version) */
void _CudaScaleAndShift(const XTensor * a, XTensor * b, DTYPE scale, DTYPE shift);

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)

#endif // __SCALEANDSHIFT_CUH__