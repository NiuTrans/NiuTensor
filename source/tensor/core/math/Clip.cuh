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
* $Created by: Lin Ye (email: linye2015@outlook.com) 2018-08-03
*/

#ifndef __CLIP_CUH__
#define __CLIP_CUH__

#include "Clip.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* set each entry to its clip value (CUDA Kernel) */
__global__
void KernelClip(DTYPE * a, DTYPE * b, DTYPE lower, DTYPE upper, int size);

/* set each entry to its clip value (CUDA Kernel) with float16 data type*/
__global__
void KernelClip(__half * a, __half * b, DTYPE lower, DTYPE upper, int size);

/* set each entry to its clip value */
void _CudaClip(const XTensor * a, XTensor * b, DTYPE lower, DTYPE upper);

/* backward of Clip function (CUDA Kernel) */
__global__
void KernelClipBackward(DTYPE * dedy, DTYPE * dedx, DTYPE * y, DTYPE * x, DTYPE lower, DTYPE upper, int size);

/* backward of Clip function */
void _CudaClipBackward(XTensor * y, XTensor * x, XTensor * dedy, XTensor * dedx, DTYPE lower, DTYPE upper);

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)

#endif // __CLIP_H__