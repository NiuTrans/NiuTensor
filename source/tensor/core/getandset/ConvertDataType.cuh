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

#ifndef __CONVERTDATATYPE_CUH__
#define __CONVERTDATATYPE_CUH__

#include "ConvertDataType.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* convert data type from X_FLOAT to X_FLOAT16 (CUDA Kernel) */
__global__
void KernelFloatToFloat16(float * s, __half * t, int size);

/* convert data type from X_FLOAT16 to X_FLOAT (CUDA Kernel) */
__global__
void KernelFloat16ToFloat(__half * s, float * t, int size);

/* convert data type from X_FLOAT to X_INT (CUDA Kernel) */
__global__
void KernelFloatToInt(float * inputData, int * outputData, int size);

/* convert data type from X_INT to X_FLOAT (CUDA Kernel) */
__global__
void KernelIntToFloat(int * inputData, float * outputData, int size);

/* convert data type */
void _CudaConvertDataType(const XTensor * input, XTensor * output);

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)

#endif // __CONVERTDATATYPE_H__