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

#ifndef __UNARY_CUH__
#define __UNARY_CUH__

#include "../../XTensor.h"
#include "Unary.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* set each entry to its absolute value (CUDA Kernel) */
__global__
void KernelAbsolute(DTYPE * a, DTYPE * b, int size);
/* set each entry to its absolute value (CUDA Kernel) with float16 data type*/
__global__
void KernelAbsolute(__half * a, __half * b, int size);
/* set each entry to its absolute value */
void _CudaAbsolute(const XTensor * a, XTensor * b);

/* set each entry to its ceil value (CUDA Kernel) */
__global__
void KernelCeil(DTYPE * a, DTYPE * b, int size);
/* set each entry to its ceil value (CUDA Kernel) with float16 data type*/
__global__
void KernelCeil(__half * a, __half * b, int size);
/* set each entry to its ceil value */
void _CudaCeil(const XTensor * a, XTensor * b);

/* set each entry to its exponent value (CUDA Kernel) */
__global__
void KernelExp(DTYPE * a, DTYPE * b, int size);
/* set each entry to its exponent value (CUDA Kernel) with float16 data type*/
__global__
void KernelExp(__half * a, __half * b, int size);
/* set each entry to its exponent value */
void _CudaExp(const XTensor * a, XTensor * b);

/* set each entry to its floor value (CUDA Kernel) */
__global__
void KernelFloor(DTYPE * a, DTYPE * b, int size);
/* set each entry to its floor value (CUDA Kernel) with float16 data type*/
__global__
void KernelFloor(__half * a, __half * b, int size);
/* set each entry to its floor value */
void _CudaFloor(const XTensor * a, XTensor * b);

/* if source entry is non-zero, set target entry to be one, otherwise zero (CUDA Kernel) */
__global__
void KernelIsNonZero(DTYPE * a, DTYPE * b, int size);
/* if source entry is non-zero, set target entry to be one, otherwise zero (CUDA Kernel) with float16 data type*/
__global__
void KernelIsNonZero(__half * a, __half * b, int size);
/* if source entry is non-zero, set target entry to be one, otherwise zero */
void _CudaIsNonZero(const XTensor * a, XTensor * b);

/* if source entry is zero, set target entry to be one, otherwise zero (CUDA Kernel) */
__global__
void KernelIsZero(DTYPE * a, DTYPE * b, int size);
/* if source entry is zero, set target entry to be one, otherwise zero (CUDA Kernel) with float16 data type*/
__global__
void KernelIsZero(__half * a, __half * b, int size);
/* if source entry is zero, set target entry to be one, otherwise zero */
void _CudaIsZero(const XTensor * a, XTensor * b);

/* set each entry to its logarithm value (CUDA Kernel) */
__global__
void KernelLog(DTYPE * a, DTYPE * b, int size);
/* set each entry to its logarithm value (CUDA Kernel) with float16 data type*/
__global__
void KernelLog(__half * a, __half * b, int size);
/* set each entry to its logarithm value */
void _CudaLog(const XTensor * a, XTensor * b);

/* set each entry to its round value (CUDA Kernel) */
__global__
void KernelRound(DTYPE * a, DTYPE * b, int size);
/* set each entry to its round value (CUDA Kernel) with float16 data type*/
__global__
void KernelRound(__half * a, __half * b, int size);
/* set each entry to its round value */
void _CudaRound(const XTensor * a, XTensor * b);

/* set each entry to its sqrt value (CUDA Kernel) */
__global__
void KernelSqrt(DTYPE * a, DTYPE * b, int size);
/* set each entry to its sqrt value (CUDA Kernel) with float16 data type*/
__global__
void KernelSqrt(__half * a, __half * b, int size);
/* set each entry to its sqrt value */
void _CudaSqrt(const XTensor * a, XTensor * b);

/* set each entry to its square value (CUDA Kernel) */
__global__
void KernelSquare(DTYPE * a, DTYPE * b, int size);
/* set each entry to its square value (CUDA Kernel) with float16 data type*/
__global__
void KernelSquare(__half * a, __half * b, int size);
/* set each entry to its square value */
void _CudaSquare(const XTensor * a, XTensor * b);


/* set each entry to its sine value (CUDA Kernel) */
__global__
void KernelSin(DTYPE * a, DTYPE * b, int size);
/* set each entry to its sine value (CUDA Kernel) with float16 data type*/
__global__
void KernelSin(__half * a, __half * b, int size);
/* set each entry to its sine value */
void _CudaSin(const XTensor * a, XTensor * b);

/* set each entry to its cosine value (CUDA Kernel) */
__global__
void KernelCos(DTYPE * a, DTYPE * b, int size);
/* set each entry to its cosine value (CUDA Kernel) with float16 data type*/
__global__
void KernelCos(__half * a, __half * b, int size);
/* set each entry to its cosine value */
void _CudaCos(const XTensor * a, XTensor * b);

/* set each entry to its tangent value (CUDA Kernel) */
__global__
void KernelTan(DTYPE * a, DTYPE * b, int size);
/* set each entry to its tangent value (CUDA Kernel) with float16 data type*/
__global__
void KernelTan(__half * a, __half * b, int size);
/* set each entry to its tangent value */
void _CudaTan(const XTensor * a, XTensor * b);

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)

#endif // __UNARY_CUH__