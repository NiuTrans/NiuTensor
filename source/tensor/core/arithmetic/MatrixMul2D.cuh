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

#ifndef __MATRIXMUL2D_CUH__
#define __MATRIXMUL2D_CUH__

#include "MatrixMul2D.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/*
mutilication of a dense matrix with a sparse vector
c = a * b * \alpha
*/
__global__
void KernelMatrixMulDenseMSparseMV2(DTYPE * a, MATRIX_TRANS_TYPE transposedA, int aColSize, int aRowSize,
                                    void * b, MATRIX_TRANS_TYPE transposedB, int bNonZeroNum, int bColSize, int bRowSize,
                                    DTYPE * c, int cColSize, int cRowSize, DTYPE alpha);

/*
matrix multiplication (for 2d tensors) (cuda version)
c = trans(a) * trans(b) * alpha + c * beta
where trans() return the transposed matrix if the flag is fired
*/
void _CudaMatrixMul2D(const XTensor * a, MATRIX_TRANS_TYPE transposedA, const XTensor * b, MATRIX_TRANS_TYPE transposedB, XTensor * c,
                      DTYPE alpha = (DTYPE)1.0, DTYPE beta = 0, XStream * stream = NULL);

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)

#endif // __MATRIXMUL2D_CUH__

