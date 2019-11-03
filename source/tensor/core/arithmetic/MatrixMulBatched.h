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

#ifndef __MATRIXMULBATCHED_H__
#define __MATRIXMULBATCHED_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#define BMMul MatrixMulBatched

/*
matrix multiplication of the two tensors c = trans(a) * trans(b) * alpha + c * beta

for each 2-dimensional data array in a (denoted as ai) and
each 2-dimensional data array in b (denoted as bi), we have
ci = trans(ai) * trans(bi) * alpha + cm * beta
where trans() returns the transposed matrix if the flag is fired
*/
void _MatrixMulBatched(const XTensor * a, MATRIX_TRANS_TYPE transposedA, const XTensor * b, MATRIX_TRANS_TYPE transposedB,
                       XTensor * c, DTYPE alpha = (DTYPE)1.0, DTYPE beta = 0, XPRunner * parallelRunner = NULL);


/*
matrix multiplication of the two tensors c = trans(a) * trans(b) * alpha + c * beta
optimized for GPU
*/
void _MatrixMulBatchedGPU(const XTensor * a, MATRIX_TRANS_TYPE transposedA, const XTensor * b, MATRIX_TRANS_TYPE transposedB,
                          XTensor * c, DTYPE alpha = (DTYPE)1.0, DTYPE beta = 0);

/*
matrix multiplication of the two tensors c = trans(a) * trans(b) * alpha + c * beta
optimized for GPU
*/
void _MatrixMulBatchedCPU(const XTensor * a, MATRIX_TRANS_TYPE transposedA, const XTensor * b, MATRIX_TRANS_TYPE transposedB, 
                          XTensor * c, DTYPE alpha = (DTYPE)1.0, DTYPE beta = 0);

/*
matrix multiplication of the two tensors c = trans(a) * trans(b) * alpha + c * beta (for list inputs)
optimized for GPU
*/
void _MatrixMulBatchedCPU(const TensorList * a, MATRIX_TRANS_TYPE transposedA, const TensorList * b, MATRIX_TRANS_TYPE transposedB, 
                          TensorList * c, DTYPE alpha = (DTYPE)1.0, DTYPE beta = 0);

/*
matrix multiplication of the two tensors (return an XTensor structure) c = trans(a) * trans(b) * alpha
make a new tensor to keep the result and return it

for each 2-dimensional data array in a (denoted as ai) and
each 2-dimensional data array in b (denoted as bi), we have
ci = trans(ai) * trans(bi) * alpha + cm * beta
where trans() returns the transposed matrix if the flag is fired
*/
XTensor MatrixMulBatched(const XTensor &a, MATRIX_TRANS_TYPE transposedA, const XTensor &b, MATRIX_TRANS_TYPE transposedB,
                         DTYPE alpha = (DTYPE)1.0, XPRunner * parallelRunner = NULL);

/*
matrix multiplication of the two tensors (return an XTensor structure) c = a * b * alpha
make a new tensor to keep the result and return it

for each 2-dimensional data array in a (denoted as ai) and
each 2-dimensional data array in b (denoted as bi), we have
ci = ai * bi * alpha + cm * beta
*/
XTensor MatrixMulBatched(const XTensor &a, const XTensor &b, 
                         DTYPE alpha = (DTYPE)1.0, XPRunner * parallelRunner = NULL);

} // namespace nts(NiuTrans.Tensor)

#endif // __MATRIXMULBATCHED_H__