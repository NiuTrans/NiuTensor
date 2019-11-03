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

#ifndef __MATRIXMUL_H__
#define __MATRIXMUL_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#define MMul MatrixMul

/*
matrix multiplication c = trans(a) * trans(b) * alpha + c * beta

For the input tensors a and b, we perform matrix multiplicationon the first two dimentsions. 
E.g., let A be a tensor of size y * z * m and B bea tensor of size x * y * n. 
For A * B, we go over each order-2 tensor of A (of size x * y) and each order-2 tensor B (of size z * x), 
like this c_{i,j} = trans(ai) * trans(bj) * alpha + c_{i,j} * beta
where trans() returns the transposed matrix if the flag is fired, ai is the i-th element tensor of A,
bj is the j-th element tensor of B, and c_{i,j} is the (i,j) elementtensor of the result C. 
C should be a tensor of z * x * n * m. 
Obviously C = A * B performs normal matrix multiplication if A = y * z and B = x * y.
*/
void _MatrixMul(const XTensor * a, MATRIX_TRANS_TYPE transposedA, 
                const XTensor * b, MATRIX_TRANS_TYPE transposedB, 
                XTensor * c,
                DTYPE alpha = (DTYPE)1.0, DTYPE beta = 0, 
                XPRunner * parallelRunner = NULL);

/* 
matrix multiplication (return an XTensor structure) c = trans(a) * trans(b) * alpha
make a new tensor c to keep the result and return it

For the input tensors a and b, we perform matrix multiplicationon the first two dimentsions. 
E.g., let A be a tensor of size y * z * m and B bea tensor of size x * y * n. 
For A * B, we go over each order-2 tensor of A (of size x * y) and each order-2 tensor B (of size z * x), 
like this c_{i,j} = trans(ai) * trans(bj) * alpha + c_{i,j} * beta
where trans() returns the transposed matrix if the flag is fired, ai is the i-th element tensor of A,
bj is the j-th element tensor of B, and c_{i,j} is the (i,j) elementtensor of the result C. 
C should be a tensor of z * x * n * m. 
Obviously C = A * B performs normal matrix multiplication if A = y * z and B = x * y.
*/
XTensor MatrixMul(const XTensor &a, MATRIX_TRANS_TYPE transposedA, 
                  const XTensor &b, MATRIX_TRANS_TYPE transposedB, 
                  DTYPE alpha = (DTYPE)1.0, 
                  XPRunner * parallelRunner = NULL);

void MatrixMul(const XTensor &a, MATRIX_TRANS_TYPE transposedA, 
               const XTensor &b, MATRIX_TRANS_TYPE transposedB,
               XTensor &c, 
               DTYPE alpha = (DTYPE)1.0, DTYPE beta = 0, 
               XPRunner * parallelRunner = NULL);

/* matrix multiplication with no transposition c = a * b * alpha*/
XTensor MatrixMul(const XTensor &a, const XTensor &b, 
                  DTYPE alpha = (DTYPE)1.0, XPRunner * parallelRunner = NULL);

void MatrixMul(const XTensor &a, const XTensor &b, XTensor &c, 
               DTYPE alpha = (DTYPE)1.0, XPRunner * parallelRunner = NULL);

} // namespace nts(NiuTrans.Tensor)

#endif // __MATRIXMUL_H__