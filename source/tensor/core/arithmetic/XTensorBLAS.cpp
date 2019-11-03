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

#include "XTensorBLAS.h"
#include "../../XTensor.h"
#include "../../XBLAS.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
Matrix multiplication by BLAS
c = trans(a) * trans(b) * \alpha + c * \beta
>> a - an input matrix (2d tensor)
>> transposedA - indicate whether the matrix a is transposed
>> b - another input matrix (2d tensor)
>> transposedB - indicate whether the matrix b is transposed
>> alpha - scalar
>> beta - scalar
>> c - output matrix (2d tensor)
*/
void _MatrixMULCPU(const XTensor * a, MATRIX_TRANS_TYPE transposedA,
                   const XTensor * b, MATRIX_TRANS_TYPE transposedB,
                   XTensor * c, DTYPE alpha, DTYPE beta)
{
    CheckNTErrors((a && b && c), "Empty input tensors!");
    CheckNTErrors((a->order == 2 && b->order == 2 && c->order == 2),
                  "Input tensors must have a order = 2!");
    CheckNTErrors((a->dataType == DEFAULT_DTYPE), "TODO!");
    CheckNTErrors((b->dataType == DEFAULT_DTYPE), "TODO!");
    CheckNTErrors((c->dataType == DEFAULT_DTYPE), "TODO!");

#if defined(USE_BLAS)
    int an = a->dimSize[0];
    int am = a->dimSize[1];
    int bn = b->dimSize[0];
    int bm = b->dimSize[1];
    int cn = c->dimSize[0];
    int cm = c->dimSize[1];

    if (transposedA == X_NOTRANS && transposedB == X_NOTRANS)
        GEMM(CblasRowMajor, CblasNoTrans, CblasNoTrans, cn, cm, am, alpha, (DTYPE*)a->data, am, (DTYPE*)b->data, bm, beta, (DTYPE*)c->data, cm);
    else if (transposedA == X_TRANS && transposedB == X_NOTRANS)
        GEMM(CblasRowMajor, CblasTrans, CblasNoTrans, cn, cm, an, alpha, (DTYPE*)a->data, am, (DTYPE*)b->data, bm, beta, (DTYPE*)c->data, cm);
    else if (transposedA == X_NOTRANS && transposedB == X_TRANS)
        GEMM(CblasRowMajor, CblasNoTrans, CblasTrans, cn, cm, am, alpha, (DTYPE*)a->data, am, (DTYPE*)b->data, bm, beta, (DTYPE*)c->data, cm);
    else if (transposedA == X_TRANS && transposedB == X_TRANS)
        GEMM(CblasRowMajor, CblasTrans, CblasNoTrans, cn, cm, an, alpha, (DTYPE*)a->data, am, (DTYPE*)b->data, bm, beta, (DTYPE*)c->data, cm);
#else
        ShowNTErrors("TODO!");
#endif
}

} // namespace nts(NiuTrans.Tensor)
