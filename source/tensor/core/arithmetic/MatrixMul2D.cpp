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

#include "../../XTensor.h"
#include "../../XName.h"
#include "MatrixMul2D.h"
#include "MatrixMul2D.cuh"
#include "MatrixMul2DParallel.h"
#include "XTensorBLAS.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
matrix multiplication (for 2d tensors)

c = trans(a) * trans(b) * alpha + c * beta
where trans() return the transposed matrix if the flag is fired

>> a - tensor a
>> transposedA - indicates whether the matrices in a are transposed
>> b - tensor b
>> transposedB - indicates whether teh matrices in b are transposed
>> c - where we put a*b
>> alpha - a coefficient
>> beta - another coefficient
>> parallelRunner - parallel processing module
>> stream - the string for creating the job pipeline
*/
void _MatrixMul2D(const XTensor * a, MATRIX_TRANS_TYPE transposedA,
                  const XTensor * b, MATRIX_TRANS_TYPE transposedB,
                  XTensor * c, DTYPE alpha, DTYPE beta,
                  XPRunner * parallelRunner, XStream * stream)
{
    CheckNTErrors((a && b && c), "Empty input tensors!");
    CheckNTErrors((a->dataType == b->dataType), "Input tensors should                have the same data type!");
    CheckNTErrors((a->order == 2 && b->order == 2 && c->order == 2),
                  "Input tensors must have a order = 2!");

    int an = a->dimSize[0], am = a->dimSize[1];
    int bn = b->dimSize[0], bm = b->dimSize[1];
    int cn = c->dimSize[0], cm = c->dimSize[1];
    int am2 = transposedA == X_TRANS ? an : am;
    int an2 = transposedA == X_TRANS ? am : an;
    int bm2 = transposedB == X_TRANS ? bn : bm;
    int bn2 = transposedB == X_TRANS ? bm : bn;
    int cm2 = cm;
    int cn2 = cn;

    CheckNTErrors((am2 == bn2 && an2 == cn2 && bm2 == cm2),
                  "Unmatched tensors in multiplication!");

#ifdef USE_CUDA
    if (a->devID >= 0 || b->devID >= 0 || c->devID >= 0) {
        _CudaMatrixMul2D(a, transposedA, b, transposedB, c, alpha, beta, stream);
        return;
    }
#endif

    /* a dense matrix multiply a dense matrix */
    if (!a->isSparse && !b->isSparse) {
        CheckNTErrors(!c->isSparse, "Illegal use of sparse matrix in multiplication!");

        if (a->dataType == DEFAULT_DTYPE &&
            b->dataType == DEFAULT_DTYPE &&
            c->dataType == DEFAULT_DTYPE)
        {
#if defined(USE_BLAS)
                _MatrixMULCPU(a, transposedA, b, transposedB, c, alpha, beta);
#else
                _MatrixMul2DParallel(a, transposedA, b, transposedB, c, alpha, beta, parallelRunner);
#endif
        }
        else {
            // TODO!!
            ShowNTErrors("TODO!");
        }
    }
    /* a dense matrix multiply a sparse matrix */
    else if (!a->isSparse && b->isSparse) {
        CheckNTErrors(!c->isSparse, "Illegal use of sparse matrix in multiplication!");

        if (a->dataType == DEFAULT_DTYPE &&
            b->dataType == DEFAULT_DTYPE &&
            c->dataType == DEFAULT_DTYPE)
        {
            CheckNTErrors((beta == 0 || beta == 1.0), "beta must be 0 or 1.");

            if (beta == 0)
                c->SetZeroAll();
            int num = *((int*)b->data);
            char * p = (char*)b->data + sizeof(int); // pointer to the first tuple

            /* a * b */
            if (transposedA == X_NOTRANS && transposedB == X_NOTRANS) {
                for (int i = 0; i < num; i++) {
                    int key = *((int*)p);
                    int ni = key / bm;
                    int mi = key % bm;

                    for (int k = 0; k < an; k++) {
                        DTYPE dot = a->Get2D(k, ni) * b->Get2D(ni, mi) * alpha;
                        c->Add2D(dot, k, mi);
                    }

                    p = p + sizeof(int) + sizeof(DTYPE);
                }
            }
            /* trans(a) * b */
            else if (transposedA == X_TRANS && transposedB == X_NOTRANS) {
                for (int i = 0; i < num; i++) {
                    int key = *((int*)p);
                    int ni = key / bm;
                    int mi = key % bm;

                    for (int k = 0; k < an; k++) {
                        DTYPE dot = a->Get2D(ni, k) * b->Get2D(ni, mi) * alpha;
                        c->Add2D(dot, k, mi);
                    }

                    p = p + sizeof(int) + sizeof(DTYPE);
                }
            }
            /* a * trans(b) */
            else if (transposedA == X_NOTRANS && transposedB == X_TRANS) {
                for (int i = 0; i < num; i++) {
                    int key = *((int*)p);
                    int mi = key / bm;
                    int ni = key % bm;

                    for (int k = 0; k < an; k++) {
                        DTYPE dot = a->Get2D(k, ni) * b->Get2D(mi, ni) * alpha;
                        c->Add2D(dot, k, mi);
                    }

                    p = p + sizeof(int) + sizeof(DTYPE);
                }
            }
            /* trans(a) * trans(b) */
            else if (transposedA == X_TRANS && transposedB == X_TRANS) {
                for (int i = 0; i < num; i++) {
                    int key = *((int*)p);
                    int mi = key / bm;
                    int ni = key % bm;

                    for (int k = 0; k < an; k++) {
                        DTYPE dot = a->Get2D(ni, k) * b->Get2D(mi, ni) * alpha;
                        c->Add2D(dot, k, mi);
                    }

                    p = p + sizeof(int) + sizeof(DTYPE);
                }
            }
        }
        else {
            // TODO!!
            ShowNTErrors("TODO!");
        }

    }
    else {
        // TODO!!
        ShowNTErrors("TODO!");
    }
}

} // namespace nts(NiuTrans.Tensor)
