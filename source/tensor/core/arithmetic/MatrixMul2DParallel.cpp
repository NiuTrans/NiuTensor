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
#include "MatrixMul2DParallel.h"
#include "MatrixMul2DMultiTheading.h"
#include "../utilities/XMatrixSegment.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
matrix multiplication (for 2d tensors) with multi-threading
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
*/
void _MatrixMul2DParallel(const XTensor * a, MATRIX_TRANS_TYPE transposedA,
                          const XTensor * b, MATRIX_TRANS_TYPE transposedB,
                          XTensor * c, DTYPE alpha, DTYPE beta, XPRunner * parallelRunner)
{
    CheckNTErrors((a && b && c), "Empty input tensors!");
    CheckNTErrors((a->order == 2 && b->order == 2 && c->order == 2),
        "Input tensors must have a order = 2!");

    int an = a->dimSize[0], am = a->dimSize[1];
    int bm = b->dimSize[1];
    int cn = c->dimSize[0], cm = c->dimSize[1];
    int aColNum = am;
    int bColNum = bm;

    /* a * b */
    if (transposedA == X_NOTRANS && transposedB == X_NOTRANS) {
        RunParallel2D(parallelRunner, (void*)_MatrixMul2DMultiTheading, an * am * bm,
                      cn, cm, 5,
                      a, b, c, &alpha, &beta);
    }
    /* trans(a) * b */
    else if (transposedA == X_TRANS && transposedB == X_NOTRANS) {
        int num = an;
        for (int i = 0; i < cn; i++) {
            DTYPE * p3 = (DTYPE*)c->data + i * cm;
            for (int j = 0; j < cm; j++) {
                DTYPE r = 0;
                DTYPE * p1 = (DTYPE*)a->data + 0 * am + i;
                DTYPE * p2 = (DTYPE*)b->data + 0 * bm + j;

                for (int k = 0; k < num; k++) {
                    r += (*p1) * (*p2) * alpha;
                    p1 += aColNum;
                    p2 += bColNum;
                }

                *p3 = *p3 * beta + r;
                p3 += 1;
            }
        }
    }
    /* a * trans(b) */
    else if (transposedA == X_NOTRANS && transposedB == X_TRANS) {
        int num = am;
        for (int i = 0; i < cn; i++) {
            DTYPE * p3 = (DTYPE*)c->data + i * cm;
            for (int j = 0; j < cm; j++) {
                DTYPE r = 0;
                DTYPE * p1 = (DTYPE*)a->data + i * am + 0;
                DTYPE * p2 = (DTYPE*)b->data + j * bm + 0;

                for (int k = 0; k < num; k++) {
                    r += (*p1) * (*p2) * alpha;
                    p1 += 1;
                    p2 += 1;
                }

                *p3 = *p3 * beta + r;
                p3 += 1;
            }
        }
    }
    /* trans(a) * trans(b) */
    else if (transposedA == X_TRANS && transposedB == X_TRANS) {
        int num = an;
        for (int i = 0; i < cn; i++) {
            DTYPE * p3 = (DTYPE*)c->data + i * cm;
            for (int j = 0; j < cm; j++) {
                DTYPE r = 0;
                DTYPE * p1 = (DTYPE*)a->data + 0 * am + i;
                DTYPE * p2 = (DTYPE*)b->data + j * bm + 0;

                for (int k = 0; k < num; k++) {
                    r += (*p1) * (*p2) * alpha;
                    p1 += aColNum;
                    p2 += 1;
                }

                *p3 = *p3 * beta + r;
                p3 += 1;
            }
        }
    }
}

} // namespace nts(NiuTrans.Tensor)
