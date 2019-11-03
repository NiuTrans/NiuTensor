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
#include "MatrixMul2DMultiTheading.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
matrix multiplication for a block (x1,y1) - (x2,y2)
where (x1,y1) is the upper-left corner and (x2,y2) is the bottom-right corner
NOTE: this is a instance of the TFunction type and would be used in XThread
(see more information in XThread.h/cpp)
>> args - arguments
argument0: x1 - row index (upper-left corner)
argument1: y1 - column index (upper-left corner)
argument3: x2 - row index (bottom-right corner)
argument4: y2 - column index (bottom-right corner)
argument5: matrix a
argument6: matrix b
argument7: matrix c (c=a*b*\alpha + c*beta)
*/
void _MatrixMul2DMultiTheading(TensorList * args)
{
    CheckNTErrors(args->count == 2, "invalid argument number!");
    IntList * indexArgs = (IntList*)args->GetItem(0);
    TensorList * matrixArgs = (TensorList*)args->GetItem(1);
    CheckNTErrors(indexArgs->count == 4, "invalid argument number!");
    CheckNTErrors(matrixArgs->count == 5, "invalid argument number!");

    XTensor * a = matrixArgs->GetItem(0);
    XTensor * b = matrixArgs->GetItem(1);
    XTensor * c = matrixArgs->GetItem(2);
    DTYPE alpha = *(DTYPE*)(matrixArgs->GetItem(3));
    DTYPE beta = *(DTYPE*)(matrixArgs->GetItem(4));
    int x1 = indexArgs->GetItem(0);
    int y1 = indexArgs->GetItem(1);
    int x2 = indexArgs->GetItem(2);
    int y2 = indexArgs->GetItem(3);

#ifdef FAST_MATRIX
    int am = a->dimSize[1];
    int bm = b->dimSize[1];
    int cm = c->dimSize[1];

    int num = am;
    int bColNum = bm;
    if (beta == 0) {
        if (alpha == 1) {
            for (int i = x1; i <= x2; i++) {
                DTYPE * p3 = (DTYPE*)c->data + i * cm + y1;
                for (int j = y1; j <= y2; j++) {
                    DTYPE r = 0;
                    DTYPE * p1 = (DTYPE*)a->data + i * am + 0;
                    DTYPE * p2 = (DTYPE*)b->data + 0 * bm + j;

                    for (int k = 0; k < num; k++) {
                        r += (*p1) * (*p2);
                        p1 += 1;
                        p2 += bColNum;
                    }

                    *p3 = r;
                    p3 += 1;
                }
            }
        }
        else {
            for (int i = x1; i <= x2; i++) {
                DTYPE * p3 = (DTYPE*)c->data + i * cm + y1;
                for (int j = y1; j <= y2; j++) {
                    DTYPE r = 0;
                    DTYPE * p1 = (DTYPE*)a->data + i * am + 0;
                    DTYPE * p2 = (DTYPE*)b->data + 0 * bm + j;

                    for (int k = 0; k < num; k++) {
                        r += (*p1) * (*p2) * alpha;
                        p1 += 1;
                        p2 += bColNum;
                    }

                    *p3 = r;
                    p3 += 1;
                }
            }
        }
    }
    else {
        if (alpha == 1) {
            for (int i = x1; i <= x2; i++) {
                DTYPE * p3 = (DTYPE*)c->data + i * cm + y1;
                for (int j = y1; j <= y2; j++) {
                    DTYPE r = 0;
                    DTYPE * p1 = (DTYPE*)a->data + i * am + 0;
                    DTYPE * p2 = (DTYPE*)b->data + 0 * bm + j;

                    for (int k = 0; k < num; k++) {
                        r += (*p1) * (*p2);
                        p1 += 1;
                        p2 += bColNum;
                    }

                    *p3 = *p3 * beta + r;
                    p3 += 1;
                }
            }
        }
        else {
            for (int i = x1; i <= x2; i++) {
                DTYPE * p3 = (DTYPE*)c->data + i * cm + y1;
                for (int j = y1; j <= y2; j++) {
                    DTYPE r = 0;
                    DTYPE * p1 = (DTYPE*)a->data + i * am + 0;
                    DTYPE * p2 = (DTYPE*)b->data + 0 * bm + j;

                    for (int k = 0; k < num; k++) {
                        r += (*p1) * (*p2) * alpha;
                        p1 += 1;
                        p2 += bColNum;
                    }

                    *p3 = *p3 * beta + r;
                    p3 += 1;
                }
            }
        }
    }
#else
    int num = am;
    for (int i = x1; i <= x2; i++) {
        for (int j = y1; j <= y2; j++) {
            DTYPE r = 0;
            for (int k = 0; k < num; k++) {
                r += a->Get2D(i, k) * b->Get2D(k, j);
            }
            c->Set2D(r, i, j);
        }
    }
#endif
}

} // namespace nts(NiuTrans.Tensor)
