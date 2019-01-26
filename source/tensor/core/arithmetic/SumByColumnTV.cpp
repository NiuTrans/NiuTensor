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
#include "SumByColumnTV.h"
#include "SumByColumnTV.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
sum of a tensor and a vector (column vector) in a column by column manner

for each column a_col (in a block), we have
c_col = a_col + b * \beta
where b is a vector.

>> a - a tensor
>> b - a vector with the same column size with a
>> c - where we put a+b. we save it in a if c is NULL
>> beta - the scaling factor
*/
void _SumByColumnTV(const XTensor * a, const XTensor * b, XTensor * c, DTYPE beta)
{
    CheckNTErrors((a && b && c), "Empty input tensors!");
    CheckNTErrors((XTensor::IsSameShaped(a, c)), "Unmatched tensors in addition!");
    CheckNTErrors((b->order == 2 && b->dimSizeRDI[0] == 1 && b->dimSizeRDI[1] == a->dimSizeRDI[1]),
                  "Illegal input vector size!");

    int rowNum = a->dimSize[0];
    int colNum = a->dimSize[1];
    int blockNum = 1;
    for (int i = 2; i < a->order; i++)
        blockNum *= a->dimSizeRDI[i];
    int blockSize = colNum * rowNum;

    if (a->devID >= 0 || b->devID >= 0 || c->devID >= 0) {
#ifdef USE_CUDA
        _CudaSumByColumnTV(a, b, c, beta);
#endif
    }
    else {
        if (!a->isSparse && !b->isSparse) {
            CheckNTErrors(!c->isSparse, "TODO!");

            if (a->dataType == DEFAULT_DTYPE &&
                b->dataType == DEFAULT_DTYPE &&
                c->dataType == DEFAULT_DTYPE)
            {
                for (int k = 0; k < blockNum; k++) {
                    for (int i = 0; i < rowNum; i++) {
                        DTYPE * ap = (DTYPE*)a->data + k * blockSize + i * colNum;
                        DTYPE * bp = (DTYPE*)b->data;
                        DTYPE * cp = (DTYPE*)c->data + k * blockSize + i * colNum;
                        DTYPE v = bp[i];
                        for (int j = 0; j < colNum; j++)
                            cp[j] = ap[j] + v * beta;
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
}

} // namespace nts(NiuTrans.Tensor)