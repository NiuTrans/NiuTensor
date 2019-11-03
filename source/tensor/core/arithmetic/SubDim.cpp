/* NiuTrans.Tensor - an open-source tensor library
* Copyright (C) 2018, Natural Language Processing Lab, Northestern University.
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
* $Created by: Lin Ye (email: linye2015@outlook.com) 2018-08-13
*/

#include <math.h>
#include "Sub.h"
#include "SubDim.h"
#include "SubDim.cuh"
#include "../../XName.h"
#include "../../XUtility.h"
#include "../movement/CopyValues.h"
#include "../shape/IsSameShaped.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
tensor subtraction

c = a - b * \beta
where the size of b is equal to the n-th dimension of a,
i.e., a is subtracted with b by broadcasting

>> a - a tensor
>> b - another tensor whose size is equal to that of dimension n of a
>> c - where we put a-b*\beta. we save it in a if c is NULL
>> n - the dimension index
>> beta - the scaling factor
*/
void _SubDim(const XTensor * a, const XTensor * b, XTensor * c, int n, DTYPE beta)
{
    n = MODX(n, a->order);

    CheckNTErrors(a && b && c, "Empty tensor input!");
    CheckNTErrors(a->unitNum == c->unitNum, "Unmatched tensors in subtraction!");
    CheckNTErrors(a->dataType == b->dataType && a->dataType == c->dataType,
                  "Unmatched data types in subtraction!");
    CheckNTErrors(a->order == c->order, "The input tensors do not have the same order in subtraction!");
    CheckNTErrors(!a->isSparse && !b->isSparse && !c->isSparse, "Dense tensors are required!");
    CheckNTErrors(a->dimSize[n] == b->unitNum, "Wrong tensor size!");

    CheckDev(a->devID, b->devID);

    if (beta == 0) {
        _CopyValues(a, c);
        return;
    }

    if (_IsSameShaped(a, b)) {
        _Sub(a, b, c, beta);
        return;
    }

    if (a->devID >= 0 || b->devID >= 0 || c->devID >= 0) {
#ifdef USE_CUDA
        _CudaSubDim(a, b, c, n, beta);
#else
        ShowNTErrors("Please specify USE_CUDA and recompile the code!");
#endif
    }
    else {
        int stride = 1;
        int blockSize = a->dimSize[n];
        int blockNum = 1;

        for (int i = a->order - 1; i >= 0; i--) {
            if (i > n)
                stride *= a->dimSize[i];
            else if (i < n)
                blockNum *= a->dimSize[i];
        }

        if (a->dataType == DEFAULT_DTYPE) {
            int num = a->unitNum;
            if (stride > 1) {
                for (int i = 0, j = 0; i < num; i += stride, j++) {
                    DTYPE * ap = (DTYPE*)a->data + i;
                    DTYPE   bv = *((DTYPE*)b->data + j % blockSize) * beta;
                    DTYPE * cp = (DTYPE*)c->data + i;
                    for (int k = 0; k < stride; k++)
                        cp[k] = ap[k] - bv;
                }
            }
            else if (stride == 1) {
                DTYPE * bp = (DTYPE*)b->data;
                for (int i = 0; i < num; i += blockSize) {
                    DTYPE * ap = (DTYPE*)a->data + i;
                    DTYPE * cp = (DTYPE*)c->data + i;
                    if (beta == 1.0F) {
                        for (int j = 0; j < blockSize; j++)
                            cp[j] = ap[j] - bp[j];
                    }
                    else {
                        for (int j = 0; j < blockSize; j++)
                            cp[j] = ap[j] - bp[j] * beta;
                    }
                }
            }
            else {
                ShowNTErrors("Something is wrong!");
            }
        }
        else {
            ShowNTErrors("TODO!");
        }
    }
}

/*
tensor subtraction (do it on site)
keep the result in the input tensor and return nothing

c = a - b * \beta
where the size of b is equal to the n-th dimension of a,
i.e., a is subtracted with b by broadcasting

>> a - a tensor
>> b - another tensor whose size is equal to that of dimension n of a
>> n - the dimension index
>> beta - the scaling factor
*/
void _SubDim(XTensor * a, const XTensor * b, int n, DTYPE beta)
{
    _SubDim(a, b, a, n, beta);
}

/*
tensor subtraction (return an XTensor structure and make tensor connections)
make a new tensor to keep the result and return it

c = a - b * \beta
where the size of b is equal to the n-th dimension of a,
i.e., a is subtracted with b by broadcasting

>> a - a tensor
>> b - another tensor whose size is equal to that of dimension n of a
>> n - the dimension index
>> beta - the scaling factor
<< return - the result tensor by tensor subtraction
*/
XTensor SubDim(const XTensor &a, const XTensor &b, int n, DTYPE beta)
{
    XTensor c(&a);
    c.SetTMPFlag();

    n = MODX(n, a.order);

    /* call _Sub function */
    _SubDim(&a, &b, &c, n, beta);

    /* tensor connections */
    if (a.enableGrad && b.enableGrad) {
        XLink::MakeLink(&a, &b, &c, MATH_SUBDIM);
        XLink::AddParamToHeadInt(&c, n);
        XLink::AddParamToHead(&c, beta);
    }

    return c;
}

/*
tensor subtraction

c = a - b * \beta
where the size of b is equal to the n-th dimension of a,
i.e., a is subtracted with b by broadcasting

>> a - a tensor
>> b - another tensor whose size is equal to that of dimension n of a
>> c - where we put a-b*\beta. we save it in a if c is NULL
>> n - the dimension index
>> beta - the scaling factor
*/
void SubDim(const XTensor &a, const XTensor &b, XTensor &c, int n, DTYPE beta)
{
    if (!c.isInit || !IsSameShaped(a, c)) {
        InitTensorV2(&c, &a);
    }

    /* call _Sub function */
    _SubDim(&a, &b, &c, n, beta);

    if (a.enableGrad && b.enableGrad) {
        /* tensor connections */
        XLink::MakeLink(&a, &b, &c, MATH_SUBDIM);
        XLink::AddParamToHeadInt(&c, n);
        XLink::AddParamToHead(&c, beta);
    }
}

}
