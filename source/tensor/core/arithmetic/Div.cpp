/* NiuTrans.Tensor - an open-source tensor library
* Copyright (C) 2017, Natural Language Processing Lab, Northeastern University.
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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-08-01
 */

#include "../../XTensor.h"
#include "../../XName.h"
#include "../../XUtility.h"
#include "../shape/IsSameShaped.h"
#include "Sum.h"
#include "../math/ScaleAndShift.h"
#include "Div.h"
#include "Div.cuh"
#include "DivDim.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
element-wise division of two tensors

c(i) = a(i)/b(i) + \alpha * c(i)
where i is the index of the item

>> a - tensor a
>> b - tensor b
>> c - result tensor
>> alpha - the coefficient
>> leadingDim - the dimension along which we perform broadcasting
*/
void _Div(const XTensor * a, const XTensor * b, XTensor * c, DTYPE alpha, int leadingDim)
{
    CheckNTErrors((a->unitNum <= c->unitNum && b->unitNum <= c->unitNum),
                  "Unmatched tensors in multiplication!");
    CheckNTErrors((a->order == b->order && a->order == c->order), 
                  "Unmatched tensors!");

    CheckDev(a->devID, b->devID);
#ifdef USE_CUDA
    if (a->devID >= 0 || b->devID >= 0 || c->devID >= 0) {
        _CudaDiv(a, b, c, alpha, leadingDim);
        return;
    }
#endif

    int stride = 1;
    int blockSizeA = 1;
    int blockSizeB = 1;
    int blockSizeC = 1;
    int blockNum = 1;
    int dimensionSizeA = a->dimSize[leadingDim];
    int dimensionSizeB = b->dimSize[leadingDim];
    int dimensionSizeC = c->dimSize[leadingDim];

    for (int i = 0; i < a->order; i++) {
        if (i != leadingDim) {
            CheckNTErrors((a->dimSize[i] == b->dimSize[i] && a->dimSize[i] == c->dimSize[i]),
                          "Unmatched tensors!");
        }
        if (i > leadingDim)
            stride *= a->dimSize[i];
    }

    blockSizeA = stride * dimensionSizeA;
    blockSizeB = stride * dimensionSizeB;
    blockSizeC = stride * dimensionSizeC;
    blockNum = a->unitNum / blockSizeA;

    if (!a->isSparse && !b->isSparse) {
        if (a->dataType == DEFAULT_DTYPE && b->dataType == DEFAULT_DTYPE) {
            if (a->unitNum == c->unitNum && b->unitNum == c->unitNum) {
                int size = a->unitNum;
                DTYPE * ap = (DTYPE*)a->data;
                DTYPE * bp = (DTYPE*)b->data;
                DTYPE * cp = (DTYPE*)c->data;
                if (alpha == 0) {
                    for (int i = 0; i < size; i++)
                        cp[i] = ap[i] / bp[i];
                }
                else {
                    for (int i = 0; i < size; i++)
                        cp[i] = ap[i] / bp[i] + alpha * cp[i];
                }
            }
            else {
                for (int k = 0; k < blockNum; k++) {

                    for (int ci = 0, ai = 0, bi = 0; ci < dimensionSizeC; ci++, ai++, bi++) {
                        if (ai >= dimensionSizeA)
                            ai = 0;
                        if (bi >= dimensionSizeB)
                            bi = 0;
                        DTYPE * ap = (DTYPE*)a->data + k * blockSizeA + ai * stride;
                        DTYPE * bp = (DTYPE*)b->data + k * blockSizeB + bi * stride;
                        DTYPE * cp = (DTYPE*)c->data + k * blockSizeC + ci * stride;
                        for (int j = 0; j < stride; j++)
                            cp[j] = ap[j] / bp[j] + cp[j] * alpha;
                    }
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

/*
element-wise division of two tensors (do it on site)
keep the result in the input tensor a and return nothing

a(i) = a(i)/b(i) + \alpha * a(i)
where i is the index of the item

>> a - tensor a (where keep the result)
>> b - tensor b
>> alpha - the coefficient
>> leadingDim - the dimension along which we perform broadcasting
*/
void _DivMe(XTensor * a, const XTensor * b, DTYPE alpha, int leadingDim)
{
    _Div(a, b, a, alpha, leadingDim);
}

/*
element-wise division of two tensors (do it on site)
keep the result in the input tensor a and return nothing

a(i) = a(i)/b(i) + \alpha * a(i)
where i is the index of the item

>> a - tensor a (where keep the result)
>> b - tensor b
>> alpha - the coefficient
>> leadingDim - the dimension along which we perform broadcasting
*/
void DivMe(XTensor & a, const XTensor & b, DTYPE alpha, int leadingDim)
{
    if (b.order == 0){
        DTYPE scale = 1.0F / b.Get0D() + alpha;

        _ScaleAndShift(&a, &a, scale, 0.0F);
    }
    else {
        int n = GetBroadcastDimIndex(a, b);

        if (n == -1) {
            CheckNTErrors(a.dimSize[leadingDim] == b.dimSize[leadingDim], "TODO!");

            /* call _Div function */
            _Div(&a, &b, &a, alpha, leadingDim);
        }
        else if (n >= 0 && n < a.order)
            /* call _DivDim function */
            _DivDim(&a, &b, &a, n, alpha);
        else
            ShowNTErrors("Something is wrong!");
    }
}

/*
element-wise division of two tensors (return an XTensor structure)
make a new tensor c to keep the result and return it

c(i) = a(i)/b(i)
where i is the index of the item

>> a - tensor a
>> b - tensor b
>> alpha - the coefficient
>> leadingDim - the dimension along which we perform broadcasting
<< return - the product of the tensors
*/
XTensor Div(const XTensor & a, const XTensor & b, int leadingDim)
{
    XTensor c(&a);
    c.SetTMPFlag();

    if (b.order == 0){
        DTYPE scale = 1.0F / b.Get0D();
        ScaleAndShift(a, c, scale, 0.0F);
    }
    else {
        DTYPE alpha = 0.0F;
        int n = GetBroadcastDimIndex(a, b);

        if(n == -1){
            CheckNTErrors(a.dimSize[leadingDim] == b.dimSize[leadingDim], "TODO!");

            /* call _Div function */
            _Div(&a, &b, &c, alpha, leadingDim);

            /* tensor connections */
            if (a.enableGrad && b.enableGrad) {
                XLink::MakeLink(&a, &b, &c, MATH_DIV);
            }
        }
        else if(n >= 0 && n < a.order){
            /* call _DivDim function */
            _DivDim(&a, &b, &c, n, alpha);

            /* tensor connections */
            if (a.enableGrad && b.enableGrad) {
                XLink::MakeLink(&a, &b, &c, MATH_DIVDIM);
                XLink::AddParamToHeadInt(&c, n);
            }
        }
        else{
            ShowNTErrors("Something is wrong!");
        }
    }

    return c;
}

/*
element-wise division of two tensors

c(i) = a(i)/b(i) + \alpha * c(i)
where i is the index of the item

>> a - tensor a
>> b - tensor b
>> c - result tensor
>> alpha - the coefficient
>> leadingDim - the dimension along which we perform broadcasting
*/
void Div(const XTensor & a, const XTensor & b, XTensor & c, DTYPE alpha, int leadingDim)
{
    if (!c.isInit || !IsSameShaped(a, c)) {
        InitTensorV2(&c, &a);
    }

    if (b.order == 0){
        DTYPE scale = 1.0F / b.Get0D();
        if (a.mem != NULL)
            a.mem->LockBuf();
        XTensor * tmp1 = NewTensorBufV2(&a, a.devID, a.mem);
        if ((c.mem != NULL) && (c.mem != a.mem)) {
            c.mem->LockBuf();
        }
        XTensor * tmp2 = NewTensorBufV2(&c, c.devID, c.mem);

        ScaleAndShift(a, *tmp1, scale, 0.0F);
        ScaleAndShift(c, *tmp2, alpha, 0.0F);
        Sum(*tmp2, *tmp1, c);

        DelTensorBuf(tmp2);
        if ((c.mem != NULL) && (c.mem != a.mem)) {
            c.mem->UnlockBuf();
        }
        DelTensorBuf(tmp1);
        if (a.mem != NULL)
            a.mem->UnlockBuf();
    }
    else {
        int n = GetBroadcastDimIndex(a, b);

        if (n == -1) {
            CheckNTErrors(a.dimSize[leadingDim] == b.dimSize[leadingDim], "TODO!");

            /* call _Div function */
            _Div(&a, &b, &c, alpha, leadingDim);

            if (a.enableGrad && b.enableGrad) {
                /* tensor connections */
                XLink::MakeLink(&a, &b, &c, MATH_DIV);
            }
        }
        else if (n >= 0 && n < a.order) {
            /* call _DivDim function */
            _DivDim(&a, &b, &c, n, alpha);

            if (a.enableGrad && b.enableGrad) {
                /* tensor connections */
                XLink::MakeLink(&a, &b, &c, MATH_DIVDIM);
                XLink::AddParamToHeadInt(&c, n);
            }
        }
        else {
            ShowNTErrors("Something is wrong!");
        }
    }
}

} // namespace nts(NiuTrans.Tensor)
