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
* $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-04-24
*/

#include "../../XTensor.h"
#include "../../XName.h"
#include "../../XUtility.h"
#include "../../XBLAS.h"
#include "../movement/CopyValues.h"
#include "../shape/IsSameShaped.h"
#include "../math/ScaleAndShift.h"
#include "Sum.h"
#include "Sum.cuh"
#include "SumDim.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
tensor summation c = a + b * \beta

>> a - a tensor
>> b - another tensor
>> c - where we put a+b*\beta. we save it in a if c is NULL
>> beta - the scaling factor
*/
void _Sum(const XTensor * a, const XTensor * b, XTensor * c, DTYPE beta)
{
    CheckNTErrors(a && b && c, "Empty tensor input!");
    CheckNTErrors(a->unitNum == b->unitNum && a->unitNum == c->unitNum,
                  "Unmatched tensors in addition!");
    CheckNTErrors(a->dataType == b->dataType && a->dataType == c->dataType,
                  "Unmatched tensors in addition!");

    CheckDev(a->devID, b->devID);

    if(beta == 0){
        _CopyValues(a, c);
        return;
    }

    if (a->devID >= 0 || b->devID >= 0 || c->devID >= 0) {
#ifdef USE_CUDA
        if (a == c) {
            int P2PAccesible = 0;
#ifdef CUDA_UVA
            cudaDeviceCanAccessPeer(&P2PAccesible, a->devID, b->devID);
#endif
            if ((a->devID < 0 && b->devID >= 0) ||
                (a->devID >= 0 && b->devID < 0) ||
                (a->devID >= 0 && b->devID >= 0 && a->devID != b->devID && !P2PAccesible))
            {
                ShowNTErrors("Cannot run this method on multiple devices simultaneously!");
            }
            else
                _CudaSum(a, b, c, beta);
        }
        else
            _CudaSum(a, b, c, beta);

#endif
    }
    else {
        if (!a->isSparse && !b->isSparse) {
            CheckNTErrors(!c->isSparse, "Illegal use of sparse tensor in addition!");

            if (a->dataType == DEFAULT_DTYPE &&
                b->dataType == DEFAULT_DTYPE &&
                c->dataType == DEFAULT_DTYPE)
            {
                DTYPE * ap = (DTYPE*)a->data;
                DTYPE * bp = (DTYPE*)b->data;
                DTYPE * cp = (DTYPE*)c->data;
                /* when c != a, OpenBLAS needs to copy a to c first. This operation
                 slow down the speed, so just use OpenBLAS when c == a */
#if defined(USE_BLAS)
                if (c == a) {
                    AXPY(a->unitNum,beta,bp,1,cp,1);
                }
                else {
                    int num = a->unitNum;
                    if (num % 4 == 0) {
                        for (int i = 0; i < num; i += 4) {
                                cp[i] = ap[i] + bp[i] * beta;
                                cp[i + 1] = ap[i + 1] + bp[i + 1] * beta;
                                cp[i + 2] = ap[i + 2] + bp[i + 2] * beta;
                                cp[i + 3] = ap[i + 3] + bp[i + 3] * beta;
                        }
                    }
                    else if (num % 2 == 0) {
                        for (int i = 0; i < num; i += 2) {
                            cp[i] = ap[i] + bp[i] * beta;
                            cp[i + 1] = ap[i + 1] + bp[i + 1] * beta;
                        }
                    }
                    else {
                        for (int i = 0; i < num; i++) {
                            cp[i] = ap[i] + bp[i] * beta;
                        }
                    }
                }
#else
                /* unrolling */
                int num = a->unitNum;
                if (num % 4 == 0) {
                    for (int i = 0; i < num; i += 4) {
                        cp[i] = ap[i] + bp[i] * beta;
                        cp[i + 1] = ap[i + 1] + bp[i + 1] * beta;
                        cp[i + 2] = ap[i + 2] + bp[i + 2] * beta;
                        cp[i + 3] = ap[i + 3] + bp[i + 3] * beta;
                    }
                }
                else if (num % 2 == 0) {
                    for (int i = 0; i < num; i += 2) {
                        cp[i] = ap[i] + bp[i] * beta;
                        cp[i + 1] = ap[i + 1] + bp[i + 1] * beta;
                    }
                }
                else {
                    for (int i = 0; i < num; i++) {
                        cp[i] = ap[i] + bp[i] * beta;
                    }
                }
#endif
            }
            else if (a->dataType == X_INT &&
                     b->dataType == X_INT &&
                     c->dataType == X_INT)
            {
                int * ap = (int*)a->data;
                int * bp = (int*)b->data;
                int * cp = (int*)c->data;

                /* unrolling */
                int num = a->unitNum;
                if (num % 4 == 0) {
                    for (int i = 0; i < num; i += 4) {
                        cp[i] = ap[i] + bp[i] * beta;
                        cp[i + 1] = ap[i + 1] + bp[i + 1] * beta;
                        cp[i + 2] = ap[i + 2] + bp[i + 2] * beta;
                        cp[i + 3] = ap[i + 3] + bp[i + 3] * beta;
                    }
                }
                else if (num % 2 == 0) {
                    for (int i = 0; i < num; i += 2) {
                        cp[i] = ap[i] + bp[i] * beta;
                        cp[i + 1] = ap[i + 1] + bp[i + 1] * beta;
                    }
                }
                else {
                    for (int i = 0; i < num; i++) {
                        cp[i] = ap[i] + bp[i] * beta;
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
    
/*
tensor summation a = a + b * \beta (do it on site)
keep the result in the tensor a and return nothing

>> a - a tensor
>> b - another tensor
>> beta - the scaling factor
*/
void _SumMe(XTensor * a, const XTensor * b, DTYPE beta)
{
    _Sum(a, b, a, beta);
}

/*
tensor summation a = a + b * \beta (do it on site)
keep the result in the tensor a and return nothing

>> a - a tensor
>> b - another tensor
>> beta - the scaling factor
*/
void SumMe(XTensor & a, const XTensor & b, DTYPE beta)
{
    if (b.order == 0){
        DTYPE shift = b.Get0D() * beta;
        _ScaleAndShift(&a, &a, 1.0F, shift);
    }
    else {
        int n = GetBroadcastDimIndex(a, b);

        if (n == -1)
            /* call _Sum function */
            _Sum(&a, &b, &a, beta);
        else if (n >= 0 && n < a.order)
            /* call _SumDim function */
            _SumDim(&a, &b, &a, n, beta);
        else
            ShowNTErrors("Something is wrong!");
    }
}

/* 
return a dimension if the operation is performed as broadcast(e.g. SumDim function)
>> a - a tensor
>> b - another tensor for operation
*/
int GetBroadcastDimIndex(const XTensor & a, const XTensor & b)
{
    if(a.order < b.order)
        return -1;
    if(IsSameShaped(a, b))
        return -1;

    int hitDim = -1;
    bool isHit = false;
    for(int i = 0; i < b.order; i++){
        if(b.dimSize[b.order - 1 - i] == 1)
            continue;
        else {
            if (isHit == true)
                return -1;
            else
                isHit = true;
            for (int j = 0; j < a.order; j++){
                if (b.dimSize[b.order - 1 - i] == a.dimSize[a.order - 1 - j]){
                    hitDim = a.order - 1 - j;
                    break;
                }
            }
        }
    }

    return hitDim;
}
    
/*
tensor summation c = a + b * \beta (return an XTensor structure)
make a new tensor c to keep the result and return it

>> a - a tensor
>> b - another tensor
>> inplace - indicates whether the result will be placed in the input tensor
>> beta - the scaling factor
<< return - the result of tensor summation
*/
XTensor Sum(const XTensor &a, const XTensor &b, bool inplace, DTYPE beta)
{
    XTensor c;

    if (inplace) {
        /* the result is stored into the input tensor */
        int dims[MAX_TENSOR_DIM_NUM];
        memcpy(&(dims[0]), &(a.dimSize[0]), sizeof(int) * a.order);
        dims[0] = -dims[0];
        InitTensor(&c, a.order, dims, a.dataType, a.devID, a.enableGrad);
        c.data = a.data;
    }
    else {
        InitTensorV2(&c, &a);
    }
    c.SetTMPFlag();
    c.enableGrad = a.enableGrad;

    if (b.order == 0){
        DTYPE shift = b.Get0D() * beta;
        ScaleAndShift(a, c, 1.0F, shift);
    }
    else {
        int n = GetBroadcastDimIndex(a, b);

        if(n == -1){
            /* call _Sum function */
            _Sum(&a, &b, &c, beta);

            /* tensor connections */
            if (a.enableGrad && b.enableGrad) {
                XLink::MakeLink(&a, &b, &c, MATH_SUM);
                XLink::AddParamToHead(&c, beta);
            }
        }
        else if(n >= 0 && n < a.order){
            /* call _SumDim function */
            _SumDim(&a, &b, &c, n, beta);

            /* tensor connections */
            if (a.enableGrad && b.enableGrad) {
                XLink::MakeLink(&a, &b, &c, MATH_SUMDIM);
                XLink::AddParamToHeadInt(&c, n);
                XLink::AddParamToHead(&c, beta);
            }
        }
        else{
            ShowNTErrors("Something is wrong!");
        }
    }

    XTensor* p = const_cast<XTensor*>(&a);
    if (inplace)
        p->data = NULL;
    return c;
}

/*
tensor summation c = a + b * \beta

>> a - a tensor
>> b - another tensor
>> beta - the scaling factor
*/
void Sum(const XTensor & a, const XTensor & b, XTensor & c, DTYPE beta)
{
    if (!c.isInit || !IsSameShaped(a, c)) {
        InitTensorV2(&c, &a);
    }

    if (b.order == 0){
        DTYPE shift = b.Get0D() * beta;
        ScaleAndShift(a, c, 1.0F, shift);
    }
    else {
        int n = GetBroadcastDimIndex(a, b);

        if (n == -1) {
            /* call _Sum function */
            _Sum(&a, &b, &c, beta);

            /* tensor connections */
            if (a.enableGrad && b.enableGrad) {
                XLink::MakeLink(&a, &b, &c, MATH_SUM);
                XLink::AddParamToHead(&c, beta);
            }
        }
        else if (n >= 0 && n < a.order) {
            /* call _SumDim function */
            _SumDim(&a, &b, &c, n, beta);

            /* tensor connections */
            if (a.enableGrad && b.enableGrad) {
                XLink::MakeLink(&a, &b, &c, MATH_SUMDIM);
                XLink::AddParamToHeadInt(&c, n);
                XLink::AddParamToHead(&c, beta);
            }
        }
        else {
            ShowNTErrors("Something is wrong!");
        }
    }
}

} // namespace nts(NiuTrans.Tensor)
