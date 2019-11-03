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
#include "../../XUtility.h"
#include "../../XBLAS.h"
#include "../movement/CopyValues.h"
#include "../shape/IsSameShaped.h"
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
                if( c == a){
                    AXPY(a->unitNum,beta,bp,1,cp,1);
                } else{
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
void SumMe(XTensor& a, const XTensor& b, DTYPE beta)
{
    _Sum(&a, &b, &a, beta);
}

/* 
return a dimension if the sum is performed as SumDim (in more details in SumDim.h)
>> a - a tensor
>> b - another tensor for sum
*/
int GetSumDimIndex(const XTensor &a, const XTensor &b)
{
    if(a.order < b.order)
        return -1;
    if(IsSameShaped(a, b))
        return -1;

    int hitCount = 0;
    int hitDim = -1;
    for(int i = 0; i < b.order; i++){
        if(b.dimSize[b.order - 1 - i] == 1)
            continue;
        else if(b.dimSize[b.order - 1 - i] == a.dimSize[a.order - 1 - i]){
            hitCount++;
            hitDim = a.order - b.order + i;
        }
    }

    if(hitCount == 1)
        return hitDim;
    else
        return -1;
}
    
/*
tensor summation c = a + b * \beta (return an XTensor structure)
make a new tensor c to keep the result and return it

>> a - a tensor
>> b - another tensor
>> beta - the scaling factor
<< return - the result of tensor summation
*/
XTensor Sum(const XTensor &a, const XTensor &b, DTYPE beta)
{
    XTensor c(&a);
    c.SetTMPFlag();

    int n = GetSumDimIndex(a, b);

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
    
    return c;
}

/*
tensor summation c = a + b * \beta

>> a - a tensor
>> b - another tensor
>> beta - the scaling factor
*/
void Sum(const XTensor &a, const XTensor &b, XTensor &c, DTYPE beta)
{
    if (!c.isInit || !IsSameShaped(a, c)) {
        InitTensorV2(&c, &a);
    }

    int n = GetSumDimIndex(a, b);

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

} // namespace nts(NiuTrans.Tensor)
