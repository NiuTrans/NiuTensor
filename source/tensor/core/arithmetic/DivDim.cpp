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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-08-15
 */

#include <math.h>
#include "Div.h"
#include "DivDim.h"
#include "DivDim.cuh"
#include "../../XName.h"
#include "../../XUtility.h"
#include "../movement/CopyValues.h"
#include "../shape/IsSameShaped.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
tensor division

c = a / b + \alpha * c
where the size of b is equal to the n-th dimension of a, 
i.e., a is divided with b by broadcasting 

>> a - a tensor
>> b - another tensor whose size is equal to that of dimension n of a
>> c - where we put result. we save it in a if c is NULL
>> n - the dimension index
>> alpha - the scaling factor
*/
void _DivDim(const XTensor * a, const XTensor * b, XTensor * c, int n, DTYPE alpha)
{
    n = MODX(n, a->order);

    CheckNTErrors(a && b && c, "Empty tensor input!");
    CheckNTErrors(a->unitNum == c->unitNum, "Unmatched tensors in division!");
    CheckNTErrors(a->dataType == b->dataType && a->dataType == c->dataType,
                 "Unmatched data types in addition!");
    CheckNTErrors(a->order == c->order, "The input tensors do not have the same order in division!");
    CheckNTErrors(!a->isSparse && !b->isSparse && !c->isSparse, "Dense tensors are required!");
    CheckNTErrors(a->dimSize[n] == b->unitNum, "Wrong tensor size!");

    CheckDev(a->devID, b->devID);

    if(_IsSameShaped(a, b)){
        _Div(a, b, c, alpha);
        return;
    }

    if(a->devID >= 0 || b->devID >= 0 || c->devID >= 0){
#ifdef USE_CUDA
        _CudaDivDim(a, b, c, n, alpha);
#else
        ShowNTErrors("Please specify USE_CUDA and recompile the code!");
#endif
    }
    else{
        int stride = 1;
        int blockSize = a->dimSize[n];
        int blockNum = 1;

        for(int i = a->order - 1; i >= 0; i--){
            if(i > n)
                stride *= a->dimSize[i];
            else if(i < n)
                blockNum *= a->dimSize[i];
        }
    
        if (a->dataType == DEFAULT_DTYPE){
            int num = a->unitNum;
            if(stride > 1){
                for(int i = 0, j = 0; i < num; i += stride, j++){
                    DTYPE * ap = (DTYPE*)a->data + i;
                    DTYPE   bv = *((DTYPE*)b->data + j % blockSize);
                    DTYPE * cp = (DTYPE*)c->data + i;
                    for(int k = 0; k < stride; k++){
                        if(alpha == 0.0F)
                            cp[k] = ap[k] / bv;
                        else
                            cp[k] = ap[k] / bv + alpha * cp[k];
                    }
                }
            }
            else if(stride == 1){
                DTYPE * bp = (DTYPE*)b->data;
                for(int i = 0; i < num; i += blockSize){
                    DTYPE * ap = (DTYPE*)a->data + i;
                    DTYPE * cp = (DTYPE*)c->data + i;
                    if(alpha == 0.0F){
                        for(int j = 0; j < blockSize; j++)
                            cp[j] = ap[j] / bp[j];
                    }
                    else{
                        for(int j = 0; j < blockSize; j++)
                            cp[j] = ap[j] / bp[j] + alpha * cp[j];
                    }
                }
            }
            else{
                ShowNTErrors("Something is wrong!");
            }
        }
        else {
            ShowNTErrors("TODO!");
        }
    }
}
    
/*
tensor division of two tensors (do it on site)
keep the result in the input tensor and return nothing

a = a/b + \alpha * a
where the size of b is equal to the n-th dimension of a, 
i.e., a is divided with b by broadcasting 

>> a - a tensor
>> b - another tensor whose size is equal to that of dimension n of a
>> n - the dimension index
>> alpha - the scaling factor
*/
void _DivDim(XTensor * a, const XTensor * b, int n, DTYPE alpha)
{
    _DivDim(a, b, a, n, alpha);
}
    
/*
tensor division of two tensors (return an XTensor structure and make tensor connections)
make a new tensor to keep the result and return it

c = a/b + \alpha * c
where the size of b is equal to the n-th dimension of a, 
i.e., a is divided with b by broadcasting 


>> a - a tensor
>> b - another tensor whose size is equal to that of dimension n of a
>> n - the dimension index
>> alpha - the scaling factor
<< return - the result tensor by tensor division
*/
XTensor DivDim(const XTensor &a, const XTensor &b, int n, DTYPE alpha)
{
    XTensor c(&a);
    c.SetTMPFlag();

    n = MODX(n, a.order);
    
    /* call _Div function */
    _DivDim(&a, &b, &c, n, alpha);
    
    /* tensor connections */
    if (a.enableGrad && b.enableGrad) {
        XLink::MakeLink(&a, &b, &c, MATH_DIVDIM);
        XLink::AddParamToHeadInt(&c, n);
        XLink::AddParamToHead(&c, alpha);
    }

    return c;
}

/*
tensor division

c = a / b + \alpha * c
where the size of b is equal to the n-th dimension of a, 
i.e., a is divided with b by broadcasting 

>> a - a tensor
>> b - another tensor whose size is equal to that of dimension n of a
>> c - where we put result. we save it in a if c is NULL
>> n - the dimension index
>> alpha - the scaling factor
*/
void DivDim(const XTensor &a, const XTensor &b, XTensor &c, int n, DTYPE alpha)
{
    if (!c.isInit || !IsSameShaped(a, c)) {
        InitTensorV2(&c, &a);
    }

    /* call _Div function */
    _DivDim(&a, &b, &c, n, alpha);

    if (a.enableGrad && b.enableGrad) {
        /* tensor connections */
        XLink::MakeLink(&a, &b, &c, MATH_DIVDIM);
        XLink::AddParamToHeadInt(&c, n);
        XLink::AddParamToHead(&c, alpha);
    }
}
    
}
