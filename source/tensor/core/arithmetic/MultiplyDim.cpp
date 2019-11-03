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
 * $Created by: JIANG Yufan (email: jiangyufan2018@outlook.com) 2018-08-14
 */

#include <math.h>
#include "Multiply.h"
#include "MultiplyDim.h"
#include "MultiplyDim.cuh"
#include "../shape/Unsqueeze.h"
#include "../shape/IsSameShaped.h"
#include "../../XName.h"
#include "../../XUtility.h"
#include "../movement/CopyValues.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
tensor multiplication

c = a * b + \alpha * c
where the size of b is equal to the n-th dimension of a,
i.e., a is multiplied with b by broadcasting

>> a - a tensor
>> b - another tensor whose size is equal to that of dimension n of a
>> c - where we put a * b + \alpha * c. we save it in a if c is NULL
>> n - the dimension index
>> alpha - the scaling factor
*/
void _MultiplyDim(const XTensor * a, const XTensor * b, XTensor * c, int n, DTYPE alpha) 
{
    n = MODX(n, a->order);

    CheckNTErrors(a && b && c, "Empty tensor input!");
    CheckNTErrors(a->unitNum == c->unitNum, "Unmatched tensors in multiplication!");
    CheckNTErrors(a->dataType == b->dataType && a->dataType == c->dataType,
                 "Unmatched data types in multiplication!");
    CheckNTErrors(a->order == c->order, "The input tensors do not have the same order in multiplication!");
    CheckNTErrors(!a->isSparse && !b->isSparse && !c->isSparse, "Dense tensors are required!");
    CheckNTErrors(a->dimSize[n] == b->unitNum, "Wrong tensor size!");

    CheckDev(a->devID, b->devID);

    if(_IsSameShaped(a, b)){
        _Multiply(a, b, c, alpha);
        return;
    }

    if(a->devID >= 0 || b->devID >= 0 || c->devID >= 0){
#ifdef USE_CUDA
        _CudaMultiplyDim(a, b, c, n, alpha);
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

        if(a->dataType == DEFAULT_DTYPE){
            int num = a->unitNum;
            if(stride > 1){
                for(int i = 0, j = 0; i < num; i += stride, j++){
                    DTYPE * ap = (DTYPE*)a->data + i;
                    DTYPE   bv = *((DTYPE*)b->data + j % blockSize);
                    DTYPE * cp = (DTYPE*)c->data + i;
                    for(int k = 0; k < stride; k++)
                        if(alpha == 0.0F)
                            cp[k] = ap[k] * bv;
                        else
                            cp[k] = ap[k] * bv + alpha * cp[k];
                }
            }
            else if(stride == 1){
                DTYPE * bp = (DTYPE*)b->data;
                for(int i = 0; i < num; i += blockSize){
                    DTYPE * ap = (DTYPE*)a->data + i;
                    DTYPE * cp = (DTYPE*)c->data + i;
                    if(alpha == 0.0F){
                        for(int j = 0; j < blockSize; j++)
                            cp[j] = ap[j] * bp[j];
                    }
                    else{
                        for(int j = 0; j < blockSize; j++)
                            cp[j] = ap[j] * bp[j] + alpha * cp[j];
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
tensor multiplication(do it on site)
make a new tensor to keep the result and return it

c = a * b + \alpha * c
where the size of b is equal to the n-th dimension of a,
i.e., a is multiplied with b by broadcasting

>> a - a tensor
>> b - another tensor whose size is equal to that of dimension n of a
>> n - the dimension index
>> alpha - the scaling factor
*/
void _MultiplyDimMe(XTensor * a, const XTensor * b, int n, DTYPE alpha)
{
    _MultiplyDim(a, b, a, n, alpha);
}

/*
tensor multiplication(do it on site)
make a new tensor to keep the result and return it

c = a * b + \alpha * c
where the size of b is equal to the n-th dimension of a,
i.e., a is multiplied with b by broadcasting

>> a - a tensor
>> b - another tensor whose size is equal to that of dimension n of a
>> n - the dimension index
>> alpha - the scaling factor
*/
void MultiplyDimMe(XTensor& a, const XTensor& b, int n, DTYPE alpha)
{
    _MultiplyDim(&a, &b, &a, n, alpha);
}

/*
tensor multiplication (return an XTensor structure and make tensor connections)
make a new tensor to keep the result and return it

c = a * b
where the size of b is equal to the n-th dimension of a,
i.e., a is multiplied with b by broadcasting

>> a - a tensor
>> b - another tensor whose size is equal to that of dimension n of a
>> n - the dimension index
<< return - the result tensor by tensor multiplication
*/
XTensor MultiplyDim(const XTensor &a, const XTensor &b, int n)
{
    XTensor c(&a);
    c.SetTMPFlag();

    n = MODX(n, a.order);

    /* call _Multiply function */
    _MultiplyDim(&a, &b, &c, n, 0);

    /* tensor connections */
    if (a.enableGrad && b.enableGrad) {
        XLink::MakeLink(&a, &b, &c, MATH_MULTIPLYDIM);
        XLink::AddParamToHeadInt(&c, n);
        XLink::AddParamToHead(&c, 0);
    }

    return c;
}

/*
tensor multiplication

c = a * b + \alpha * c
where the size of b is equal to the n-th dimension of a,
i.e., a is multiplied with b by broadcasting

>> a - a tensor
>> b - another tensor whose size is equal to that of dimension n of a
>> c - where we put a * b + \alpha * c. we save it in a if c is NULL
>> n - the dimension index
*/
void MultiplyDim(const XTensor &a, const XTensor &b, XTensor &c, int n)
{
    if (!c.isInit || !IsSameShaped(a, c)) {
        InitTensorV2(&c, &a);
    }

    /* call _Multiply function */
    _MultiplyDim(&a, &b, &c, n, 0);

    if (a.enableGrad && b.enableGrad) {
        /* tensor connections */
        XLink::MakeLink(&a, &b, &c, MATH_MULTIPLYDIM);
        XLink::AddParamToHeadInt(&c, n);
        XLink::AddParamToHead(&c, 0);
    }
}

/* 
tensor broadcast multiplication
c = a * b + c * \beta 
where some of dimensions of b can be of size 1

>> a - a tensor
>> b - another tensor that would be broadcasted
>> c - the resulting tensor
>> beta - the scaling factor
*/
void _MultiplyBroadcast(const XTensor * a, const XTensor * b, XTensor * c, DTYPE beta)
{
    CheckNTErrors(a->order == b->order, "Wrong tensor orders!");
    CheckNTErrors(a->order == c->order, "Wrong tensor orders!");
    CheckNTErrors(a->order > 0, "TODO!");
    
    int order = a->order;
    int count = 0;
    void * source = 0;
    void * target = 0;
    
    for(int i = 0; i < order; i++){
        if(a->GetDim(i) == b->GetDim(i))
            continue;
        
        if(b->GetDim(i) == 1){
            int fitSize = a->GetDim(i);
            int j = i + 1;
            
            /* we define a range over dimensions. It is to be unsqueezed */
            for(; j < order; j++){
                if(a->GetDim(j) == b->GetDim(j))
                    break;
                fitSize *= a->GetDim(j);
            }
            
            int dimsS[MAX_TENSOR_DIM_NUM];
            int dimsT[MAX_TENSOR_DIM_NUM];
            
            for(int k = 0; k < i; k++){
                dimsS[k] = a->GetDim(k);
                dimsT[k] = a->GetDim(k);
            }
            
            dimsT[i] = fitSize;
            
            bool isLast = true;
            for(int k = j; k < order; k++){
                dimsS[i + k - j + 0] = b->GetDim(k);
                dimsT[i + k - j + 1] = b->GetDim(k);
                if(a->GetDim(k) != b->GetDim(k)){
                    if(b->GetDim(k) == 1)
                        isLast = false;
                    else{
                        ShowNTErrors("Wrong dimension size!")
                    }
                }
            }
            
            dimsS[0] = -dimsS[0];
            dimsT[0] = -dimsT[0];
            
            XTensor * s = NewTensorV2(order - (j - i), dimsS, a->dataType, a->denseRatio, a->devID, a->mem);
            XTensor * t = NewTensorV2(order - (j - i) + 1, dimsT, b->dataType, b->denseRatio, b->devID, b->mem);
            
            if(count == 0)
                source = b->data;
            else{
                source = target;
            }
            
            target = t->mem != NULL ?
                     t->mem->AllocBuf(t->devID, t->unitNum * t->unitSize):
                     XMemAlloc(t->devID, t->unitNum * t->unitSize);
            
            s->data = source;
            t->data = target;
            
            _Unsqueeze(s, t, i, fitSize);
            
            /* free the memory space of the one before the last allocation */
            if(count > 0){
                int size = s->unitNum * s->unitSize;
                if(t->mem != NULL)
                    t->mem->ReleaseBuf(t->devID, size);
                else
                    XMemFree(t->devID, source);
            }
            
            /* we do multiplication here */
            if(isLast){
                CheckNTErrors(t->unitNum == c->unitNum, "Wrong tensor size!");
                _Multiply(a, t, c, beta);
                if(t->mem != NULL)
                    t->mem->ReleaseBuf(t->devID, t->unitNum * t->unitSize);
                else
                    XMemFree(t->devID, target);
                target = NULL;
            }
            
            s->data = NULL;
            t->data = NULL;
            DelTensor(s);
            DelTensor(t);
            
            i = j;
            count++;
        }
    }

    if(count == 0)
        _Multiply(a, b, c, beta);
    
    CheckNTErrors(target == NULL, "Something is wrong!");
}

/* 
tensor broadcast multiplication
c = a * b
where some of dimensions of b can be of size 1

>> a - a tensor
>> b - another tensor that would be broadcasted
<< return - the resulting tensor c
*/
XTensor MultiplyBroadcast(const XTensor &a, const XTensor &b)
{
    XTensor c(&a);
    c.SetTMPFlag();
    
    /* call _SumBroadcast function */
    _MultiplyBroadcast(&a, &b, &c, 0);
    
    /* tensor connections */
    if (a.enableGrad && b.enableGrad) {
        XLink::MakeLink(&a, &b, &c, MATH_MULTIPLYBROADCAST);
        XLink::AddParamToHead(&c, 0);
    }
    
    return c;
}

/* 
tensor broadcast multiplication
c = a * b + c * \beta 
where some of dimensions of b can be of size 1

>> a - a tensor
>> b - another tensor that would be broadcasted
>> c - the resulting tensor
*/
void MultiplyBroadcast(const XTensor &a, const XTensor &b, XTensor &c)
{
    if (!c.isInit || !IsSameShaped(a, c)) {
        InitTensorV2(&c, &a);
    }

    /* call _SumBroadcast function */
    _MultiplyBroadcast(&a, &b, &c, 0);

    if (a.enableGrad && b.enableGrad) {
        /* tensor connections */
        XLink::MakeLink(&a, &b, &c, MATH_MULTIPLYBROADCAST);
        XLink::AddParamToHead(&c, 0);
    }
}

}
