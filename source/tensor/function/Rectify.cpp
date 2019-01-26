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

#include "../XName.h"
#include "Rectify.h"
#include "Rectify.cuh"
#include "CrossEntropy.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/*
rectify function y = max(0, x)
>> input - input tensor
>> output - result
*/
void _Rectify(const XTensor * x, XTensor * y)
{
#ifdef USE_CUDA
    if(x->devID >= 0 || y->devID >= 0){
        _CudaRectify(x, y);
        return;
    }
#endif

    if(x->dataType == DEFAULT_DTYPE && y->dataType == DEFAULT_DTYPE){
        DTYPE * ip = (DTYPE*)x->data;
        DTYPE * op = (DTYPE*)y->data;
        int n = x->GetSize();
        for(int i = 0; i < n; i++){
            DTYPE p = ip[i];
            if(p < 0)
                p = 0;

            op[i] = p;
        }
    }
    else
        ShowNTErrors("TODO!");
}

/*
rectify function y = max(0, x) (return an XTensor structure) 
make a new tensor to keep the result and return it

>> input - input tensor
<< return - y
*/
XTensor Rectify(const XTensor &x)
{
    XTensor y(&x);
    y.SetTMPFlag();

    /* call _Rectify function */
    _Rectify(&x, &y);

    /* tensor connection */
    XLink::MakeLink(&x, NULL, &y, FUNC_RECTIFY);

    return y;
}


/*
backward computation

dE/dx = dE/dy * dy/dx

rectified: y = max(0, x)

or

rectified: y = 0     if x < 0
               x     otherwise

   and dy/ds = 0     if x < 0
               1     otherwise

>> gold - gold standard to measure error (or loss)
>> y - output of the function
>> x - input of the function
>> dedy - dE/dy
>> dedx - dE/dx
>> lossName - type of loss function, e.g., cross entropy
*/
void _RectifyBackward(XTensor * gold, XTensor * y, XTensor * x, 
                      XTensor * dedy, XTensor * dedx,
                      LOSS_FUNCTION_NAME lossName)
{
    CheckNTErrors((gold == NULL || XTensor::IsSameShaped(gold, y)), 
                  "The tensors must be of the same size!");

#ifdef USE_CUDA
    if(x->devID >= 0 || y->devID >= 0){
        _CudaRectifyBackward(gold, y, x, dedy, dedx, lossName);
        return;
    }
#endif

    if(x->dataType == DEFAULT_DTYPE && y->dataType == DEFAULT_DTYPE)
    {
        /* calculate dE/dy */
        if(lossName == CROSSENTROPY)
            _CrossEntropyBackward(dedy, y, gold);
        else if(lossName != NOLOSS)
            _LossBackward(dedy, gold, y, lossName);

        DTYPE * dedyp = (DTYPE*)dedy->data;
        DTYPE * dedxp = (DTYPE*)dedx->data;
        DTYPE * ip = (DTYPE*)x->data;
        int size = y->unitNum;
        for(int i = 0; i < size; i++){
            /* dE/ds = dE/dy * dy/ds = dE/dy */
            DTYPE s = ip[i];
            if(s < 0)
                dedxp[i] = 0;
            else
                dedxp[i] = dedyp[i];
        }
    }
    else
        ShowNTErrors("TODO!");
}

} // namespace nts(NiuTrans.Tensor)
