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
 * $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-04-25
 */

#include "../XName.h"
#include "../core/shape/IsSameShaped.h"
#include <math.h>
#include "Sigmoid.h"
#include "Sigmoid.cuh"
#include "../loss/LHeader.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/*
sigmoid function y = 1/(1+exp(-x))
>> x - input tensor
>> y - result
*/
void _Sigmoid(const XTensor * x, XTensor * y)
{
    CheckNTErrors(_IsSameShaped(x, y), 
                 "The input tensor and output tensor must have the same shape!")

#ifdef USE_CUDA
    if(x->devID >= 0 || y->devID >= 0){
        _CudaSigmoid(x, y);
        return;
    }
#endif

    if(x->dataType == DEFAULT_DTYPE && y->dataType == DEFAULT_DTYPE){
        DTYPE * ip = (DTYPE*)x->data;
        DTYPE * op = (DTYPE*)y->data;
        int n = x->GetSize();
        for(int i = 0; i < n; i++){
            DTYPE p = ip[i];
            op[i] = (DTYPE)1.0/((DTYPE)1.0+(DTYPE)exp(-p));
        }
    }
    else
        ShowNTErrors("TODO!");
}

/*
sigmoid function y = 1/(1+exp(-x)) (return an XTensor structure) 
make a new tensor to keep the result and return it

>> x - input tensor
<< return - output tensor
*/
XTensor Sigmoid(const XTensor &x)
{
    XTensor y(&x);
    y.SetTMPFlag();

    /* call _Sigmoid function */
    _Sigmoid(&x, &y);

    /* tensor connection */
    if (x.enableGrad) {
        XLink::MakeLink(&x, NULL, &y, FUNC_SIGMOID);
    }

    return y;
}

void Sigmoid(const XTensor &x, XTensor &y)
{
    if (!y.isInit || !IsSameShaped(y, x)) {
        InitTensorV2(&y, &x);
    }

    /* call _Sigmoid function */
    _Sigmoid(&x, &y);

    if (x.enableGrad) {
        /* tensor connection */
        XLink::MakeLink(&x, NULL, &y, FUNC_SIGMOID);
    }
}

/*
backward computation

dE/ds = dE/dy * dy/dx

sigmoid: y = 1/(1+exp(-x))

   and dy/dx = y * (1 - y)

>> y - output of the function
>> x - input of the function
>> dedy - dE/dy
>> dedx - dE/dx
*/
void _SigmoidBackward(XTensor * y, XTensor * x, 
                      XTensor * dedy, XTensor * dedx)
{
#ifdef USE_CUDA
    if(x->devID >= 0){
        _CudaSigmoidBackward(y, x, dedy, dedx);
        return;
    }
#endif
    DTYPE * dedyp = (DTYPE*)dedy->data;
    DTYPE * dedxp = (DTYPE*)dedx->data;
    DTYPE * op = (DTYPE*)y->data;
    int size = y->unitNum;

    /* dE/dx = dE/dy * dy/dx */
    for(int i = 0; i < size; i++){
        DTYPE y = op[i];
        dedxp[i] = dedyp[i] * (DTYPE)y * ((DTYPE)1.0 - y);
    }
}

} // namespace nts(NiuTrans.Tensor)
