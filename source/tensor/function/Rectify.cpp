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
#include "../core/shape/IsSameShaped.h"
#include "Rectify.h"
#include "Rectify.cuh"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/*
rectify function y = max(0, x)
>> x - input tensor
>> y - output tensor
*/
void _Rectify(const XTensor * x, XTensor * y)
{
    CheckNTErrors(_IsSameShaped(x, y), 
                 "The input tensor and output tensor must have the same shape!")

#ifdef USE_CUDA
    if(x->devID >= 0 || y->devID >= 0){
        _CudaRectify(x, y);
        return;
    }
#endif

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

/*
rectify function y = max(0, x) (return an XTensor structure) 
make a new tensor to keep the result and return it

>> x - input tensor
<< return - output tensor
*/
XTensor Rectify(const XTensor &x)
{
    XTensor y(&x);
    y.SetTMPFlag();

    /* call _Rectify function */
    _Rectify(&x, &y);

    /* tensor connection */
    if (x.enableGrad) {
        XLink::MakeLink(&x, NULL, &y, FUNC_RECTIFY);
    }

    return y;
}

void Rectify(const XTensor &x, XTensor &y)
{
    if (!y.isInit || !IsSameShaped(y, x)) {
        InitTensorV2(&y, &x);
    }

    /* call _Rectify function */
    _Rectify(&x, &y);

    if (x.enableGrad) {
        /* tensor connection */
        XLink::MakeLink(&x, NULL, &y, FUNC_RECTIFY);
    }
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

>> y - output of the rectify function
>> x - input of the rectify function
>> dedy - dE/dy
>> dedx - dE/dx
*/
void _RectifyBackward(XTensor * y, XTensor * x, 
                      XTensor * dedy, XTensor * dedx)
{
    CheckNTErrors(x != NULL, "The input tensor x must be not NULL!")

#ifdef USE_CUDA
    if(x->devID >= 0){
        _CudaRectifyBackward(y, x, dedy, dedx);
        return;
    }
#endif

    DTYPE * dedyp = (DTYPE*)dedy->data;
    DTYPE * dedxp = (DTYPE*)dedx->data;
    DTYPE * ip = (DTYPE*)x->data;
    int size = x->unitNum;

    for(int i = 0; i < size; i++){
        /* dE/ds = dE/dy * dy/ds = dE/dy */
        DTYPE s = ip[i];
        if(s < 0)
            dedxp[i] = 0;
        else
            dedxp[i] = dedyp[i];
    }
}

} // namespace nts(NiuTrans.Tensor)
