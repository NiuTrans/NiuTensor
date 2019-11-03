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

#include <stdlib.h>
#include "../XName.h"
#include "../../tensor/core/shape/IsSameShaped.h"
#include "HardTanH.h"
#include "HardTanH.cuh"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/*
hard tanh function 
y =  1    if x > 1
     x    if -1 <= x <= 1
    -1    if x < -1
>> x - input tensor
>> y - result
*/
void _HardTanH(const XTensor * x, XTensor * y)
{
    CheckNTErrors(_IsSameShaped(x, y), 
                 "The input tensor and output tensor must have the same shape!")

#ifdef USE_CUDA
    if(x->devID >= 0 || y->devID >= 0){
        _CudaHardTanH(x, y);
        return;
    }
#endif

    int n = x->GetSize();
    DTYPE * ip = (DTYPE*)x->data;
    DTYPE * op = (DTYPE*)y->data;
    for(int i = 0; i < n; i++){
        DTYPE p = ip[i];
        if(p > 1.0)
            p = 1.0;
        else if(p < -1.0)
            p = -1.0;
        op[i] = p;
    }
}

/* 
hard tanh function (return an XTensor structure) 
make a new tensor to keep the result and return it

y =  1    if x > 1
     x    if -1 <= x <= 1
    -1    if x < -1
>> x - input tensor
<< return - y
*/
XTensor HardTanH(const XTensor &x)
{
    XTensor y(&x);
    y.SetTMPFlag();

    /* call _HardTanH function */
    _HardTanH(&x, &y);

    /* tensor connection */
    if (x.enableGrad) {
        XLink::MakeLink(&x, NULL, &y, FUNC_HARDTANH);
    }

    return y;
}

void HardTanH(const XTensor &x, XTensor &y)
{
    if (!y.isInit || !IsSameShaped(y, x)) {
        InitTensorV2(&y, &x);
    }

    /* call _HardTanH function */
    _HardTanH(&x, &y);

    if (x.enableGrad) {
        /* tensor connection */
        XLink::MakeLink(&x, NULL, &y, FUNC_HARDTANH);
    }
}

/*
backward computation

dE/dx = dE/dy * dy/dx

hard tanh: y =  1    if x > 1
                x    if -1 <= x <= 1
               -1    if x< -1

   and dy/dx =  1    if -1 <= x <= 1
                0    otherwise

>> y - output of the hardtanh function
>> x - input of the hardtanh function
>> dedy - dE/dy
>> dedx - dE/dx
*/
void _HardTanHBackward(XTensor * y, XTensor * x, 
                       XTensor * dedy, XTensor * dedx)
{
    CheckNTErrors(x != NULL, "The input tensor x must be not NULL!")

#ifdef USE_CUDA
    if(x->devID >= 0){
        _CudaHardTanHBackward(y, x, dedy, dedx);
        return;
    }
#endif

    DTYPE * dedyp = (DTYPE*)dedy->data;
    DTYPE * dedxp = (DTYPE*)dedx->data;
    DTYPE * ip = (DTYPE*)x->data;
    int size = x->unitNum;

    /* dE/dx = dE/dy * dy/dx */
    for(int i = 0; i < size; i++){
        DTYPE s =ip[i];
        if(s > 1.0 || s < -1.0)
            dedxp[i] = 0;
        else
            dedxp[i] = dedyp[i];
    }
}

} // namespace nts(NiuTrans.Tensor)
