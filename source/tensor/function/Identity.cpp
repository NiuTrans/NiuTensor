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
* $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-04-27
*/

#include "Identity.h"
#include "../XName.h"
#include "../XUtility.h"
#include "../core/movement/CopyValues.h"
#include "../core/shape/IsSameShaped.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/* 
identity function y = x 
>> x - input tensor
>> y - output tensor
*/
void _Identity(const XTensor * x, XTensor * y)
{
    CheckNTErrors(_IsSameShaped(x, y), 
                 "The input tensor and output tensor must have the same shape!")
    _CopyValues(x, y);
}

/* 
identity function y = x (return an XTensor structure) 
make a new tensor to keep the result and return it

>> x - input tensor
<< return - output tensor
*/
XTensor Identity(const XTensor &x)
{
    XTensor y(&x);
    y.SetTMPFlag();

    /* call _Identity function */
    _Identity(&x, &y);

    /* tensor connection */
    if (x.enableGrad) {
        XLink::MakeLink(&x, NULL, &y, FUNC_IDENTITY);
    }

    return y;
}

void Identity(const XTensor &x, XTensor &y)
{
    if (!y.isInit || !IsSameShaped(y, x)) {
        InitTensorV2(&y, &x);
    }

    /* call _Identity function */
    _Identity(&x, &y);

    if (x.enableGrad) {
        /* tensor connection */
        XLink::MakeLink(&x, NULL, &y, FUNC_IDENTITY);
    }
}

/* 
backward computation for identity function y = x 

dE/dx = dE/dy * dy/dx = dE/dy

>> y - output of the identity function
>> x - input of the identity function
>> dedy - dE/dy
>> dedx - dE/dx
*/
void _IdentityBackward(const XTensor * y, const XTensor * x,
                       const XTensor * dedy, XTensor * dedx)
{
    if(dedy->data != dedx->data)
        _CopyValues(dedy, dedx);
}

} // namespace nts(NiuTrans.Tensor)
