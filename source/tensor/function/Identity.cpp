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

#include "../XName.h"
#include "Identity.h"
#include "CrossEntropy.h"
#include "../XUtility.h"
#include "../core/movement/CopyValues.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/* 
identity function y = x 
>> x - input tensor
>> y - result
*/
void _Identity(const XTensor * x, XTensor * y)
{
    _CopyValues(x, y);
}

/* 
identity function y = x (return an XTensor structure) 
make a new tensor to keep the result and return it

>> x - input tensor
<< return - y
*/
XTensor Identity(const XTensor &x)
{
    XTensor y(&x);
    y.SetTMPFlag();

    /* call _Identity function */
    _Identity(&x, &y);

    /* tensor connection */
    XLink::MakeLink(&x, NULL, &y, FUNC_IDENTITY);

    return y;
}
/* 
backward computation for identity function y = x 

dE/dx = dE/dy * dy/dx = dE/dy

>> gold - gold standard to measure error (or loss)
>> y - output of the function
>> x - input of the function
>> dedy - dE/dy
>> dedx - dE/dx
>> lossName - type of loss function, e.g., cross entropy
*/
void _IdentityBackward(XTensor * gold, XTensor * y, XTensor * x, 
                       XTensor * dedy, XTensor * dedx,
                       LOSS_FUNCTION_NAME lossName)
{
    CheckNTErrors((gold == NULL || XTensor::IsSameShaped(gold, y)), 
                  "The tensors must be of the same size!");

    if(x->dataType == DEFAULT_DTYPE && y->dataType == DEFAULT_DTYPE)
    {
        /* calculate dE/dy */
        if(lossName == CROSSENTROPY)
            _CrossEntropyBackward(dedy, y, gold);
        else if(lossName != NOLOSS)
            _LossBackward(dedy, gold, y, lossName);

        if(dedy->data != dedx->data)
            _CopyValues(dedy, dedx);
    }
    else
        ShowNTErrors("TODO!");
}

} // namespace nts(NiuTrans.Tensor)
