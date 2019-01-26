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
* $Created by: LI Yinqiao (li.yin.qiao.2012@hotmail.com) 2018-7-11
*/

#include "../../XTensor.h"
#include "../../XName.h"
#include "Sign.h"
#include "Sign.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
set every entry to its sign value
>> a - input tensor we are processing
>> b - output tensor we are processing
*/
void _Sign(const XTensor * a, XTensor * b)
{
#ifdef USE_CUDA
    /* run it on GPUs */
    if (a->devID >= 0) {
        _CudaSign(a, b);
    return;
}
#endif

    CheckNTErrors((XTensor::IsSameShaped(a, b)), "Input tensors should have the same type!");
    CheckNTErrors((a->dataType == DEFAULT_DTYPE), "TODO!");
    DTYPE * d = (DTYPE*)a->data;
    DTYPE * db = (DTYPE*)b->data;
    for (int i = 0; i < a->unitNum; i++) {
        if (d[i] > 0)
            db[i] = 1.0F;
        else if (d[i] == 0)
            db[i] = 0.0F;
        else
            db[i] = -1.0F;
    }
}

/*
set every entry to its sign value (do it on site)
keep the result in the input tensor a and return nothing
>> a - the tensor we are processing
*/
void _SignMe(XTensor * a)
{
    _Sign(a, a);
}

/*
set every entry to its sign value (return an XTensor structure)
make a new tensor to keep the result and return it
>> a - input tensor we are processing
<< return - the sign value of the input tensor
*/
XTensor Sign(const XTensor & a)
{
    XTensor b(&a);
    b.SetTMPFlag();

    /* call _Sign function */
    _Sign(&a, &b);

    /* tensor connections */
    XLink::MakeLink(&a, NULL, &b, MATH_SIGN);

    return b;
}
} // namespace nts(NiuTrans.Tensor)