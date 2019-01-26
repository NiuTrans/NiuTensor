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
#include "Negate.h"
#include "Negate.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
set every entry to its minus value
>> a - input tensor we are processing
>> b - output tensor we are processing
*/
void _Negate(const XTensor * a, XTensor * b)
{
#ifdef USE_CUDA
    /* run it on GPUs */
    if (a->devID >= 0) {
        _CudaNegate(a, b);
    return;
    }
#endif

    CheckNTErrors((XTensor::IsSameShaped(a, b)), "Input tensors should have the same type!");
    CheckNTErrors((a->dataType == DEFAULT_DTYPE), "TODO!");
    DTYPE * d = (DTYPE*)a->data;
    DTYPE * db = (DTYPE*)b->data;
    for (int i = 0; i < a->unitNum; i++)
        db[i] = -d[i];
}

/*
set every entry to its minus value (do it on site)
keep the result in the input tensor a and return nothing
>> a - the tensor we are processing
*/
void _NegateMe(XTensor * a)
{
    _Negate(a, a);
}

/*
set every entry to its minus value (return an XTensor structure)
make a new tensor to keep the result and return it
>> a - input tensor we are processing
<< return - the minus value of input tensor
*/
XTensor Negate(const XTensor & a)
{
    XTensor b(&a);
    b.SetTMPFlag();
    
    /* call _Negate function */
    _Negate(&a, &b);
    
    /* tensor connections */
    XLink::MakeLink(&a, NULL, &b, MATH_NEGATE);
    
    return b;
}

} // namespace nts(NiuTrans.Tensor)