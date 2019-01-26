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

#include <math.h>
#include "../../XTensor.h"
#include "../../XName.h"
#include "Power.h"
#include "Power.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
get the power(a, p)
>> a - input tensor
>> b - output tensor
>> p - parameter
*/
void _Power(const XTensor * a, XTensor * b, DTYPE p)
{
#ifdef USE_CUDA
    /* run it on GPUs */
    if (a->devID >= 0) {
        _CudaPower(a, b, p);
        return;
    }
#endif

    CheckNTErrors((a->dataType == DEFAULT_DTYPE), "TODO!");

    DTYPE * aData = (DTYPE*)a->data;
    DTYPE * bData = (DTYPE*)b->data;
    if (p == 0) {
        for (int i = 0; i < a->unitNum; i++)
            bData[i] = (DTYPE)1.0;
    }
    else if (p == (DTYPE)0.5) {
        for (int i = 0; i < a->unitNum; i++)
            bData[i] = (DTYPE)sqrt(aData[i]);
    }
    else if (p == (DTYPE)2.0) {
        for (int i = 0; i < a->unitNum; i++)
            bData[i] = aData[i] * aData[i];
    }
    else {
        for (int i = 0; i < a->unitNum; i++) {
            if (p < 0 && aData[i] == 0)
                bData[i] = 1e20F;
            else
                bData[i] = (DTYPE)pow(aData[i], p);
        }
    }
}

/*
get the power(a, p) (do it on site)
keep the result in the input tensor a and return nothing
>> a - the tensor
>> p - parameter
*/
void _PowerMe(XTensor * a, DTYPE p)
{
    _Power(a, a, p);
}

/*
get the power(a, p) (return an XTensor structure)
make a new tensor to keep the result and return it
>> a - input tensor
>> p - parameter
<< return - the power value of the input tensor
*/
XTensor Power(const XTensor & a, DTYPE p)
{
    XTensor b(&a);
    b.SetTMPFlag();
    
    /* call _Power function */
    _Power(&a, &b, p);
    
    /* tensor connections */
    XLink::MakeLink(&a, NULL, &b, MATH_POWER);
    XLink::AddParamToHead(&b, p);

    return b;
}

} // namespace nts(NiuTrans.Tensor)
