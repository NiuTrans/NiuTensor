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
* $Created by: LI Yinqiao (email: li.yin.qiao.2012@hotmail.com) 2019-10-22
*/

#include "../../XTensor.h"
#include "../../XUtility.h"
#include "CheckData.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* compare two numbers */
bool IsFloatEqual(DTYPE a, DTYPE b, float absError, float relError)
{
    if(a == b)
        return true;
    if(fabs(a - b) < absError)
        return true;
    if(fabs(a) < fabs(b))
        return (fabs((a - b) / b) < relError) ? true : false;
    else
        return (fabs((a - b) / a) < relError) ? true : false;
}

/* check whether the data array is the same as the answer
>> tensor - input tensor
>> d - input data (it must be on CPUs)
>> num - number of data items
>> beg - where we start this in the data array of the tensor
*/
bool _CheckData(const XTensor * tensor, const void * d, int num, int beg)
{
    if (tensor->data == NULL || d == NULL)
        return false;

    CheckNTErrors(!tensor->isSparse, "TODO");
    CheckNTErrors(num == tensor->unitNum - beg, "Illegal size!");

    if (tensor->devID < 0) {
        return !memcmp(tensor->data, d, num * tensor->unitSize);
    }
#ifdef USE_CUDA
    else {
        char * copy = new char[num * tensor->unitSize];
        XMemCopy(copy, -1, tensor->data, tensor->devID, num * tensor->unitSize);
        int cmpResult = memcmp(copy, d, num * tensor->unitSize);
        bool result = (cmpResult == 0) ? true : false;
        delete[] copy;
        return result;
    }
#endif
    return true;
}

/* check whether the data array is the same as the answer
>> tensor - input tensor
>> d - input data (it must be on CPUs)
>> num - number of data items
>> tolerance - error value we tolerant between result and answer
>> beg - where we start this in the data array of the tensor
*/bool _CheckData(const XTensor * tensor, const void * d, int num, float tolerance, int beg)
{
    if (tensor->data == NULL || d == NULL)
        return false;

    CheckNTErrors(!tensor->isSparse, "TODO");
    CheckNTErrors(num == tensor->unitNum - beg, "Illegal size!");

    DTYPE * valuePrt = (DTYPE*)tensor->data;
    DTYPE value = 0;
    DTYPE * answerPrt = (DTYPE*)d;
    for (int i = beg; i < num; i++) {
        value = ToCPU(tensor->devID, valuePrt);
        if(IsFloatEqual(value, *answerPrt, tolerance, 1e-4F) == false)
            return false;
        valuePrt++;
        answerPrt++;
    }
    return true;
}

} // namespace nts(NiuTrans.Tensor)