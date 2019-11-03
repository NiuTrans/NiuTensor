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
#include "IsSameShaped.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* 
check whether the two matrices are in the same type and size
>> a - input tensor
>> b - anther tensor to compare with
<< return - whether the two input tensors are identical
*/
bool _IsSameShaped(const XTensor * a, const XTensor * b)
{
    if(a == NULL || b == NULL)
        return false;

    if(a->order != b->order)
        return false;

    for(int i = 0; i < a->order; i++){
        if(a->dimSize[i] != b->dimSize[i])
            return false;
    }

    if(a->dataType != b->dataType)
        return false;

    if(a->denseRatio != b->denseRatio)
        return false;

    if(a->isSparse != b->isSparse)
        return false;

    return true;
}

/*
check whether the two matrices are in the same type and size
>> a - input tensor
>> b - anther tensor to compare with
<< return - whether the two input tensors are identical
*/
bool IsSameShaped(const XTensor & a, const XTensor & b)
{
    return _IsSameShaped(&a, &b);
}

/*
check whether the three matrices are in the same type and size
>> a - input tensor
>> b - anther tensor to compare with
>> c - a tensor again
<< return - whether the two input tensors are identical
*/
bool _IsSameShaped(const XTensor * a, const XTensor * b, const XTensor * c)
{
    return IsSameShaped(a, b) && IsSameShaped(a, c);
}

/*
check whether the three matrices are in the same type and size
>> a - input tensor
>> b - anther tensor to compare with
>> c - a tensor again
<< return - whether the two input tensors are identical
*/
bool IsSameShaped(const XTensor & a, const XTensor & b, const XTensor & c)
{
    return _IsSameShaped(&a, &b, &c);
}

} // namespace nts(NiuTrans.Tensor)