/* NiuTrans.Tensor - an open-source tensor library
* Copyright (C) 2017, Natural Language Processing Lab, Northeastern University.
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
 * $Created by: Li Yinqiao (email: li.yin.qiao.2012@hotmail.com) 2020-02-11
 * Paper review rebuttal of ACL2020 will start at this Thursday. So nervous :(
 */

#include "../../XName.h"
#include "../shape/IsSameShaped.h"
#include "Sum.h"
#include "SumDim.h"
#include "../math/ScaleAndShift.h"
#include "Sub.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
tensor subtraction c = a - b * \beta

>> a - a tensor
>> b - another tensor
>> c - where we put a-b*\beta. we save it in a if c is NULL
>> beta - the scaling factor
*/
void _Sub(const XTensor * a, const XTensor * b, XTensor * c, DTYPE beta)
{
    _Sum(a, b, c, -beta);
}
    
/*
tensor subtraction a = a - b * \beta (do it on site)
keep the result in the tensor a and return nothing

>> a - a tensor
>> b - another tensor
>> beta - the scaling factor
*/
void _SubMe(XTensor * a, const XTensor * b, DTYPE beta)
{
    _Sub(a, b, a, beta);
}

/*
tensor subtraction a = a - b * \beta (do it on site)
keep the result in the tensor a and return nothing

>> a - a tensor
>> b - another tensor
>> beta - the scaling factor
*/
void SubMe(XTensor & a, const XTensor & b, DTYPE beta)
{
    if (b.order == 0){
        DTYPE shift = -(b.Get0D() * beta);
        _ScaleAndShift(&a, &a, 1.0F, shift);
    }
    else {
        int n = GetBroadcastDimIndex(a, b);

        if (n == -1)
            /* call _Sub function */
            _Sub(&a, &b, &a, beta);
        else if (n >= 0 && n < a.order)
            /* call _SumDim function to do the SubDim operation */
            _SumDim(&a, &b, &a, n, -beta);
        else
            ShowNTErrors("Something is wrong!");
    }
}

/*
tensor subtraction c = a - b * \beta (return an XTensor structure)
make a new tensor c to keep the result and return it

>> a - a tensor
>> b - another tensor
>> inplace - indicates whether the result will be placed in the input tensor
>> beta - the scaling factor
<< return - the result of tensor subtraction
*/
XTensor Sub(const XTensor & a, const XTensor & b, bool inplace, DTYPE beta)
{
    XTensor c;

    if (inplace) {
        /* the result is stored into the input tensor */
        int dims[MAX_TENSOR_DIM_NUM];
        memcpy(&(dims[0]), &(a.dimSize[0]), sizeof(int) * a.order);
        dims[0] = -dims[0];
        InitTensor(&c, a.order, dims, a.dataType, a.devID, a.enableGrad);
        c.data = a.data;
    }
    else {
        InitTensorV2(&c, &a);
    }
    c.SetTMPFlag();

    if (b.order == 0){
        DTYPE shift = -(b.Get0D() * beta);
        ScaleAndShift(a, c, 1.0F, shift);
    }
    else {
        int n = GetBroadcastDimIndex(a, b);

        if(n == -1){
            /* call _Sub function */
            _Sub(&a, &b, &c, beta);

            /* tensor connections */
            if (a.enableGrad && b.enableGrad) {
                XLink::MakeLink(&a, &b, &c, MATH_SUB);
                XLink::AddParamToHead(&c, beta);
            }
        }
        else if(n >= 0 && n < a.order){
            /* call _SumDim function to do the SubDim operation */
            _SumDim(&a, &b, &c, n, -beta);

            /* tensor connections */
            if (a.enableGrad && b.enableGrad) {
                XLink::MakeLink(&a, &b, &c, MATH_SUBDIM);
                XLink::AddParamToHeadInt(&c, n);
                XLink::AddParamToHead(&c, beta);
            }
        }
        else{
            ShowNTErrors("Something is wrong!");
        }
    }

    XTensor* p = const_cast<XTensor*>(&a);
    if (inplace)
        p->data = NULL;
    return c;
}

/*
tensor subtraction c = a - b * \beta

>> a - a tensor
>> b - another tensor
>> c - where we put a-b*\beta. we save it in a if c is NULL
>> beta - the scaling factor
*/
void Sub(const XTensor & a, const XTensor & b, XTensor & c, DTYPE beta)
{
    if (!c.isInit || !IsSameShaped(a, c)) {
        InitTensorV2(&c, &a);
    }

    if (b.order == 0){
        DTYPE shift = -(b.Get0D() * beta);
        ScaleAndShift(a, c, 1.0F, shift);
    }
    else {
        int n = GetBroadcastDimIndex(a, b);

        if (n == -1) {
            /* call _Sub function */
            _Sub(&a, &b, &c, beta);

            if (a.enableGrad && b.enableGrad) {
                /* tensor connections */
                XLink::MakeLink(&a, &b, &c, MATH_SUB);
                XLink::AddParamToHead(&c, beta);
            }
        }
        else if (n >= 0 && n < a.order) {
            /* call _SumDim function to do the SubDim operation */
            _SumDim(&a, &b, &c, n, -beta);

            if (a.enableGrad && b.enableGrad) {
                /* tensor connections */
                XLink::MakeLink(&a, &b, &c, MATH_SUBDIM);
                XLink::AddParamToHeadInt(&c, n);
                XLink::AddParamToHead(&c, beta);
            }
        }
        else {
            ShowNTErrors("Something is wrong!");
        }
    }
}

} // namespace nts(NiuTrans.Tensor)
