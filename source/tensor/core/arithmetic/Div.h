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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-08-01
 */

#ifndef __DIV_H__
#define __DIV_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* 
element-wise division of two tensors:
c(i) = a(i)/b(i) + \alpha * c(i) 
where i is the index of the element
*/
void _Div(const XTensor * a, const XTensor * b, XTensor * c, DTYPE alpha = 0.0, int leadingDim = 0);

/* 
element-wise division of two tensors (do it on site)
keep the result in the input tensor a and return nothing
a(i) = a(i)/b(i) + \alpha * a(i) 
where i is the index of the element 
*/
void _DivMe(XTensor * a, const XTensor * b, DTYPE alpha = 0.0, int leadingDim = 0);
void DivMe(XTensor & a, const XTensor & b, DTYPE alpha = 0.0, int leadingDim = 0);

/* 
element-wise division of two tensors (return an XTensor structure)
make a new tensor to keep the result and return it
c(i) = a(i)/b(i)
where i is the index of the element 
*/
XTensor Div(const XTensor &a, const XTensor &b, DTYPE alpha = 0.0, int leadingDim = 0);

/*
element-wise division of two tensors:
c(i) = a(i)/b(i) + \alpha * c(i)
where i is the index of the element
*/
void Div(const XTensor &a, const XTensor &b, XTensor &c, DTYPE alpha = 0.0, int leadingDim = 0);

} // namespace nts(NiuTrans.Tensor)

#endif // __DIV_H__