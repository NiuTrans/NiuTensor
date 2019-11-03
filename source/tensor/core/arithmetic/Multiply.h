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

#ifndef __MULTIPLY_H__
#define __MULTIPLY_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* 
element-wise product of two tensors:
c(i) = a(i)*b(i) + \alpha * c(i) 
where i is the index of the element
*/
void _Multiply(const XTensor * a, const XTensor * b, XTensor * c, DTYPE alpha = 0.0, int leadingDim = 0);

/* 
element-wise product of two tensors (do it on site)
keep the result in the input tensor a and return nothing
a(i) = a(i)*b(i) + \alpha * a(i) 
where i is the index of the element 
*/
void _MultiplyMe(XTensor * a, const XTensor * b, DTYPE alpha = 0.0, int leadingDim = 0);
void MultiplyMe(XTensor & a, const XTensor & b, DTYPE alpha = 0.0, int leadingDim = 0);

/* 
element-wise product of two tensors (return an XTensor structure)
make a new tensor to keep the result and return it
c(i) = a(i)*b(i)
where i is the index of the element 
*/
XTensor Multiply(const XTensor &a, const XTensor &b, DTYPE alpha = 0.0, int leadingDim = 0);

/* 
element-wise product of two tensors:
c(i) = a(i)*b(i) + \alpha * c(i) 
where i is the index of the element
*/
void Multiply(const XTensor &a, const XTensor &b, XTensor &c, DTYPE alpha = 0.0, int leadingDim = 0);

} // namespace nts(NiuTrans.Tensor)

#endif // __MULTIPLY_H__