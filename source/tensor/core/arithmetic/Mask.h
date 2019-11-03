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
 * $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2019-04-24
 * I'll attend several conferences and workshops in the following weeks -
 * busy days :(
 */

#ifndef __MASK_H__
#define __MASK_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* 
mask entries of a given tensor:
c(i) = a(i) if mask(i) is non-zero
c(i) = alpha if mask(i) = 0
where i is the index of the element
*/
void _Mask(const XTensor * a, const XTensor * mask, XTensor * c, DTYPE alpha = 0.0);

/* 
mask entries of a given tensor (on site):
a(i) = a(i) if mask(i) is non-zero
a(i) = alpha if mask(i) = 0
where i is the index of the element
*/
void _MaskMe(XTensor * a, const XTensor * mask, DTYPE alpha = 0.0);
void MaskMe(XTensor & a, const XTensor & mask, DTYPE alpha = 0.0);

/*
mask entries of a given tensor (return an XTensor structure):
a(i) = a(i) if mask(i) is non-zero
a(i) = alpha if mask(i) = 0
where i is the index of the element
*/
XTensor Mask(const XTensor &a, const XTensor &mask, DTYPE alpha = 0.0);

/*
mask entries of a given tensor (return an XTensor structure):
a(i) = a(i) if mask(i) is non-zero
a(i) = alpha if mask(i) = 0
where i is the index of the element
*/
void Mask(const XTensor &a, const XTensor &mask, XTensor &c, DTYPE alpha = 0.0);

} // namespace nts(NiuTrans.Tensor)

#endif // __MASK_H__
