/* NiuTrans.Tensor - an open-source tensor library
* Copyright (C) 2018, Natural Language Processing Lab, Northestern University.
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
* $Created by: Lin Ye (email: linye2015@outlook.com) 2018-08-13
*/

#ifndef __SUBDIM_H__
#define __SUBDIM_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* tensor subtraction c = a - b * \beta where the size of b is equal to the n-th dimension of a, 
   i.e., a is subtracted with b by broadcasting*/
void _SubDim(const XTensor * a, const XTensor * b, XTensor * c, int n, DTYPE beta = (DTYPE)1.0);

/* tensor subtraction c = a - b * \beta where the size of b is equal to the n-th dimension of a, 
   i.e., a is subtracted with b by broadcasting. we keep the result in the input tensor a and return nothing */
void _SubDim(XTensor * a, const XTensor * b, int n, DTYPE beta = (DTYPE)1.0);

/* tensor subtraction c = a - b * \beta where the size of b is equal to the n-th dimension of a,
   i.e., a is subtracted with b by broadcasting. We make a new tensor c to keep the result and return it */
XTensor SubDim(const XTensor &a, const XTensor &b, int n, DTYPE beta = (DTYPE)1.0);

/* tensor subtraction c = a - b * \beta where the size of b is equal to the n-th dimension of a, 
   i.e., a is subtracted with b by broadcasting*/
void SubDim(const XTensor &a, const XTensor &b, XTensor &c, int n, DTYPE beta = (DTYPE)1.0);

} // namespace nts(NiuTrans.Tensor)

#endif // __SUBDIM_H__
