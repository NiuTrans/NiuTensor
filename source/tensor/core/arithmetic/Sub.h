/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2018, Natural Language Processing Lab, Northeastern University.
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
 * Today is the first day of August. It's still very hot.
 */

#ifndef __SUB_H__
#define __SUB_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* tensor subtraction c = a - b * \beta */
void _Sub(const XTensor * a, const XTensor * b, XTensor * c, DTYPE beta = (DTYPE)1.0);

/* 
tensor subtraction a = a - b * \beta
keep the result in the input tensor a and return nothing
*/
void _SubMe(XTensor * a, const XTensor * b, DTYPE beta = (DTYPE)1.0);
void SubMe(XTensor & a, const XTensor & b, DTYPE beta = (DTYPE)1.0);
    
/*
tensor subtraction c = a - b * \beta
make a new tensor c to keep the result and return it
*/
XTensor Sub(const XTensor &a, const XTensor &b, bool inplace = false, DTYPE beta = (DTYPE)1.0);

/* tensor subtraction c = a - b * \beta */
void Sub(const XTensor &a, const XTensor &b, XTensor &c, DTYPE beta = (DTYPE)1.0);

} // namespace nts(NiuTrans.Tensor)

#endif // __SUB_H__
