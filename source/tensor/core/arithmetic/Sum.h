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
 * $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-04-24
 */

#ifndef __SUM_H__
#define __SUM_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* tensor summation c = a + b * \beta */
void _Sum(const XTensor * a, const XTensor * b, XTensor * c, DTYPE beta = (DTYPE)1.0);

/* 
tensor summation a = a + b * \beta
keep the result in the input tensor a and return nothing
*/
void _SumMe(XTensor * a, const XTensor * b, DTYPE beta = (DTYPE)1.0);
void SumMe(XTensor & a, const XTensor & b, DTYPE beta = (DTYPE)1.0);
    
/*
tensor summation c = a + b * \beta
make a new tensor c to keep the result and return it
*/
XTensor Sum(const XTensor &a, const XTensor &b, DTYPE beta = (DTYPE)1.0);

/* tensor summation c = a + b * \beta */
void Sum(const XTensor &a, const XTensor &b, XTensor &c, DTYPE beta = (DTYPE)1.0);

} // namespace nts(NiuTrans.Tensor)

#endif // __SUM_H__
