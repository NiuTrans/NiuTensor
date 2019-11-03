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
 * $Created by: JIANG Yufan (email: jiangyufan2018@outlook.com) 2018-08-14
 */

#ifndef __MULTIPLYDIM_H__
#define __MULTIPLYDIM_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* tensor multiplication c = a * b + \alpha * c  where the size of b is equal to the n-th dimension of a,
   i.e., a is multiplied with b by broadcasting */
void _MultiplyDim(const XTensor * a, const XTensor * b, XTensor * c, int n, DTYPE alpha = 0.0);

/* tensor multiplication a = a * b + \alpha * c where the size of b is equal to the n-th dimension of a,
   i.e., a is multiplied with b by broadcasting. we keep the result in the input tensor a and return nothing */
void _MultiplyDimMe(XTensor * a, const XTensor * b, int n, DTYPE alpha = 0.0);
void MultiplyDimMe(XTensor & a, const XTensor & b, int n, DTYPE alpha = 0.0);

/* tensor multiplication c = a * b where the size of b is equal to the n-th dimension of a,
   i.e., a is multiplied with b by broadcasting. We make a new tensor c to keep the result and return it */
XTensor MultiplyDim(const XTensor &a, const XTensor &b, int n);

/* tensor multiplication c = a * b + \alpha * c  where the size of b is equal to the n-th dimension of a,
   i.e., a is multiplied with b by broadcasting */
void MultiplyDim(const XTensor &a, const XTensor &b, XTensor &c, int n);

/* tensor multiplication summation c = a * b + c * \beta where some of dimensions of b can be of size 1 */
void _MultiplyBroadcast(const XTensor * a, const XTensor * b, XTensor * c, DTYPE beta = (DTYPE)1.0);

/* tensor broadcast multiplication c = a * b where some of dimensions of b can be of size 1.
   we return the resulting tensor here */
XTensor MultiplyBroadcast(const XTensor &a, const XTensor &b);

/* tensor multiplication summation c = a * b + c * \beta where some of dimensions of b can be of size 1 */
void MultiplyBroadcast(const XTensor &a, const XTensor &b, XTensor &c);

} // namespace nts(NiuTrans.Tensor)

#endif // __MULTIPLYDIM_H__
