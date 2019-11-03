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
* $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-04-27
*/

#ifndef __IDENTITY_H__
#define __IDENTITY_H__

#include "../XTensor.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/* identity function y = x */
void _Identity(const XTensor * x, XTensor * y);

/* identity function y = x (return an XTensor structure) */
XTensor Identity(const XTensor &x);

void Identity(const XTensor &x, XTensor &y);

/* de/dx */
void _IdentityBackward(const XTensor * y, const XTensor * x, 
                       const XTensor * dedy, XTensor * dedx);

} // namespace nts(NiuTrans.Tensor)

#endif // __IDENTITY_H__