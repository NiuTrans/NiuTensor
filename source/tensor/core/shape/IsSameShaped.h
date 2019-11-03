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

#ifndef __ISSAMESHAPED_H__
#define __ISSAMESHAPED_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* judge whether the two matrices are in the same type and size */
bool _IsSameShaped(const XTensor * a, const XTensor * b);

/* judge whether the two matrices are in the same type and size */
bool IsSameShaped(const XTensor & a, const XTensor & b);

/* judge whether the three matrices are in the same type and size */
bool _IsSameShaped(const XTensor * a, const XTensor * b, const XTensor * c);

/* judge whether the three matrices are in the same type and size */
bool IsSameShaped(const XTensor & a, const XTensor & b, const XTensor & c);

} // namespace nts(NiuTrans.Tensor)

#endif // __ISSAMESHAPED_H__