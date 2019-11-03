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
 * $Created by: Lin Ye (email: linye2015@outlook.com) 2018-08-03
 */

#ifndef __CLIP_H__
#define __CLIP_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* set every entry to its clip value */
void _Clip(const XTensor * a, XTensor * b, DTYPE lower, DTYPE upper);

/* set every entry to its clip value (do it on site)
   keep the result in the input tensor a and return nothing */
void _ClipMe(XTensor * a, DTYPE lower, DTYPE upper);

/* set every entry to its clip value (do it on site)
keep the result in the input tensor a and return nothing */
void ClipMe(XTensor & a, DTYPE lower, DTYPE upper);

/* set every entry to its clip value  (return an XTensor structure)
   make a new tensor to keep the result and return it */
XTensor Clip(const XTensor & a, DTYPE lower, DTYPE upper);

void Clip(const XTensor & a, XTensor & b, DTYPE lower, DTYPE upper);

/*
backward of Clip function
*/
void _ClipBackward(XTensor * y, XTensor * x, XTensor * dedy, XTensor * dedx, DTYPE lower, DTYPE upper);

} // namespace nts(NiuTrans.Tensor)

#endif // __CLIP_H__
