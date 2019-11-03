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

#ifndef __COPYVALUES_H__
#define __COPYVALUES_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* copy s to t */
void _CopyValues(const XTensor * s, XTensor * t, XStream * stream = NULL);

/* copy a segment of s to t  */
void _CopyValues(const XTensor * s, const int sBeg, const int sLen, XTensor * t, const int tBeg, XStream * stream = NULL);

/* copy s to t (rename _CopyValues) */
void CopyValues(const XTensor &s, XTensor &t, XStream * stream = NULL);

/* 
copy s to t (return an XTensor structure)
make a new tensor to keep the result and return it
*/
XTensor CopyValues(const XTensor &s, XStream * stream = NULL);

} // namespace nts(NiuTrans.Tensor)

#endif // __COPYVALUES_H__
