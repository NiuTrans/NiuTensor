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

#ifndef __MERGE_H__
#define __MERGE_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* transform a tensor by merging it alone with a dimension, e.g., (M, N/3, 3) -> (M, N) */
void _Merge(const XTensor * s, XTensor * t, int whereToMerge, int leadingDim = -1);

/* transform a tensor by merging it alone with a dimension (return an XTensor structure)
   e.g., (M, N/3, 3) -> (M, N) */
XTensor Merge(const XTensor &s, int whereToMerge, int leadingDim = -1);

void Merge(const XTensor &s, XTensor &t, int whereToMerge, int leadingDim = -1);

/* merge small tensors into a big tensor */
void _Merge(const TensorList * smalls, XTensor * t, int whereToMerge);

/* merge small tensors into a big tensor (return an XTensor structure) */
XTensor Merge(const TensorList &smalls, int whereToMerge);

void Merge(const TensorList &smalls, XTensor &t, int whereToMerge);

/* merge two tensors into a big tensor (return an XTensor structure) */
XTensor Merge(const XTensor &smallA, const XTensor &smallB, int whereToMerge);

} // namespace nts(NiuTrans.Tensor)

#endif // __MERGE_H__