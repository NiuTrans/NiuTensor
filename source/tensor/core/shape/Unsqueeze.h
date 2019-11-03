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

#ifndef __UNSQUEEZE_H__
#define __UNSQUEEZE_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* insert a dimension by copying the blocks for x times 
  (where x is the size of the inerted dimension) */
void _Unsqueeze(const XTensor * a, XTensor * b, int dim, int dSize);

/* insert a dimension by copying the blocks for x times 
  (where x is the size of the inerted dimension) (return an XTensor structure)
   make a new tensor to keep the result and return it */
XTensor Unsqueeze(const XTensor &a, int dim, int dSize);

void Unsqueeze(const XTensor &a, XTensor &b, int dim, int dSize);

} // namespace nts(NiuTrans.Tensor)

#endif // __UNSQUEEZE_H__
