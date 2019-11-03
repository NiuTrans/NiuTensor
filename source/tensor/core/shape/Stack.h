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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2019-10-13
 * It's so cold outside. It's too hard for me to get out.
 */

#ifndef __STACK_H__
#define __STACK_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* stack small tensors into a big tensor along with a dimension */
void _Stack(const TensorList * smalls, XTensor * t, int dim);

/* stack small tensors into a big tensor along with a dimension (return an XTensor structure) */
XTensor Stack(const TensorList &list, int leadingDim);

/* stack small tensors into a big tensor along with a dimension */
void Stack(const TensorList &smalls, XTensor &t, int dim);

} // namespace nts(NiuTrans.Tensor)

#endif // __STACK_H__