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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-12-17
 */

#ifndef __ONEHOTANDINDEX_H__
#define __ONEHOTANDINDEX_H__

#include "../../XTensor.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/* convert onehot tensor to index tensor */
void _OnehotToIndex(const XTensor * onehot, XTensor * index, int size);

/* convert onehot tensor to index tensor (return an XTensor structure)
make a new tensor to keep the result and return it */
XTensor OnehotToIndex(const XTensor & onehot, int num);

/* convert index tensor to onehot tensor */
void _IndexToOnehot(const XTensor * index, XTensor * onehot, int size, float labelSmoothingP);

/* convert index tensor to onehot tensor */
void _IndexToOnehot(int * index, int n, XTensor * onehot, int size, float labelSmoothingP);

/* convert index tensor to onehot tensor (return an XTensor structure)
make a new tensor to keep the result and return it */
XTensor IndexToOnehot(const XTensor & index, int num, float labelSmoothingP);

} // namespace nts(NiuTrans.Tensor)

#endif // __ONEHOTANDINDEX_H__