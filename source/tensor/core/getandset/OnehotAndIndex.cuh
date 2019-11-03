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

#ifndef __ONEHOTANDINDEX_CUH__
#define __ONEHOTANDINDEX_CUH__

#include "../../XTensor.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/* convert onehot tensor to index tensor (cuda version) */
void _CudaOnehotToIndex(const XTensor * onehot, XTensor * index, int size);

/* convert index tensor to onehot tensor (cuda version) */
void _CudaIndexToOnehot(const XTensor * index, XTensor * onehot, 
                        int size, float confidence, float lowconfidence);

} // namespace nts(NiuTrans.Tensor)

#endif // __ONEHOTANDINDEX_CUH__