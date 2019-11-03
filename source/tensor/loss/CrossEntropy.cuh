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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-09-17
 */

#ifndef __CROSSENTROPY_CUH__
#define __CROSSENTROPY_CUH__

#include "../XTensor.h"
#include "CrossEntropy.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/* compute the cross entropy loss */
void _CudaCrossEntropyFast(const XTensor * output, const XTensor * gold,
                           XTensor * loss, const XTensor * weight = NULL, 
                           const XTensor * padding = NULL, int leadingDim = -1);

/* compute the cross entropy loss */
DTYPE _CudaCrossEntropyFast(const XTensor * output, const XTensor * gold,
                            LOSS_COMPUTE_WAY reduceWay, const XTensor * weight = NULL, 
                            const XTensor * padding = NULL, int leadingDim = -1);

/* backward computation of cross entropy function */
void _CudaCrossEntropyBackward(XTensor * dedy, const XTensor * output, 
                               const XTensor * gold, const XTensor * weight = NULL, 
                               XTensor * padding = NULL, int leadingDim = -1);


} // namespace nts(NiuTrans.Tensor)

#endif // __CROSSENTROPY_CUH__