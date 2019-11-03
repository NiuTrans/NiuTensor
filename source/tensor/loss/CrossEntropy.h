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

#ifndef __CROSSENTROPY_H__
#define __CROSSENTROPY_H__

#include "../XTensor.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

enum LOSS_COMPUTE_WAY{
REDUCE_SUM,
REDUCE_MEAN
};

/* compute the cross entropy loss */
void _CrossEntropy(const XTensor * output, const XTensor * gold, 
                   XTensor * loss, const XTensor * weight = NULL, 
                   const XTensor * padding = NULL, int leadingDim = -1);

/* compute the cross entropy loss */
void _CrossEntropyFast(const XTensor * output, const XTensor * gold,
                         XTensor * loss, const XTensor * weight = NULL, 
                         const XTensor * padding = NULL, int leadingDim = -1);

/* compute the cross entropy loss */
XTensor CrossEntropy(const XTensor & output, const XTensor & gold, 
                     int leadingDim = -1);

/* compute the cross entropy loss with padding */
XTensor CrossEntropy(const XTensor & output, const XTensor & gold,
                     const XTensor & padding,
                     int leadingDim = -1);

/* compute the cross entropy loss with weight */
XTensor CrossEntropyWeight(const XTensor & output, const XTensor & gold,
                           const XTensor & weight,
                           int leadingDim = -1);

/* compute the cross entropy loss with weight and padding */
XTensor CrossEntropyWeight(const XTensor & output, const XTensor & gold,
                           const XTensor & padding, const XTensor & weight,
                           int leadingDim = -1);

/* compute the cross entropy loss (return the loss) */
DTYPE _CrossEntropy(const XTensor * output, const XTensor * gold,
                    LOSS_COMPUTE_WAY reduceWay, const XTensor * weight = NULL, 
                    const XTensor * padding = NULL, int leadingDim = -1);

/* compute the cross entropy loss (return the loss) */
DTYPE _CrossEntropyFast(const XTensor * output, const XTensor * gold,
                        LOSS_COMPUTE_WAY reduceWay = REDUCE_MEAN, const XTensor * weight = NULL,
                        const XTensor * padding = NULL, int leadingDim = -1);

/* backward computation of cross entropy function */
void _CrossEntropyBackward(XTensor * dedy, const XTensor * output, 
                           const XTensor * gold, const XTensor * weight = NULL, 
                           XTensor * padding = NULL, int leadingDim = -1);

} // namespace nts(NiuTrans.Tensor)

#endif // __CROSSENTROPY_H__