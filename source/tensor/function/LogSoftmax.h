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
 * $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-04-25
 */

#ifndef __LOGSOFTMAX_H__
#define __LOGSOFTMAX_H__

#include "../XTensor.h"
#include "Loss.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/* log scale softmax y = log(e^x / \sum_{i} e^{x_i}) */
void _LogSoftmax(const XTensor * x, XTensor * y, int leadDim);

/* log scale softmax y = log(e^x / \sum_{i} e^{x_i}) (return an XTensor structure) */
XTensor LogSoftmax(const XTensor &x, int leadDim);

/* log scale softmax y = log(e^x / \sum_{i} e^{x_i}) (with both argument of x and y) */
void LogSoftmax(const XTensor &x, XTensor &y, int leadDim);

/* log scale softmax y = log(e^x / \sum_{i} e^{x_i}) (with both argument of x and y) */
void LogSoftmax(const XTensor &x, XTensor &y, int leadDim);

/* de/dx */
void _LogSoftmaxBackward(XTensor * gold, XTensor * y, XTensor * x, 
                         XTensor * dedy, XTensor * dedx, 
                         XTensor * padding, int leadDim, 
                         LOSS_FUNCTION_NAME lossName);

} // namespace nts(NiuTrans.Tensor)

#endif // __LOGSOFTMAX_H__