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
* Good start of the new project - but full of meetings today.
*/

#ifndef __LOSS_H__
#define __LOSS_H__

#include "../XTensor.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/*
loss function name, e.g., crossentropy.
*/
enum LOSS_FUNCTION_NAME {
NOLOSS, 
SQUAREDERROR,  // loss = sum_{i} 0.5*(t_i - y_i)^2, where t_i is the gold standard and y_i is the model output
               // dloss/dy_i = y_i - t_i
               // it is actually a squared euclidean distance
CROSSENTROPY,  // loss = sum_{i} (-t_i * log(y_i)), where t and y are distributions 
               // dloss/dy_i = -t_i / y_i
ONEHOTERROR    // loss = sum_{i} e_i
               // where e_i = 0.5*(t_i - y_i)^2 if t_i = 1, 
               // e_i = 0 otherwise
};

/*
loss function to measure the "number" of errors
*/

/* compute the loss */
DTYPE _LossCompute(XTensor * gold, XTensor * output, LOSS_FUNCTION_NAME LFName,
                   bool isLogOutput, int leadDim, int gBeg, int gLen, int oBeg);

/* compute the loss (log version) */
DTYPE _LossComputeForLogScale(XTensor * gold, XTensor * output, LOSS_FUNCTION_NAME LFName,
                              int leadDim, int gBeg, int gLen, int oBeg);

/* backward compuation for a single element */
DTYPE _LossBackwardPoint(DTYPE t, DTYPE y, LOSS_FUNCTION_NAME LFName);

/* backward compuation for (dense) vectors */
void _LossBackward(XTensor * dEdY, XTensor * t, XTensor * y, 
                   LOSS_FUNCTION_NAME LFName, 
                   int leadDim = -1, int tBeg = 0, int tLen = -1, int yBeg = 0);

} // namespace nts(NiuTrans.Tensor)

#endif // __LOSS_H__
