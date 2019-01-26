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

#ifndef __LOSS_CUH__
#define __LOSS_CUH__

#include "../XTensor.h"
#include "Loss.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* compute the loss (cuda version) */
DTYPE _CudaLossCompute(XTensor * gold, XTensor * output, LOSS_FUNCTION_NAME LFName,
                      bool isLogOutput, int leadDim, int gBeg, int gLen, int oBeg);

/* compute the loss in log scale (cuda version) */
DTYPE _CudaLossComputeForLogScale(XTensor * gold, XTensor * output, LOSS_FUNCTION_NAME LFName,
                                 int leadDim, int gBeg, int gLen, int oBeg);

/* backward compuation for a single element (cuda version) */
DTYPE _CudaLossBackwardPoint(DTYPE t, DTYPE y, LOSS_FUNCTION_NAME LFName);

/* backward compuation for (dense) vectors (cuda version) */
void _CudaLossBackward(XTensor * dedy, XTensor * t, XTensor * y, 
                      LOSS_FUNCTION_NAME LFName, 
                      int leadDim = -1, int tBeg = 0, int tLen = -1, int yBeg = 0);

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)

#endif // __RECTIFY_CUH__