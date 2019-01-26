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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-09-25
 */

#ifndef __SPREAD_CUH__
#define __SPREAD_CUH__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* spread a collection tensor to source tensor (cuda version) */
void _CudaSpread(XTensor * source, XTensor * collection, int dim, 
                 int * srcIndex, int indexSize, int * collIndex);

/* special spread function for backward computation of CopyIndexed function (cuda version) */
void _CudaSpreadForCopyIndexed(XTensor * s, XTensor * c, int dim, 
                               XTensor * srcIndex, XTensor * collIndex, 
                               int copyNum);

/* special spread function for backward computation of gather function (cuda version) */
void _CudaSpreadForGather(XTensor * source, XTensor * collection, XTensor * srcIndex);

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)

#endif // __SPREAD_CUH__