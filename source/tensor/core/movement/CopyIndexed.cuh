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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-11-30
 * Tomorrow is the celebration of the laboratory, I'm so happy!
 */

#ifndef __CopyIndexed_CUH__
#define __CopyIndexed_CUH__

#include "../../XTensor.h"
#include "CopyIndexed.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* copy selected sub-tensors where indeces are kept in tensors (cuda version) */
void _CudaCopyIndexed(const XTensor * s, XTensor * t, int dim, 
                      const XTensor * srcIndex, const XTensor * tgtIndex, 
                      int copyNum);

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)

#endif // __CopyIndexed_CUH__