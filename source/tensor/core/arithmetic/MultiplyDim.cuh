/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2018, Natural Language Processing Lab, Northestern University.
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
 * $Created by: JIANG Yufan (email: jiangyufan2018@outlook.com) 2018-08-14
 */

#ifndef __MULTIPLYDIM_CUH__
#define __MULTIPLYDIM_CUH__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* tensor summation a * b + \alpha * c where the size of b is equal to the n-th dimension of a,
   i.e., a is multiplied with b by broadcasting (cuda version) */
void _CudaMultiplyDim(const XTensor * a, const XTensor * b, XTensor * c, int n, DTYPE alpha = 0);

#endif

} // namespace nts(NiuTrans.Tensor)

#endif // __MULTIPLYDIM_CUH__
