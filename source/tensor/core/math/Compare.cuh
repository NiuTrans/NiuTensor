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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-12-10
 */

#ifndef __COMPARE_CUH__
#define __COMPARE_CUH__

#include "../../XTensor.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* check whether every entry is equal to the given value (cuda version) */
void _CudaEqual(const XTensor * a, XTensor * b, DTYPE value);

/* check whether every entry is not equal to the given value (cuda version) */
void _CudaNotEqual(const XTensor * a, XTensor * b, DTYPE value);

/* return maximum of two tensor for each items (cuda version) */
void _CudaMax(const XTensor * a, const XTensor * b, XTensor *c);

/* return minimum of two tensor for each items (cuda version) */
void _CudaMin(const XTensor * a, const XTensor * b, XTensor *c);

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)

#endif //end __COMPARE_CUH__