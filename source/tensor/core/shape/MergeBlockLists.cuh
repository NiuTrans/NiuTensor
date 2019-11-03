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

#ifndef __MERGEBLOCKLISTS_CUH__
#define __MERGEBLOCKLISTS_CUH__

#include "MergeBlockLists.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* copy a number of blocks (of different sizes) to target positions */
__global__
void KernelCopyBlockLists(DTYPE ** sourceList, int * sourceBlockSizes, int sourceBlockNum, DTYPE ** targetList);

/* merge data by blocks (cuda version) */
void _CudaMergeBlockLists(const StrList* sourceList, int * blockSizes, int blockNum, void * target, XMem * myMem);

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)

#endif // __MERGEBLOCKLISTS_CUH__

