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

#ifndef __CUDAMAKEMERGEBLOCKINDEX_CUH__
#define __CUDAMAKEMERGEBLOCKINDEX_CUH__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* set target data block index for the data movement in split */
void _CudaMakeMergeBlockIndex(int devID, int * blockIndex, int blockNum, int blockNumInMerge,
                              int splitSizeInGrid, int gridSize, int gridNum);

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)

#endif // __CUDAMAKEMERGEBLOCKINDEX_CUH__