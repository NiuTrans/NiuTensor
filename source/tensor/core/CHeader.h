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

/* this is a header to include all functions in the "core" workspace */

#ifndef __CHEADER_H__
#define __CHEADER_H__

#include "../XTensor.h"

#include "arithmetic/Div.h"
#include "arithmetic/DivDim.h"
#include "arithmetic/Mask.h"
#include "arithmetic/MatrixMul.h"
#include "arithmetic/MatrixMul2D.h"
#include "arithmetic/MatrixMul2DMultiTheading.h"
#include "arithmetic/MatrixMul2DParallel.h"
#include "arithmetic/MatrixMulBatched.h"
#include "arithmetic/Multiply.h"
#include "arithmetic/MultiplyDim.h"
#include "arithmetic/Sub.h"
#include "arithmetic/SubDim.h"
#include "arithmetic/Sum.h"
#include "arithmetic/SumDim.h"
#include "arithmetic/XTensorBLAS.h"
#include "arithmetic/MulAndShift.h"

#include "getandset/ConvertDataType.h"
#include "getandset/OnehotAndIndex.h"
#include "getandset/Select.h"
#include "getandset/SetData.h"

#include "math/Binary.h"
#include "math/Clip.h"
#include "math/Compare.h"
#include "math/Normalize.h"
#include "math/ScaleAndShift.h"
#include "math/Unary.h"

#include "movement/CopyBlocks.h"
#include "movement/CopyBlocksInGrid.h"
#include "movement/CopyBlocksOnSite.h"
#include "movement/CopyData2D.h"
#include "movement/CopyIndexed.h"
#include "movement/CopyInGrid.h"
#include "movement/CopyValues.h"
#include "movement/Gather.h"
#include "movement/Spread.h"

#include "reduce/ReduceMax.h"
#include "reduce/ReduceMean.h"
#include "reduce/ReduceStandardVariance.h"
#include "reduce/ReduceSum.h"
#include "reduce/ReduceSumAll.h"
#include "reduce/ReduceSumSquared.h"
#include "reduce/ReduceVariance.h"

#include "shape/Concatenate.h"
#include "shape/ConcatenateSolely.h"
#include "shape/MakeMergeBlockIndex.h"
#include "shape/MakeSplitBlockIndex.h"
#include "shape/Merge.h"
#include "shape/MergeBlockLists.h"
#include "shape/Reshape.h"
#include "shape/Permute.h"
#include "shape/Split.h"
#include "shape/Squeeze.h"
#include "shape/Stack.h"
#include "shape/Transpose.h"
#include "shape/Unsqueeze.h"
#include "shape/IsSameShaped.h"

#include "sort/Sort.h"
#include "sort/TopK.h"

#include "utilities/XMatrixSegment.h"
#include "utilities/FlushToMem.h"
#include "utilities/CheckData.h"
#include "utilities/SetAscendingOrder.h"

#endif // __CHEADER_H__
