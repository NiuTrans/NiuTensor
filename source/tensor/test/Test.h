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
* $Created by: LI Yinqiao (li.yin.qiao.2012@hotmail.com) 2018-05-01
*/

#ifndef __TEST_H__
#define __TEST_H__

#include "TAbsolute.h"
#include "TClip.h"
#include "TCompare.h"
#include "TConcatenate.h"
#include "TConcatenateSolely.h"
#include "TCos.h"
#include "TConvertDataType.h"
#include "TCopyIndexed.h"
#include "TCopyValues.h"
#include "TDiv.h"
#include "TDivDim.h"
#include "TExp.h"
#include "TGather.h"
#include "TLog.h"
#include "TMatrixMul.h"
#include "TMatrixMul2D.h"
#include "TMatrixMul2DParallel.h"
#include "TMatrixMulBatched.h"
#include "TMerge.h"
#include "TMultiply.h"
#include "TMultiplyDim.h"
#include "TNegate.h"
#include "TNormalize.h"
#include "TPower.h"
#include "TReduceMax.h"
#include "TReduceMean.h"
#include "TReduceSum.h"
#include "TReduceSumAll.h"
#include "TReduceSumSquared.h"
#include "TReduceVariance.h"
#include "TRound.h"
#include "TScaleAndShift.h"
#include "TSelect.h"
#include "TSetAscendingOrder.h"
#include "TSetData.h"
#include "TSign.h"
#include "TSin.h"
#include "TSort.h"
#include "TSplit.h"
#include "TSpread.h"
#include "TSub.h"
#include "TSum.h"
#include "TSumDim.h"
#include "TTan.h"
#include "TTranspose.h"
#include "TTopK.h"
#include "TUnsqueeze.h"
#include "TXMem.h"

#include "TCrossEntropy.h"
#include "TDropout.h"
#include "THardTanH.h"
#include "TIdentity.h"
#include "TLogSoftmax.h"
#include "TLoss.h"
#include "TRectify.h"
#include "TSigmoid.h"
#include "TSoftmax.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* test for all Function */
bool Test();

} // namespace nts(NiuTrans.Tensor)
#endif // __TEST_H__
