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

#include "Test.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* test for all Function */
bool Test()
{
    bool wrong = false;
    XPRINT(0, stdout, "Testing the XTensor utilites ... \n\n");
    
    wrong = !TestAbsolute() || wrong;
    wrong = !TestClip() || wrong;
    wrong = !TestCompare() || wrong;
    wrong = !TestConcatenate() || wrong;
    wrong = !TestConcatenateSolely() || wrong;
    wrong = !TestCos() || wrong;
    //wrong = !TestConvertDataType() || wrong;
    wrong = !TestCopyIndexed() || wrong;
    wrong = !TestCopyValues() || wrong;
    wrong = !TestDiv() || wrong;
    wrong = !TestDivDim() || wrong;
    wrong = !TestExp() || wrong;
    wrong = !TestGather() || wrong;
    wrong = !TestLog() || wrong;
    wrong = !TestMatrixMul() || wrong;
    wrong = !TestMatrixMul2D() || wrong;
    wrong = !TestMatrixMul2DParallel() || wrong;
    wrong = !TestMatrixMulBatched() || wrong;
    wrong = !TestMerge() || wrong;
    wrong = !TestMultiply() || wrong;
    wrong = !TestMultiplyDim() || wrong;
    wrong = !TestNegate() || wrong;
    wrong = !TestNormalize() || wrong;
    wrong = !TestPower() || wrong;
    wrong = !TestReduceMax() || wrong;
    wrong = !TestReduceMean() || wrong;
    wrong = !TestReduceSum() || wrong;
    wrong = !TestReduceSumAll() || wrong;
    wrong = !TestReduceSumSquared() || wrong;
    wrong = !TestReduceVariance() || wrong;
    wrong = !TestRound() || wrong;
    wrong = !TestScaleAndShift() || wrong;
    wrong = !TestSelect() || wrong;
    wrong = !TestSetAscendingOrder() || wrong;
    wrong = !TestSetData() || wrong;
    wrong = !TestSign() || wrong;
    wrong = !TestSin() || wrong;
    wrong = !TestSort() || wrong;
    wrong = !TestSplit() || wrong;
    wrong = !TestSpread() || wrong;
    wrong = !TestSub() || wrong;
    wrong = !TestSum() || wrong;
    wrong = !TestSumDim() || wrong;
    wrong = !TestTan() || wrong;
    wrong = !TestTranspose() || wrong;
    wrong = !TestTopK() || wrong;
    wrong = !TestUnsqueeze() || wrong;
    wrong = !TestXMem() || wrong;
    
    wrong = !TestCrossEntropy() || wrong;
    wrong = !TestDropout() || wrong;
    wrong = !TestHardTanH() || wrong;
    wrong = !TestIdentity() || wrong;
    wrong = !TestLogSoftmax() || wrong;
    wrong = !TestLoss() || wrong;
    wrong = !TestRectify() || wrong;
    wrong = !TestSigmoid() || wrong;
    wrong = !TestSoftmax() || wrong;

    /* other test */
    /*
    TODO!!
    */

    if (wrong) {
        XPRINT(0, stdout, "Something goes wrong! Please check the code!\n");
        return false;
    }
    else {
        XPRINT(0, stdout, "OK! Everything is good!\n");
        return true;
    }
}

} // namespace nts(NiuTrans.Tensor)