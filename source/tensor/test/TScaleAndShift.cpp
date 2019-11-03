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
* $Created by: Xu Chen (email: hello_master1954@163.com) 2018-06-27
*/

#include "../core/utilities/CheckData.h"
#include "TScaleAndShift.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* 
case 1: scale and shift all tensor entires.
p = p * scale + shift
*/
bool TestScaleAndShift1()
{
    /* a input tensor of size (2, 4) */
    int sOrder = 2;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 2;
    sDimSize[1] = 4;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    DTYPE sData[2][4] = { {0.0F, 1.0F, 2.0F, 3.0F},
                          {4.0F, 5.0F, 6.0F, 7.0F} };
    DTYPE answer[2][4] = { {0.5F, 2.5F, 4.5F, 6.5F},
                           {8.5F, 10.5F, 12.5F, 14.5F} };

    DTYPE scaleFactor = 2.0F;
    DTYPE shiftFactor = 0.5F;

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(sOrder, sDimSize);
    XTensor * t = NewTensorV2(sOrder, sDimSize);
    XTensor * tMe = NewTensorV2(sOrder, sDimSize);
    XTensor tUser;

    /* initialize variables */
    s->SetData(sData, sUnitNum);
    tMe->SetData(sData, sUnitNum);

    /* call ScaleAndShift function */
    _ScaleAndShift(s, t, scaleFactor, shiftFactor);
    _ScaleAndShiftMe(tMe, scaleFactor, shiftFactor);
    tUser = ScaleAndShift(*s, scaleFactor, shiftFactor);

    /* check results */
    cpuTest = _CheckData(t, answer, sUnitNum) &&
              _CheckData(tMe, answer, sUnitNum) && _CheckData(&tUser, answer, sUnitNum);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tMeGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor tUserGPU;

    /* initialize variables */
    sGPU->SetData(sData, sUnitNum);
    tMeGPU->SetData(sData, sUnitNum);

    /* call ScaleAndShift function */
    _ScaleAndShift(sGPU, tGPU, scaleFactor, shiftFactor);
    _ScaleAndShiftMe(tMeGPU, scaleFactor, shiftFactor);
    tUserGPU = ScaleAndShift(*sGPU, scaleFactor, shiftFactor);

    /* check results */
    gpuTest = _CheckData(tGPU, answer, sUnitNum) &&
              _CheckData(tMeGPU, answer, sUnitNum) && _CheckData(&tUserGPU, answer, sUnitNum);

    /* destroy variables */
    delete s;
    delete t;
    delete tMe;
    delete sGPU;
    delete tGPU;
    delete tMeGPU;
    delete[] sDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete t;
    delete tMe;
    delete[] sDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* other cases */
/*
TODO!!
*/

/* test for ScaleAndShift Function */
bool TestScaleAndShift()
{
    XPRINT(0, stdout, "[TEST ScaleAndShift] scale and shift all tensor entires\n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestScaleAndShift1();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 1 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 1 passed!\n");

    /* other cases test */
    /*
    TODO!!
    */

    if (returnFlag) {
        XPRINT(0, stdout, ">> All Passed!\n");
    }
    else
        XPRINT(0, stdout, ">> Failed!\n");

    XPRINT(0, stdout, "\n");

    return returnFlag;
    }

} // namespace nts(NiuTrans.Tensor)
