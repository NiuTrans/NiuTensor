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
* $Created by: LI Yinqiao (email: li.yin.qiao.2012@hotmail.com) 2018-04-30
*/

#include "../core/utilities/CheckData.h"
#include "TReduceMean.h"

namespace nts { // namespace nt(NiuTrans.Tensor)

/* case 1: get the mean value along a dimension of the tensor */
bool TestReduceMean1()
{
    /* a tensor of size (2, 4) */
    int sOrder = 2;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 2;
    sDimSize[1] = 4;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    /* a tensor of size (4) */
    int tOrder1 = 1;
    int * tDimSize1 = new int[tOrder1];
    tDimSize1[0] = 4;

    int tUnitNum1 = 1;
    for (int i = 0; i < tOrder1; i++)
        tUnitNum1 *= tDimSize1[i];

    /* a tensor of size (2) */
    int tOrder2 = 1;
    int * tDimSize2 = new int[tOrder2];
    tDimSize2[0] = 2;

    int tUnitNum2 = 1;
    for (int i = 0; i < tOrder2; i++)
        tUnitNum2 *= tDimSize2[i];

    DTYPE sData[2][4] = { {0.0F, 1.0F, 2.0F, 3.0F},
                          {4.0F, 5.0F, 6.0F, 7.0F} };
    DTYPE answer1[4] = {2.0F, 3.0F, 4.0F, 5.0F};
    DTYPE answer2[2] = {1.5F, 5.5F};

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(sOrder, sDimSize);
    XTensor * t1 = NewTensorV2(tOrder1, tDimSize1);
    XTensor * t2 = NewTensorV2(tOrder2, tDimSize2);
    XTensor tUser1;
    XTensor tUser2;

    /* initialize variables */
    s->SetData(sData, sUnitNum);
    t1->SetZeroAll();
    t2->SetZeroAll();

    /* call ReduceMean function */
    _ReduceMean(s, t1, 0);
    _ReduceMean(s, t2, 1);
    tUser1 = ReduceMean(*s, 0);
    tUser2 = ReduceMean(*s, 1);

    /* check results */
    cpuTest = _CheckData(t1, answer1, tUnitNum1) && _CheckData(&tUser1, answer1, tUnitNum1)
        && _CheckData(t2, answer2, tUnitNum2) && _CheckData(&tUser2, answer2, tUnitNum2);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensor */
    XTensor * sGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU1 = NewTensorV2(tOrder1, tDimSize1, X_FLOAT, 1.0F, 0);
    XTensor * tGPU2 = NewTensorV2(tOrder2, tDimSize2, X_FLOAT, 1.0F, 0);
    XTensor tUserGPU1;
    XTensor tUserGPU2;

    /* Initialize variables */
    sGPU->SetData(sData, sUnitNum);
    tGPU1->SetZeroAll();
    tGPU2->SetZeroAll();

    /* call ReduceMean function */
    _ReduceMean(sGPU, tGPU1, 0);
    _ReduceMean(sGPU, tGPU2, 1);
    tUserGPU1 = ReduceMean(*sGPU, 0);
    tUserGPU2 = ReduceMean(*sGPU, 1);

    /* check results */
    gpuTest = _CheckData(tGPU1, answer1, tUnitNum1) && _CheckData(&tUserGPU1, answer1, tUnitNum1)
        && _CheckData(tGPU2, answer2, tUnitNum2) && _CheckData(&tUserGPU2, answer2, tUnitNum2);

    /* destroy variables */
    delete s;
    delete t1;
    delete t2;
    delete sGPU;
    delete tGPU1;
    delete tGPU2;
    delete[] sDimSize;
    delete[] tDimSize1;
    delete[] tDimSize2;
    
    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete t1;
    delete t2;
    delete[] sDimSize;
    delete[] tDimSize1;
    delete[] tDimSize2;

    return cpuTest;
#endif // USE_CUDA
}

/* other cases */
/*
TODO!!
*/

/* test for ReduceMean Function */
bool TestReduceMean()
{
    XPRINT(0, stdout, "[TEST ReduceMean] get the mean value along a dimension of the tensor \n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestReduceMean1();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 1 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 1 passed!\n");

    ///* other cases test */
    ///*
    //TODO!!
    //*/

    if (returnFlag) {
        XPRINT(0, stdout, ">> All Passed!\n");
    }
    else
        XPRINT(0, stdout, ">> Failed!\n");

    XPRINT(0, stdout, "\n");

    return returnFlag;
}

} // namespace nts(NiuTrans.Tensor)
