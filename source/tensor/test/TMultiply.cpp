/* NiuTrans.Tensor - an open-source tensor library
* Copyright (C) 2017, Natural Language Processing Lab, Northeastern University.
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
* $Created by: Lin Ye (email: linye2015@outlook.com) 2018-06-15
*/

#include "../core/utilities/CheckData.h"
#include "TMultiply.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* 
case 1: element-wise product of two tensors
c(i) = a(i)*b(i) + \alpha * c(i)
In this case, (2, 2) * (2, 2) -> (2, 2), leadingDim=0, alpha=0.
*/
bool TestMultiply1()
{
    /* a source tensor of size (2, 2) */
    int sOrder1 = 2;
    int * sDimSize1 = new int[sOrder1];
    sDimSize1[0] = 2;
    sDimSize1[1] = 2;

    int sUnitNum1 = 1;
    for (int i = 0; i < sOrder1; i++)
        sUnitNum1 *= sDimSize1[i];

    /* a source tensor of size (2, 2) */
    int sOrder2 = 2;
    int * sDimSize2 = new int[sOrder2];
    sDimSize2[0] = 2;
    sDimSize2[1] = 2;

    int sUnitNum2 = 1;
    for (int i = 0; i < sOrder2; i++)
        sUnitNum2 *= sDimSize2[i];

    /* a target tensor of size (2, 2) */
    int tOrder = 2;
    int * tDimSize = new int[tOrder];
    tDimSize[0] = 2;
    tDimSize[1] = 2;

    int tUnitNum = 1;
    for (int i = 0; i < tOrder; i++)
        tUnitNum *= tDimSize[i];

    DTYPE sData1[2][2] = { {0.0F, 1.0F},
                           {2.0F, 3.0F} };
    DTYPE sData2[2][2] = { {0.0F, 1.0F},
                           {2.0F, 3.0F} };
    DTYPE answer[2][2] = { {0.0F, 1.0F},
                           {4.0F, 9.0F} };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s1 = NewTensorV2(sOrder1, sDimSize1);
    XTensor * s2 = NewTensorV2(sOrder2, sDimSize2);
    XTensor * t = NewTensorV2(tOrder, tDimSize);
    XTensor * tMe = NewTensorV2(tOrder, tDimSize);
    XTensor tUser;

    /* initialize variables */
    s1->SetData(sData1, sUnitNum1);
    tMe->SetData(sData1, sUnitNum1);
    s2->SetData(sData2, sUnitNum2);
    t->SetZeroAll();

    /* call Multiply function */
    _Multiply(s1, s2, t, 0, 0);
    _MultiplyMe(tMe, s2, 0, 0);
    tUser = Multiply(*s1, *s2, false, 0);

    /* check results */
    cpuTest = _CheckData(t, answer, tUnitNum, 1e-4F) &&
              _CheckData(tMe, answer, tUnitNum, 1e-4F) &&
              _CheckData(&tUser, answer, tUnitNum, 1e-4F);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensor */
    XTensor * sGPU1 = NewTensorV2(sOrder1, sDimSize1, X_FLOAT, 1.0F, 0);
    XTensor * sGPU2 = NewTensorV2(sOrder2, sDimSize2, X_FLOAT, 1.0F, 0);
    XTensor * tGPU = NewTensorV2(tOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tMeGPU = NewTensorV2(tOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor tUserGPU;

    /* Initialize variables */
    sGPU1->SetData(sData1, sUnitNum1);
    tMeGPU->SetData(sData1, sUnitNum1);
    sGPU2->SetData(sData2, sUnitNum2);
    tGPU->SetZeroAll();

    /* call Multiply function */
    _Multiply(sGPU1, sGPU2, tGPU, 0, 0);
    _MultiplyMe(tMeGPU, sGPU2, 0, 0);
    tUserGPU = Multiply(*sGPU1, *sGPU2, 0);

    /* check results */
    gpuTest = _CheckData(tGPU, answer, tUnitNum, 1e-4F) &&
              _CheckData(tMeGPU, answer, tUnitNum, 1e-4F) &&
              _CheckData(&tUserGPU, answer, tUnitNum, 1e-4F);

    /* destroy variables */
    delete s1;
    delete s2;
    delete t;
    delete tMe;
    delete sGPU1;
    delete sGPU2;
    delete tGPU;
    delete tMeGPU;
    delete[] sDimSize1;
    delete[] sDimSize2;
    delete[] tDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s1;
    delete s2;
    delete t;
    delete tMe;
    delete[] sDimSize1;
    delete[] sDimSize2;
    delete[] tDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/*
case 2: element-wise product of two tensors
c(i) = a(i)*b(i) + \alpha * c(i)
In this case, (2, 3, 4) * (2, 1, 1) -> (2, 3, 4), alpha=0.
*/
bool TestMultiply2()
{
    /* a source tensor of size (2, 3, 4) */
    int sOrder1 = 3;
    int * sDimSize1 = new int[sOrder1];
    sDimSize1[0] = 2;
    sDimSize1[1] = 3;
    sDimSize1[2] = 4;

    int sUnitNum1 = 1;
    for (int i = 0; i < sOrder1; i++)
        sUnitNum1 *= sDimSize1[i];

    /* a source tensor of size (2, 1, 1) */
    int sOrder2 = 3;
    int * sDimSize2 = new int[sOrder2];
    sDimSize2[0] = 2;
    sDimSize2[1] = 1;
    sDimSize2[2] = 1;

    int sUnitNum2 = 1;
    for (int i = 0; i < sOrder2; i++)
        sUnitNum2 *= sDimSize2[i];

    /* a target tensor of size (2, 3, 4) */
    int tOrder = 3;
    int * tDimSize = new int[tOrder];
    tDimSize[0] = 2;
    tDimSize[1] = 3;
    tDimSize[2] = 4;

    int tUnitNum = 1;
    for (int i = 0; i < tOrder; i++)
        tUnitNum *= tDimSize[i];

    DTYPE sData1[2][3][4] = { { {0.0F, 1.0F, 2.0F, 3.0F},
                                {3.0F, 2.0F, 1.0F, 0.0F},
                                {0.0F, 1.0F, 2.0F, 3.0F} },
                              { {3.0F, 2.0F, 1.0F, 0.0F},
                                {0.0F, 1.0F, 2.0F, 3.0F},
                                {3.0F, 2.0F, 1.0F, 0.0F} } };
    DTYPE sData2[2][1][1] = { { {1.0F} },
                              { {-1.0F} } };
    DTYPE answer[2][3][4] = { { {0.0F, 1.0F, 2.0F, 3.0F},
                                {3.0F, 2.0F, 1.0F, 0.0F},
                                {0.0F, 1.0F, 2.0F, 3.0F} },
                              { {-3.0F, -2.0F, -1.0F, 0.0F},
                                {0.0F, -1.0F, -2.0F, -3.0F},
                                {-3.0F, -2.0F, -1.0F, 0.0F} } };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s1 = NewTensorV2(sOrder1, sDimSize1);
    XTensor * s2 = NewTensorV2(sOrder2, sDimSize2);
    XTensor * tMe = NewTensorV2(tOrder, tDimSize);
    XTensor tUser;

    /* initialize variables */
    s1->SetData(sData1, sUnitNum1);
    tMe->SetData(sData1, sUnitNum1);
    s2->SetData(sData2, sUnitNum2);

    /* call Multiply function */
    MultiplyMe(*tMe, *s2, 0);
    tUser = Multiply(*s1, *s2);

    /* check results */
    cpuTest = _CheckData(tMe, answer, tUnitNum, 1e-4F) &&
              _CheckData(&tUser, answer, tUnitNum, 1e-4F);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensor */
    XTensor * sGPU1 = NewTensorV2(sOrder1, sDimSize1, X_FLOAT, 1.0F, 0);
    XTensor * sGPU2 = NewTensorV2(sOrder2, sDimSize2, X_FLOAT, 1.0F, 0);
    XTensor * tMeGPU = NewTensorV2(tOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor tUserGPU;

    /* Initialize variables */
    sGPU1->SetData(sData1, sUnitNum1);
    tMeGPU->SetData(sData1, sUnitNum1);
    sGPU2->SetData(sData2, sUnitNum2);

    /* call Multiply function */
    MultiplyMe(*tMeGPU, *sGPU2, 0);
    tUserGPU = Multiply(*sGPU1, *sGPU2);

    /* check results */
    gpuTest = _CheckData(tMeGPU, answer, tUnitNum, 1e-4F) &&
              _CheckData(&tUserGPU, answer, tUnitNum, 1e-4F);

    /* destroy variables */
    delete s1;
    delete s2;
    delete tMe;
    delete sGPU1;
    delete sGPU2;
    delete tMeGPU;
    delete[] sDimSize1;
    delete[] sDimSize2;
    delete[] tDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s1;
    delete s2;
    delete tMe;
    delete[] sDimSize1;
    delete[] sDimSize2;
    delete[] tDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* other cases */
/*
TODO!!
*/

/* test for Multiply Function */
bool TestMultiply()
{
    XPRINT(0, stdout, "[TEST Multiply] element-wise product of two tensors \n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestMultiply1();

    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 1 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 1 passed!\n");

    /* case 2 test */
    caseFlag = TestMultiply2();

    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 2 failed!\n");
    }
    else
    XPRINT(0, stdout, ">> case 2 passed!\n");

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
