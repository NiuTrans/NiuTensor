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
 * $Created by: LI Yinqiao (li.yin.qiao.2012@hotmail.com) 2018-04-30
 */

#include "../core/utilities/CheckData.h"
#include "TSum.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* case 1: tensor summation c = a + b * \beta */
bool TestSum1()
{
    /* a tensor of size (2, 4) */
    int order = 2;
    int * dimSize = new int[order];
    dimSize[0] = 2;
    dimSize[1] = 4;

    int unitNum = 1;
    for (int i = 0; i < order; i++)
        unitNum *= dimSize[i];

    DTYPE aData[2][4] = { {0.0F, 1.0F, 2.0F, 3.0F},
                          {4.0F, 5.0F, 6.0F, 7.0F} };
    DTYPE bData[2][4] = { {1.0F, -1.0F, -3.0F, -5.0F}, 
                          {-7.0F, -9.0F, -11.0F, -13.0F} };
    DTYPE answer[2][4] = { {1.0F, 0.0F, -1.0F, -2.0F},
                           {-3.0F, -4.0F, -5.0F, -6.0F} };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * a = NewTensorV2(order, dimSize);
    XTensor * b = NewTensorV2(order, dimSize);
    XTensor * c = NewTensorV2(order, dimSize);
    XTensor * cMe = NewTensorV2(order, dimSize);
    XTensor cUser;

    /* initialize variables */
    a->SetData(aData, unitNum);
    cMe->SetData(aData, unitNum);
    b->SetData(bData, unitNum);
    c->SetZeroAll();

    /* call Sum function */
    _Sum(a, b, c);
    _SumMe(cMe, b);
    cUser = Sum(*a, *b);

    /* check results */
    cpuTest = _CheckData(c, answer, unitNum)
              && _CheckData(cMe, answer, unitNum) && _CheckData(&cUser, answer, unitNum);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensor */
    XTensor * aGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * bGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * cGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * cMeGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor cUserGPU;

    /* Initialize variables */
    aGPU->SetData(aData, unitNum);
    cMeGPU->SetData(aData, unitNum);
    bGPU->SetData(bData, unitNum);
    cGPU->SetZeroAll();

    /* call Sum function */
    _Sum(aGPU, bGPU, cGPU);
    _SumMe(cMeGPU, bGPU);
    cUserGPU = Sum(*aGPU, *bGPU);

    /* check results */
    gpuTest = _CheckData(cGPU, answer, unitNum)
              && _CheckData(cMeGPU, answer, unitNum) && _CheckData(&cUserGPU, answer, unitNum);

    /* destroy variables */
    delete a;
    delete b;
    delete c;
    delete cMe;
    delete aGPU;
    delete bGPU;
    delete cGPU;
    delete cMeGPU;
    delete[] dimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete a;
    delete b;
    delete c;
    delete cMe;
    delete[] dimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* case 2: tensor summation c = a + b * \beta */
bool TestSum2()
{
    /* a tensor of size (2, 4) */
    int order = 2;
    int * dimSize = new int[order];
    dimSize[0] = 2;
    dimSize[1] = 4;

    int unitNum = 1;
    for (int i = 0; i < order; i++) {
        unitNum *= dimSize[i];
    }
    DTYPE aData[2][4] = { {0.0F, 1.0F, 2.0F, 3.0F},
                          {4.0F, 5.0F, 6.0F, 7.0F} };
    DTYPE bData[2][4] = { {1.0F, -1.0F, -3.0F, -5.0F}, 
                          {-7.0F, -9.0F, -11.0F, -13.0F} };
    DTYPE answer[2][4] = { {0.5F, 0.5F, 0.5F, 0.5F},
                           {0.5F, 0.5F, 0.5F, 0.5F} };
    float beta = 0.5F;

    /* CPU test */
    bool cpuTest = true;

    /* create tensor */
    XTensor * a = NewTensorV2(order, dimSize);
    XTensor * b = NewTensorV2(order, dimSize);
    XTensor * c = NewTensorV2(order, dimSize);
    XTensor * cMe = NewTensorV2(order, dimSize);
    XTensor cUser;

    /* initialize variables */
    a->SetData(aData, unitNum);
    cMe->SetData(aData, unitNum);
    b->SetData(bData, unitNum);
    c->SetZeroAll();

    /* call Sum function */
    _Sum(a, b, c, beta);
    _SumMe(cMe, b, beta);
    cUser = Sum(*a, *b, beta);

    /* check results */
    cpuTest = _CheckData(c, answer, unitNum)
              && _CheckData(cMe, answer, unitNum) && _CheckData(&cUser, answer, unitNum);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensor */
    XTensor * aGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * bGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * cGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * cMeGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor cUserGPU;

    /* Initialize variables */
    aGPU->SetData(aData, unitNum);
    cMeGPU->SetData(aData, unitNum);
    bGPU->SetData(bData, unitNum);
    cGPU->SetZeroAll();

    /* call Sum function */
    _Sum(aGPU, bGPU, cGPU, beta);
    _SumMe(cMeGPU, bGPU, beta);
    cUserGPU = Sum(*aGPU, *bGPU, beta);

    /* check results */
    gpuTest = _CheckData(cGPU, answer, unitNum)
              && _CheckData(cMeGPU, answer, unitNum) && _CheckData(&cUserGPU, answer, unitNum);

    /* destroy variables */
    delete a;
    delete b;
    delete c;
    delete cMe;
    delete aGPU;
    delete bGPU;
    delete cGPU;
    delete cMeGPU;
    delete[] dimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete a;
    delete b;
    delete c;
    delete cMe;
    delete[] dimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* other cases */
/*
    TODO!!
*/

/* test for Sum Function */
bool TestSum()
{
    XPRINT(0, stdout, "[TEST SUM] tensor summation c = a + b * beta\n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestSum1();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 1 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 1 passed!\n");

    /* case 2 test */
    caseFlag = TestSum2();
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
