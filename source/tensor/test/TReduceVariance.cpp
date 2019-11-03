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
#include "TReduceVariance.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
case 1: variance of the items along a dimension of the tensor. 
For a 1-dimensional data array a, variance = 1/n * \sum_i (a_i - mean)^2.
In this case, (2, 4) -> (4), dim = 0.
*/
bool TestReduceVariance1()
{
    /* a input tensor of size (2, 4) */
    int sOrder = 2;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 2;
    sDimSize[1] = 4;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    /* a output tensor of size (4) */
    int tOrder = 1;
    int * tDimSize = new int[tOrder];
    tDimSize[0] = 4;

    int tUnitNum = 1;
    for (int i = 0; i < tOrder; i++)
        tUnitNum *= tDimSize[i];

    /* a mean tensor of size (4) */
    int meanOrder = 1;
    int * meanDimSize = new int[meanOrder];
    meanDimSize[0] = 4;

    int meanUnitNum = 1;
    for (int i = 0; i < meanOrder; i++)
        meanUnitNum *= meanDimSize[i];

    DTYPE sData[2][4] = { {0.0F, 1.0F, 2.0F, 3.0F},
                          {4.0F, 5.0F, 6.0F, 7.0F} };
    DTYPE meanData[4] = {2.0F, 3.0F, 4.0F, 5.0F};
    DTYPE answer[4] = {4.0F, 4.0F, 4.0F, 4.0F};

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(sOrder, sDimSize);
    XTensor * t = NewTensorV2(tOrder, tDimSize);
    XTensor * mean = NewTensorV2(meanOrder, meanDimSize);
    XTensor tUser;

    /* initialize variables */
    s->SetData(sData, sUnitNum);
    mean->SetData(meanData, meanUnitNum);
    t->SetZeroAll();

    /* call ReduceVariance function */
    _ReduceVariance(s, t, 0, mean);
    tUser = ReduceVariance(*s, 0, *mean);

    /* check results */
    cpuTest = _CheckData(t, answer, tUnitNum) && _CheckData(&tUser, answer, tUnitNum);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU = NewTensorV2(tOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor * meanGPU = NewTensorV2(meanOrder, meanDimSize, X_FLOAT, 1.0F, 0);
    XTensor tUserGPU;

    /* initialize variables */
    sGPU->SetData(sData, sUnitNum);
    meanGPU->SetData(meanData, meanUnitNum);
    tGPU->SetZeroAll();

    /* call ReduceVariance function */
    _ReduceVariance(sGPU, tGPU, 0, meanGPU);
    tUserGPU = ReduceVariance(*sGPU, 0, *meanGPU);

    /* check results */
    gpuTest = _CheckData(tGPU, answer, tUnitNum) && _CheckData(&tUserGPU, answer, tUnitNum);

    /* destroy variables */
    delete s;
    delete t;
    delete mean;
    delete sGPU;
    delete tGPU;
    delete meanGPU;
    delete[] sDimSize;
    delete[] tDimSize;
    delete[] meanDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete t;
    delete mean;
    delete[] sDimSize;
    delete[] tDimSize;
    delete[] meanDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* other cases */
/*
TODO!!
*/

/* test for ReduceVariance Function */
bool TestReduceVariance()
{
    XPRINT(0, stdout, "[TEST ReduceVariance] variance of the items along a dimension of the tensor\n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestReduceVariance1();
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
