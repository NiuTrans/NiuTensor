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
* $Created by: Xu Chen (email: hello_master1954@163.com) 2018-06-19
*/

#include "../XUtility.h"
#include "../core/utilities/CheckData.h"
#include "TSigmoid.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* 
case 1: test Sigmoid function.
sigmoid function: y = 1/(1+exp(-x))
*/
bool TestSigmoid1()
{
    /* a input tensor of size (3) */
    int order = 1;
    int * dimSize = new int[order];
    dimSize[0] = 3;

    int unitNum = 1;
    for (int i = 0; i < order; i++)
        unitNum *= dimSize[i];

    DTYPE xData[3] = {0.0F, 1.0F, 2.0F};
    DTYPE answer[3] = {0.5F, 0.7311F, 0.8808F};

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * x = NewTensorV2(order, dimSize);
    XTensor * y = NewTensorV2(order, dimSize);
    XTensor yUser;

    /* initialize variables */
    x->SetData(xData, unitNum);
    y->SetZeroAll();

    /* call Sigmoid function */
    _Sigmoid(x, y);
    yUser = Sigmoid(*x);

    /* check result */
	cpuTest = _CheckData(y, answer, unitNum, 1e-4F) &&
              _CheckData(&yUser, answer, unitNum, 1e-4F);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

        /* create tensors */
    XTensor * xGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * yGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor yUserGPU;

    /* initialize variables */
    xGPU->SetData(xData, unitNum);
    yGPU->SetZeroAll();

    /* call Sigmoid function */
    _Sigmoid(xGPU, yGPU);
    yUserGPU = Sigmoid(*xGPU);

    /* check result */
	gpuTest = _CheckData(yGPU, answer, unitNum, 1e-4F) &&
              _CheckData(&yUserGPU, answer, unitNum, 1e-4F);

    /* destroy variables */
    delete x;
    delete y;
    delete xGPU;
    delete yGPU;
    delete[] dimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete x;
    delete y;
    delete[] dimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* 
case 2: test Sigmoid function and SigmoidBackward function.
sigmoid function: y = 1/(1+exp(-x))
backward computation: 
dE/ds = dE/dy * dy/dx
dy/dx = y * (1 - y)
In this case, LossName=CROSSENTROPY.
*/
bool TestSigmoid2()
{
    /* a input tensor of size (3) */
    int order = 1;
    int * dimSize = new int[order];
    dimSize[0] = 3;

    int unitNum = 1;
    for (int i = 0; i < order; i++)
        unitNum *= dimSize[i];

    DTYPE xData[3] = {0.0F, 1.0F, 2.0F};
    DTYPE yAnswer[3] = {0.5F, 0.7311F, 0.8808F};
    DTYPE dedyData[3] = {0.0F, 1.0F, 2.0F};
    DTYPE dedxAnswer[3] = {0.0F, 0.1966F, 0.2100F};

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * x = NewTensorV2(order, dimSize);
    XTensor * y = NewTensorV2(order, dimSize);
    XTensor * dedy = NewTensorV2(order, dimSize);
    XTensor * dedx = NewTensorV2(order, dimSize);

    /* initialize variables */
    x->SetData(xData, unitNum);
    y->SetZeroAll();
    dedx->SetZeroAll();
    dedy->SetData(dedyData, unitNum);

    /* call Sigmoid function */
    _Sigmoid(x, y);

    /* call SigmoidBackward function */
    _SigmoidBackward(y, x, dedy, dedx);

    /* check result */
    cpuTest = _CheckData(y, yAnswer, unitNum, 1e-4F) &&
              _CheckData(dedx, dedxAnswer, unitNum, 1e-4F);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * xGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * yGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * dedyGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * dedxGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);

    /* initialize variables */
    xGPU->SetData(xData, unitNum);
    yGPU->SetZeroAll();
    dedxGPU->SetZeroAll();
    dedyGPU->SetData(dedyData, unitNum);

    /* call Sigmoid function */
    _Sigmoid(xGPU, yGPU);

    /* call SigmoidBackward function */
    _SigmoidBackward(yGPU, xGPU, dedyGPU, dedxGPU);
    
    /* check result */
    gpuTest = _CheckData(yGPU, yAnswer, unitNum, 1e-4F) &&
              _CheckData(dedxGPU, dedxAnswer, unitNum, 1e-4F);

    /* destroy variables */
    delete x;
    delete y;
    delete dedx;
    delete dedy;
    delete xGPU;
    delete yGPU;
    delete dedxGPU;
    delete dedyGPU;
    delete[] dimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete x;
    delete y;
    delete dedx;
    delete dedy;
    delete[] dimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* other cases */
/*
    TODO!!
*/

/* test for Sigmoid Function */
bool TestSigmoid()
{
    XPRINT(0, stdout, "[TEST SIGMOID] sigmoid function and its backward computation \n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestSigmoid1();

    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 1 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 1 passed!\n");
    
    /* case 2 test */
    caseFlag = TestSigmoid2();

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
