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

#include "../XTensor.h"
#include "../XUtility.h"
#include "../core/utilities/CheckData.h"
#include "TSoftmax.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* 
case 1: test Softmax function.
softmax function: y = e^x / \sum_{i} e^{x_i}
*/
bool TestSoftmax1()
{
    /* a tensor of size (2, 3) */
    int order = 2;
    int * dimSize = new int[order];
    dimSize[0] = 2;
    dimSize[1] = 3;

    int unitNum = 1;
    for (int i = 0; i < order; i++)
        unitNum *= dimSize[i];

    DTYPE xData[2][3] = { {0.0F, 1.0F, 2.0F}, 
                          {0.5F, 0.7F, 1.4F} };
    DTYPE answer[2][3] = { {0.0900F, 0.2447F, 0.6652F}, 
                           {0.2136F, 0.2609F, 0.5254F} };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * x = NewTensorV2(order, dimSize);
    XTensor * y = NewTensorV2(order, dimSize);
    XTensor yUser;

    /* initialize variables */
    x->SetData(xData, unitNum);
    y->SetZeroAll();

    /* call Softmax function */
    _Softmax(x, y, 1);
    yUser = Softmax(*x, 1);
    
    /* check result */
    cpuTest = _CheckData(y, answer, unitNum, 1e-4F) && _CheckData(&yUser, answer, unitNum, 1e-4F);

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

    /* call Softmax function */
    _Softmax(xGPU, yGPU, 1);
    yUserGPU = Softmax(*xGPU, 1);
    
    /* check result */
    gpuTest = _CheckData(yGPU, answer, unitNum, 1e-4F) && _CheckData(&yUserGPU, answer, unitNum, 1e-4F);

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
case 2: test SoftmaxBackward function.
SoftmaxBackward function: dE/dx_j = -gold_j + y_j
In this case, LossName=CROSSENTROPY.
*/
bool TestSoftmax2()
{
    /* a input tensor of size (2, 3) */
    int order = 2;
    int * dimSize = new int[order];
    dimSize[0] = 1;
    dimSize[1] = 3;

    int unitNum = 1;
    for (int i = 0; i < order; i++)
        unitNum *= dimSize[i];

    DTYPE xData[1][3] = { {0.0F, 1.0F, 2.0F} };
    DTYPE gData[1][3] = { {0.0F, 0.0F, 1.0F} };
    DTYPE yAnswer[1][3] = { {0.0900F, 0.2447F, 0.6652F} };
    DTYPE dedxAnswer[1][3] = {0.0900F, 0.2447F, -0.3347F};

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * x = NewTensorV2(order, dimSize);
    XTensor * y = NewTensorV2(order, dimSize);
    XTensor * g = NewTensorV2(order, dimSize);
    XTensor * dedy = NewTensorV2(order, dimSize);
    XTensor * dedx = NewTensorV2(order, dimSize);

    /* initialize variables */
    x->SetData(xData, unitNum);
    g->SetData(gData, unitNum);
    y->SetZeroAll();
    dedx->SetZeroAll();
    dedy->SetZeroAll();

    /* call Softmax function */
    _Softmax(x, y, 1);
    
    /* call SoftmaxBackward function */
    _SoftmaxBackward(g, y, x, dedy, dedx, NULL, 1, CROSSENTROPY);
    
    /* check result */
    cpuTest = _CheckData(y, yAnswer, unitNum, 1e-4F)
              && _CheckData(dedx, dedxAnswer, unitNum, 1e-4F);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

        /* create tensors */
    XTensor * xGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * yGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * gGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * dedyGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * dedxGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);

    /* initialize variables */
    xGPU->SetData(xData, unitNum);
    gGPU->SetData(gData, unitNum);
    yGPU->SetZeroAll();
    dedxGPU->SetZeroAll();
    dedyGPU->SetZeroAll();

    /* call Softmax function */
    _Softmax(xGPU, yGPU, 1);

    /* call SoftmaxBackward function */
    _SoftmaxBackward(gGPU, yGPU, xGPU, dedyGPU, dedxGPU, NULL, 1, CROSSENTROPY);
    
    /* check result */
    gpuTest = _CheckData(yGPU, yAnswer, unitNum, 1e-4F)
              && _CheckData(dedxGPU, dedxAnswer, unitNum, 1e-4F);

    /* destroy variables */
    delete x;
    delete y;
    delete g;
    delete dedx;
    delete dedy;
    delete xGPU;
    delete yGPU;
    delete gGPU;
    delete dedxGPU;
    delete dedyGPU;
    delete[] dimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete x;
    delete y;
    delete g;
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

/* test for Softmax Function */
bool TestSoftmax()
{
    XPRINT(0, stdout, "[TEST SOFTMAX] softmax function and its backward computation \n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestSoftmax1();

    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 1 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 1 passed!\n");

    /* case 2 test */
    caseFlag = TestSoftmax2();

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
