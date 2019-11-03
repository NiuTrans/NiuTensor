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
* $Created by: Lin Ye (email: linye2015@outlook.com) 2018-06-14
*/

#include "../core/utilities/CheckData.h"
#include "TRectify.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* 
case 1: test rectify function
In this case, y = max(0, x) 
*/
bool TestRectify1()
{
    /* a tensor of size (2, 3) */
    int order = 2;
    int * dimSize = new int[order];
    dimSize[0] = 2;
    dimSize[1] = 3;

    int unitNum = 1;
    for (int i = 0; i < order; i++)
        unitNum *= dimSize[i];

    DTYPE xData[2][3] = { {0.0F, -1.0F, 2.0F},
                          {3.0F, -4.0F, -5.0F} };
    DTYPE answer[2][3] = { {0.0F, 0.0F, 2.0F},
                           {3.0F, 0.0F, 0.0F} };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * x = NewTensorV2(order, dimSize);
    XTensor * y = NewTensorV2(order, dimSize);
    XTensor yUser;

    /* initialize variables */
    x->SetData(xData, unitNum);
    y->SetZeroAll();

    /* call Rectify function */
    _Rectify(x, y);
    yUser = Rectify(*x);

    /* check results */
    cpuTest = _CheckData(y, answer, unitNum, 1e-4F) && _CheckData(&yUser, answer, unitNum, 1e-4F);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensor */
    XTensor * xGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * yGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor yUserGPU;

    /* Initialize variables */
    xGPU->SetData(xData, unitNum);
    yGPU->SetZeroAll();

    /* call Rectify function */
    _Rectify(xGPU, yGPU);
    yUserGPU = Rectify(*xGPU);

    /* check results */
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
case 2: backward computation 
dE/dx = dE/dy * dy/dx 
rectified: y = max(0, x) 
In this case, lossName=CROSSENTROPY.
*/
bool TestRectify2()
{
    /* a tensor of size (2, 3) */
    int order = 2;
    int * dimSize = new int[order];
    dimSize[0] = 2;
    dimSize[1] = 3;

    int unitNum = 1;
    for (int i = 0; i < order; i++)
        unitNum *= dimSize[i];

	DTYPE xData[2][3] = { {-1.0F, 1.0F, 2.0F},
	                      {-2.0F, 4.0F, 5.0F} };
    DTYPE yData[2][3] = { {0.0F, 1.0F, 2.0F},
	                      {0.0F, 4.0F, 5.0F} };
	DTYPE dedyData[2][3] = { {-0.5F, -0.5F, -0.25F},
	                         {-0.25F, -0.125F, -0.1F} };
	DTYPE dedxAnswer[2][3] = { {0.0F, -0.5F, -0.25F},
	                           {0.0F, -0.125F, -0.1F} };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * x = NewTensorV2(order, dimSize);
    XTensor * y = NewTensorV2(order, dimSize);
    XTensor * dedy = NewTensorV2(order, dimSize);
    XTensor * dedx = NewTensorV2(order, dimSize);

    /* initialize variables */
    x->SetData(xData, unitNum);
	y->SetData(yData, unitNum);
	dedy->SetData(dedyData, unitNum);
    dedx->SetZeroAll();

    /* call Rectify function */

    /* call RectifyBackward function */
	_RectifyBackward(y, x, dedy, dedx);

    /* check results */
    cpuTest = _CheckData(dedx, dedxAnswer, unitNum, 1e-4F);

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
	yGPU->SetData(yData, unitNum);
	dedyGPU->SetData(dedyData, unitNum);
    dedxGPU->SetZeroAll();
    
    /* call Rectify function */

    /* call rectifybackward function */
	_RectifyBackward(yGPU, xGPU, dedyGPU, dedxGPU);
    
    /* check results */
    gpuTest = _CheckData(dedxGPU, dedxAnswer, unitNum, 1e-4F);

    /* destroy variables */
    delete x;
    delete y;
    delete dedy;
    delete dedx;
    delete xGPU;
    delete yGPU;
    delete dedyGPU;
    delete dedxGPU;
    delete[] dimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete x;
    delete y;
    delete dedy;
    delete dedx;
	delete[] dimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* other cases */
/*
TODO!!
*/

/* test for Rectify Function */
bool TestRectify()
{
    XPRINT(0, stdout, "[TEST RECTIFY] rectify function and its backward computation \n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestRectify1();

    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 1 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 1 passed!\n");

    /* case 2 test */
    caseFlag = TestRectify2();

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
