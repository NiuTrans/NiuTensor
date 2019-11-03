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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-09-12
 */

#include "../XUtility.h"
#include "../core/getandset/SetData.h"
#include "TDropout.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* 
case 1: test Dropout function.
*/
bool TestDropout1()
{
    /* a input tensor of size (4, 5) */
    int order = 3;
    int * dimSize = new int[order];
    dimSize[0] = 40;
    dimSize[1] = 50;
    dimSize[2] = 60;

    int unitNum = 1;
    for (int i = 0; i < order; i++)
        unitNum *= dimSize[i];

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * x = NewTensorV2(order, dimSize);
    XTensor * y = NewTensorV2(order, dimSize);
    XTensor yUser;

    /* initialize variables */
    _SetDataFixedFloat(x, 1.0F);
    y->SetZeroAll();

    /* call Dropout function */
    float dropProb = 0.2F;
    int seed = 20;
    _Dropout(x, y, seed, dropProb);
    yUser = Dropout(*x, dropProb);

    /* check result */
    int zeroNum1 = 0;
    int zeroNum2 = 0;
    float * data1 = (float*)y->data;
    float * data2 = (float*)yUser.data;
    for (int i = 0; i < unitNum; i++){
        DTYPE tmp1 = data1[i];
        DTYPE tmp2 = data2[i];
        if(tmp1 == 0.0F)
            zeroNum1 += 1;
        if(tmp2 == 0.0F)
            zeroNum2 += 1;
    }
    printf("CPU Test:\n");
    printf("In tensor y, there are %d units.\n", unitNum);
    printf("There are %d zero units by Dropout layer with probability %.2f.\n", zeroNum1, dropProb);
    printf("In tensor yUser, there are %d units.\n", unitNum);
    printf("There are %d zero units by Dropout layer with default probability %.2f.\n", zeroNum2, dropProb);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * xGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * yGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor yUserGPU;

    /* initialize variables */
    _SetDataFixedFloat(xGPU, 1.0F);
    yGPU->SetZeroAll();

    /* call Dropout function */
    _Dropout(xGPU, yGPU, seed, dropProb);
    yUserGPU = Dropout(*xGPU, dropProb);

    /* check result */
    zeroNum1 = 0;
    zeroNum2 = 0;
    data1 = (float*)y->data;
    data2 = (float*)yUser.data;
    for (int i = 0; i < unitNum; i++){
        DTYPE tmp1 = data1[i];
        DTYPE tmp2 = data2[i];
        if(tmp1 == 0.0F)
            zeroNum1 += 1;
        if(tmp2 == 0.0F)
            zeroNum2 += 1;
    }
    printf("CPU Test:\n");
    printf("In tensor y, there are %d units.\n", unitNum);
    printf("There are %d zero units by Dropout layer with probability %.2f.\n", zeroNum1, dropProb);
    printf("In tensor yUser, there are %d units.\n", unitNum);
    printf("There are %d zero units by Dropout layer with default probability %.2f.\n", zeroNum2, dropProb);

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
case 2: test Dropout function and backward computation.
*/
bool TestDropout2()
{
    /* a input tensor of size (4, 5) */
    int order = 2;
    int * dimSize = new int[order];
    dimSize[0] = 4;
    dimSize[1] = 5;

    int unitNum = 1;
    for (int i = 0; i < order; i++)
        unitNum *= dimSize[i];

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * x = NewTensorV2(order, dimSize);
    XTensor * y = NewTensorV2(order, dimSize);
    XTensor * dedx = NewTensorV2(order, dimSize);
    XTensor * dedy = NewTensorV2(order, dimSize);

    /* initialize variables */
    _SetDataFixedFloat(x, 1.0F);
    y->SetZeroAll();
    dedx->SetZeroAll();
    _SetDataFixedFloat(dedy, 1.5F);

    /* call Dropout function */
    float dropProb = 0.5F;
    int seed = 1;
    _Dropout(x, y, seed, dropProb);
    _DropoutBackward(y, x, dedy, dedx, 1, dropProb);

    /* check result */
    //y->Dump(stderr, "y");
    //dedx->Dump(stderr, "dedy");

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * xGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * yGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * dedxGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * dedyGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);

    /* initialize variables */
    _SetDataFixedFloat(xGPU, 1.0F);
    yGPU->SetZeroAll();
    dedxGPU->SetZeroAll();
    _SetDataFixedFloat(dedyGPU, 1.5F);

    /* call Dropout function */
    _Dropout(xGPU, yGPU, seed, dropProb);
    _DropoutBackward(yGPU, xGPU, dedyGPU, dedxGPU, 1, dropProb);

    /* check result */
    //yGPU->Dump(stderr, "yGPU");
    //dedxGPU->Dump(stderr, "dedyGPU");

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

/* test for Dropout Function */
bool TestDropout()
{
    XPRINT(0, stdout, "[TEST DROPOUT] dropout function and its backward computation \n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestDropout1();

    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 1 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 1 passed!\n");
    
    /* case 2 test */
    caseFlag = TestDropout2();

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
