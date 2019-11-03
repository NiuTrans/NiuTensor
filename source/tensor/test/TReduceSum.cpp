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

#include "../core/getandset/SetData.h"
#include "../core/utilities/CheckData.h"
#include "TReduceSum.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* 
case 1: test ReduceSum function.
Sum the items along a dimension of the tensor.
In this case, 
(2, 4) -> (4), dim = 0
(2, 4) -> (2), dim = 1
*/
bool TestReduceSum1()
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
    DTYPE answer1[4] = {4.0F, 6.0F, 8.0F, 10.0F};
    DTYPE answer2[2] = {6.0F, 22.0F};

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(sOrder, sDimSize);
    XTensor * shift1 = NewTensorV2(tOrder1, tDimSize1);
    XTensor * shift2 = NewTensorV2(tOrder2, tDimSize2);
    XTensor * t1 = NewTensorV2(tOrder1, tDimSize1);
    XTensor * t2 = NewTensorV2(tOrder2, tDimSize2);
    XTensor tUser1;
    XTensor tUser2;

    /* initialize variables */
    s->SetData(sData, sUnitNum);
    shift1->SetZeroAll();
    shift2->SetZeroAll();
    t1->SetZeroAll();
    t2->SetZeroAll();

    /* call ReduceSum function */
    _ReduceSum(s, t1, 0);
    _ReduceSum(s, t2, 1);
    tUser1 = ReduceSum(*s, 0, *shift1);
    tUser2 = ReduceSum(*s, 1, *shift2);

    /* check results */
    cpuTest = _CheckData(t1, answer1, tUnitNum1) && _CheckData(&tUser1, answer1, tUnitNum1) &&
              _CheckData(t2, answer2, tUnitNum2) && _CheckData(&tUser2, answer2, tUnitNum2);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * shiftGPU1 = NewTensorV2(tOrder1, tDimSize1, X_FLOAT, 1.0F, 0);
    XTensor * shiftGPU2 = NewTensorV2(tOrder2, tDimSize2, X_FLOAT, 1.0F, 0);
    XTensor * tGPU1 = NewTensorV2(tOrder1, tDimSize1, X_FLOAT, 1.0F, 0);
    XTensor * tGPU2 = NewTensorV2(tOrder2, tDimSize2, X_FLOAT, 1.0F, 0);
    XTensor tUserGPU1;
    XTensor tUserGPU2;

    /* initialize variables */
    sGPU->SetData(sData, sUnitNum);
    shiftGPU1->SetZeroAll();
    shiftGPU2->SetZeroAll();
    tGPU1->SetZeroAll();
    tGPU2->SetZeroAll();

    /* call ReduceSum function */
    _ReduceSum(sGPU, tGPU1, 0);
    _ReduceSum(sGPU, tGPU2, 1);
    tUserGPU1 = ReduceSum(*sGPU, 0, *shiftGPU1);
    tUserGPU2 = ReduceSum(*sGPU, 1, *shiftGPU2);

    /* check results */
    gpuTest = _CheckData(tGPU1, answer1, tUnitNum1) && _CheckData(&tUserGPU1, answer1, tUnitNum1) &&
              _CheckData(tGPU2, answer2, tUnitNum2) && _CheckData(&tUserGPU2, answer2, tUnitNum2);

    /* destroy variables */
    delete s;
    delete shift1;
    delete shift2;
    delete t1;
    delete t2;
    delete sGPU;
    delete shiftGPU1;
    delete shiftGPU2;
    delete tGPU1;
    delete tGPU2;
    delete[] sDimSize;
    delete[] tDimSize1;
    delete[] tDimSize2;
    
    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete shift1;
    delete shift2;
    delete t1;
    delete t2;
    delete[] sDimSize;
    delete[] tDimSize1;
    delete[] tDimSize2;

    return cpuTest;
#endif // USE_CUDA
}

/* 
case 2: test ReduceSum function.
Sum the items along a dimension of the tensor.
In this case, 
C = 1, A >= 10, B >= 128
(50, 1000000) -> (50), dim = 1
*/
bool TestReduceSum2()
{
    /* a tensor of size (50, 1000000) */
    int sOrder = 2;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 50;
    sDimSize[1] = 1000000;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    /* a tensor of size (50) */
    int tOrder = 1;
    int * tDimSize = new int[tOrder];
    tDimSize[0] = 50;

    int tUnitNum = 1;
    for (int i = 0; i < tOrder; i++)
        tUnitNum *= tDimSize[i];

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(sOrder, sDimSize);
    XTensor * t = NewTensorV2(tOrder, tDimSize);
    XTensor * answer = NewTensorV2(tOrder, tDimSize);
    XTensor tUser;

    /* initialize variables */
    _SetDataFixedFloat(s, 1.0F);
    _SetDataFixedFloat(answer, (float)s->GetDim(1));

    /* call ReduceSum function */
    _ReduceSum(s, t, 1);
    tUser = ReduceSum(*s, 1);

    /* check results */
    cpuTest = _CheckData(t, answer->data, tUnitNum) && _CheckData(&tUser, answer->data, tUnitNum);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU = NewTensorV2(tOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor tUserGPU;

    /* initialize variables */
    _SetDataFixedFloat(sGPU, 1.0F);

    /* call ReduceSum function */
    _ReduceSum(sGPU, tGPU, 1);
    tUserGPU = ReduceSum(*sGPU, 1);

    /* check results */
    gpuTest = _CheckData(tGPU, answer->data, tUnitNum) && _CheckData(&tUserGPU, answer->data, tUnitNum);

    /* destroy variables */
    delete s;
    delete t;
    delete answer;
    delete sGPU;
    delete tGPU;
    delete[] sDimSize;
    delete[] tDimSize;
    
    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete t;
    delete answer;
    delete[] sDimSize;
    delete[] tDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* 
case 3: test ReduceSum function.
Sum the items along a dimension of the tensor.
In this case, 
C = 1, A >= 10, B < 128
(1000000, 50) -> (1000000), dim = 1
*/
bool TestReduceSum3()
{
    /* a tensor of size (1000000, 50) */
    int sOrder = 2;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 1000000;
    sDimSize[1] = 50;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    /* a tensor of size (1000000) */
    int tOrder = 1;
    int * tDimSize = new int[tOrder];
    tDimSize[0] = 1000000;

    int tUnitNum = 1;
    for (int i = 0; i < tOrder; i++)
        tUnitNum *= tDimSize[i];

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(sOrder, sDimSize);
    XTensor * t = NewTensorV2(tOrder, tDimSize);
    XTensor * answer = NewTensorV2(tOrder, tDimSize);
    XTensor tUser;

    /* initialize variables */
    _SetDataFixedFloat(s, 1.0F);
    _SetDataFixedFloat(answer, (float)s->GetDim(1));

    /* call ReduceSum function */
    _ReduceSum(s, t, 1);
    tUser = ReduceSum(*s, 1);

    /* check results */
    cpuTest = _CheckData(t, answer->data, tUnitNum) && _CheckData(&tUser, answer->data, tUnitNum);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU = NewTensorV2(tOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor tUserGPU;

    /* initialize variables */
    _SetDataFixedFloat(sGPU, 1.0F);

    /* call ReduceSum function */
    _ReduceSum(sGPU, tGPU, 1);
    tUserGPU = ReduceSum(*sGPU, 1);

    /* check results */
    gpuTest = _CheckData(tGPU, answer->data, tUnitNum) && _CheckData(&tUserGPU, answer->data, tUnitNum);

    /* destroy variables */
    delete s;
    delete t;
    delete answer;
    delete sGPU;
    delete tGPU;
    delete[] sDimSize;
    delete[] tDimSize;
    
    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete t;
    delete answer;
    delete[] sDimSize;
    delete[] tDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* 
case 4: test ReduceSum function.
Sum the items along a dimension of the tensor.
In this case, 
C = 1, A < 10, B is free
(5, 1000000) -> (5), dim = 1
*/
bool TestReduceSum4()
{
    /* a tensor of size (5, 1000000) */
    int sOrder = 2;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 5;
    sDimSize[1] = 1000000;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    /* a tensor of size (5) */
    int tOrder = 1;
    int * tDimSize = new int[tOrder];
    tDimSize[0] = 5;

    int tUnitNum = 1;
    for (int i = 0; i < tOrder; i++)
        tUnitNum *= tDimSize[i];

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(sOrder, sDimSize);
    XTensor * t = NewTensorV2(tOrder, tDimSize);
    XTensor * answer = NewTensorV2(tOrder, tDimSize);
    XTensor tUser;

    /* initialize variables */
    _SetDataFixedFloat(s, 1.0F);
    _SetDataFixedFloat(answer, (float)s->GetDim(1));

    /* call ReduceSum function */
    _ReduceSum(s, t, 1);
    tUser = ReduceSum(*s, 1);

    /* check results */
    cpuTest = _CheckData(t, answer->data, tUnitNum) && _CheckData(&tUser, answer->data, tUnitNum);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU = NewTensorV2(tOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor tUserGPU;

    /* initialize variables */
    _SetDataFixedFloat(sGPU, 1.0F);

    /* call ReduceSum function */
    _ReduceSum(sGPU, tGPU, 1);
    tUserGPU = ReduceSum(*sGPU, 1);

    /* check results */
    gpuTest = _CheckData(tGPU, answer->data, tUnitNum) && _CheckData(&tUserGPU, answer->data, tUnitNum);

    /* destroy variables */
    delete s;
    delete t;
    delete answer;
    delete sGPU;
    delete tGPU;
    delete[] sDimSize;
    delete[] tDimSize;
    
    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete t;
    delete answer;
    delete[] sDimSize;
    delete[] tDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* 
case 5: test ReduceSum function.
Sum the items along a dimension of the tensor.
In this case, 
C != 1, A*C > 4096
(500, 1000, 500) -> (500, 500), dim = 1
*/
bool TestReduceSum5()
{
    /* a tensor of size (500, 1000, 500) */
    int sOrder = 3;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 500;
    sDimSize[1] = 1000;
    sDimSize[2] = 500;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    /* a tensor of size (500, 500) */
    int tOrder = 2;
    int * tDimSize = new int[tOrder];
    tDimSize[0] = 50;
    tDimSize[1] = 50;

    int tUnitNum = 1;
    for (int i = 0; i < tOrder; i++)
        tUnitNum *= tDimSize[i];

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(sOrder, sDimSize);
    XTensor * t = NewTensorV2(tOrder, tDimSize);
    XTensor * answer = NewTensorV2(tOrder, tDimSize);
    XTensor tUser;

    /* initialize variables */
    _SetDataFixedFloat(s, 1.0F);
    _SetDataFixedFloat(answer, (float)s->GetDim(1));

    /* call ReduceSum function */
    _ReduceSum(s, t, 1);
    tUser = ReduceSum(*s, 1);

    /* check results */
    cpuTest = _CheckData(t, answer->data, tUnitNum) && _CheckData(&tUser, answer->data, tUnitNum);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU = NewTensorV2(tOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor tUserGPU;

    /* initialize variables */
    _SetDataFixedFloat(sGPU, 1.0F);

    /* call ReduceSum function */
    _ReduceSum(sGPU, tGPU, 1);
    tUserGPU = ReduceSum(*sGPU, 1);

    /* check results */
    gpuTest = _CheckData(tGPU, answer->data, tUnitNum) && _CheckData(&tUserGPU, answer->data, tUnitNum);

    /* destroy variables */
    delete s;
    delete t;
    delete answer;
    delete sGPU;
    delete tGPU;
    delete[] sDimSize;
    delete[] tDimSize;
    
    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete t;
    delete answer;
    delete[] sDimSize;
    delete[] tDimSize;

    return cpuTest;
#endif // USE_CUDA
}


/* 
case 6: test ReduceSum function.
Sum the items along a dimension of the tensor.
In this case, 
C != 1, A*C <= 4096
(50, 10000, 50) -> (50, 50), dim = 1
*/
bool TestReduceSum6()
{
    /* a tensor of size (50, 10000, 50) */
    int sOrder = 3;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 50;
    sDimSize[1] = 10000;
    sDimSize[2] = 50;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    /* a tensor of size (50, 50) */
    int tOrder = 2;
    int * tDimSize = new int[tOrder];
    tDimSize[0] = 50;
    tDimSize[1] = 50;

    int tUnitNum = 1;
    for (int i = 0; i < tOrder; i++)
        tUnitNum *= tDimSize[i];

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(sOrder, sDimSize);
    XTensor * t = NewTensorV2(tOrder, tDimSize);
    XTensor * answer = NewTensorV2(tOrder, tDimSize);
    XTensor tUser;

    /* initialize variables */
    _SetDataFixedFloat(s, 1.0F);
    _SetDataFixedFloat(answer, (float)s->GetDim(1));

    /* call ReduceSum function */
    _ReduceSum(s, t, 1);
    tUser = ReduceSum(*s, 1);

    /* check results */
    cpuTest = _CheckData(t, answer->data, tUnitNum) && _CheckData(&tUser, answer->data, tUnitNum);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU = NewTensorV2(tOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor tUserGPU;

    /* initialize variables */
    _SetDataFixedFloat(sGPU, 1.0F);

    /* call ReduceSum function */
    _ReduceSum(sGPU, tGPU, 1);
    tUserGPU = ReduceSum(*sGPU, 1);

    /* check results */
    gpuTest = _CheckData(tGPU, answer->data, tUnitNum) && _CheckData(&tUserGPU, answer->data, tUnitNum);

    /* destroy variables */
    delete s;
    delete t;
    delete answer;
    delete sGPU;
    delete tGPU;
    delete[] sDimSize;
    delete[] tDimSize;
    
    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete t;
    delete answer;
    delete[] sDimSize;
    delete[] tDimSize;

    return cpuTest;
#endif // USE_CUDA
}


/* other cases */
/*
TODO!!
*/

/* test for ReduceSum Function */
bool TestReduceSum()
{
    XPRINT(0, stdout, "[TEST ReduceSum] sum the items along a dimension of the tensor.\n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestReduceSum1();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 1 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 1 passed!\n");

    /* case 2 test */
    caseFlag = TestReduceSum2();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 2 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 2 passed!\n");

    ///* case 3 test */
    //caseFlag = TestReduceSum3();
    //if (!caseFlag) {
    //    returnFlag = false;
    //    XPRINT(0, stdout, ">> case 3 failed!\n");
    //}
    //else
    //    XPRINT(0, stdout, ">> case 3 passed!\n");

    /* case 4 test */
    caseFlag = TestReduceSum4();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 4 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 4 passed!\n");

    ///* case 5 test */
    //caseFlag = TestReduceSum5();
    //if (!caseFlag) {
    //    returnFlag = false;
    //    XPRINT(0, stdout, ">> case 5 failed!\n");
    //}
    //else
    //    XPRINT(0, stdout, ">> case 5 passed!\n");
    
    /* case 6 test */
    caseFlag = TestReduceSum6();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 6 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 6 passed!\n");

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
