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
* $Created by: Xu Chen (email: hello_master1954@163.com) 2018-06-13
*/

#include "../XTensor.h"
#include "../XList.h"
#include "../core/utilities/CheckData.h"
#include "TMerge.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* 
case 1: transform a tensor by merging it along with a dimension. 
In this case, (3, 2) -> (6), whereToMerge=1, leadingDim=0.
*/
bool TestMerge1()
{
    /* a source tensor of size (2, 3) */
    int sOrder = 2;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 2;
    sDimSize[1] = 3;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    /* a target tensor of size (6, ) */
    int tOrder = 1;
    int * tDimSize = new int[tOrder];
    tDimSize[0] = 6;

    int tUnitNum = 1;
    for (int i = 0; i < tOrder; i++)
        tUnitNum *= tDimSize[i];

    DTYPE sData[2][3] = { {0.0F, 1.0F, 2.0F},
                          {3.0F, 4.0F, 5.0F} };
    DTYPE answer[6] = {0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    
    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(sOrder, sDimSize);
    XTensor * t = NewTensorV2(tOrder, tDimSize);
    XTensor tUser;

    /* initialize variables */
    s->SetData(sData, sUnitNum);
    t->SetZeroAll();

    /* call Merge function */
    _Merge(s, t, 1, 0);
    tUser = Merge(*s, 1, 0);

    /* check results */
    cpuTest = _CheckData(t, answer, tUnitNum) && _CheckData(&tUser, answer, tUnitNum);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensor */
    XTensor * sGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU = NewTensorV2(tOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor tUserGPU;

    /* Initialize variables */
    sGPU->SetData(sData, sUnitNum);
    tGPU->SetZeroAll();

    /* call Merge function */
    _Merge(sGPU, tGPU, 1, 0);
    tUserGPU = Merge(*sGPU, 1, 0);

    /* check results */
    gpuTest = _CheckData(tGPU, answer, tUnitNum) && _CheckData(&tUserGPU, answer, tUnitNum);

    /* destroy variables */
    delete s;
    delete t;
    delete sGPU;
    delete tGPU;
    delete[] sDimSize;
    delete[] tDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete t;
    delete[] sDimSize;
    delete[] tDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* 
case 2: transform a tensor by merging it along with a dimension. 
In this case, 
(2, 2, 3) -> (4, 3), whereToMerge=1, leadingDim=0.
(2, 2, 3) -> (2, 6), whereToMerge=2, leadingDim=0.
*/
bool TestMerge2()
{
    /* a source tensor of size (2, 2, 3) */
    int sOrder = 3;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 2;
    sDimSize[1] = 2;
    sDimSize[2] = 3;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    /* a target tensor of size (4, 3) */
    int tOrder1 = 2;
    int * tDimSize1 = new int[tOrder1];
    tDimSize1[0] = 4;
    tDimSize1[1] = 3;

    int tUnitNum1 = 1;
    for (int i = 0; i < tOrder1; i++)
        tUnitNum1 *= tDimSize1[i];

    /* a target tensor of size (2, 6) */
    int tOrder2 = 2;
    int * tDimSize2 = new int[tOrder2];
    tDimSize2[0] = 2;
    tDimSize2[1] = 6;

    int tUnitNum2 = 1;
    for (int i = 0; i < tOrder2; i++)
        tUnitNum2 *= tDimSize2[i];

    DTYPE sData[2][2][3] = { { {0.0F, 1.0F, 2.0F},
                               {4.0F, 5.0F, 6.0F} },
                             { {-1.0F, 2.0F, 3.0F},
                               {-4.0F, -5.0F, -6.0F} } };
    DTYPE answer1[4][3] = { {0.0F, 1.0F, 2.0F},
                            {4.0F, 5.0F, 6.0F},
                            {-1.0F, 2.0F, 3.0F},
                            {-4.0F, -5.0F, -6.0F} };
    DTYPE answer2[2][6] = { {0.0F, 1.0F, 2.0F, -1.0F, 2.0F, 3.0F},
                            {4.0F, 5.0F, 6.0F, -4.0F, -5.0F, -6.0F} };

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

    /* call Merge function */
    _Merge(s, t1, 1, 0);
    _Merge(s, t2, 2, 0);
    tUser1 = Merge(*s, 1, 0);
    tUser2 = Merge(*s, 2, 0);

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

    /* call Merge function */
    _Merge(sGPU, tGPU1, 1, 0);
    _Merge(sGPU, tGPU2, 2, 0);
    tUserGPU1 = Merge(*sGPU, 1, 0);
    tUserGPU2 = Merge(*sGPU, 2, 0);

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

/* 
case 3: merge small tensors into a big tensor. 
In this case, 2 * (2, 4) -> (4, 4), whereToMerge=0.
*/
bool TestMerge3()
{
    /* create list */
    TensorList * smallList = new TensorList();

    /* a small tensor of size (2, 4) */
    int sOrder = 2;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 2;
    sDimSize[1] = 4;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    DTYPE sData1[2][4] = { {0.0F, 1.0F, 2.0F, 3.0F},
                           {4.0F, 5.0F, 6.0F, 7.0F} };
    DTYPE sData2[2][4] = { {0.0F, -1.0F, -2.0F, -3.0F},
                           {-4.0F, -5.0F, -6.0F, -7.0F} };

    /* a target tensor of size (4, 4) */
    int tOrder = 2;
    int * tDimSize = new int[tOrder];
    tDimSize[0] = 4;
    tDimSize[1] = 4;

    int tUnitNum = 1;
    for (int i = 0; i < tOrder; i++)
        tUnitNum *= tDimSize[i];

    DTYPE answer[4][4] = { {0.0F, 1.0F, 2.0F, 3.0F},
                           {4.0F, 5.0F, 6.0F, 7.0F},
                           {0.0F, -1.0F, -2.0F, -3.0F},
                           {-4.0F, -5.0F, -6.0F, -7.0F} };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s1 = NewTensorV2(sOrder, sDimSize);
    XTensor * s2 = NewTensorV2(sOrder, sDimSize);
    XTensor * t = NewTensorV2(tOrder, tDimSize);
    XTensor tUser;

    /* initialize variables */
    s1->SetData(sData1, sUnitNum);
    s2->SetData(sData2, sUnitNum);
    t->SetZeroAll();

    /* add tensors to list */
    smallList->Add(s1);
    smallList->Add(s2);

    /* call Merge function */
    _Merge(smallList, t, 0);
    tUser = Merge(*smallList, 0);

    /* check results */
    cpuTest = _CheckData(t, answer, tUnitNum) && _CheckData(&tUser, answer, tUnitNum);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* clear list */
    smallList->Clear();

    /* create tensors */
    XTensor * sGPU1 = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * sGPU2 = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU = NewTensorV2(tOrder, tDimSize);
    XTensor tUserGPU;

    /* initialize variables */
    sGPU1->SetData(sData1, sUnitNum);
    sGPU2->SetData(sData2, sUnitNum);
    tGPU->SetZeroAll();

    /* add tensors to list*/
    smallList->Add(sGPU1);
    smallList->Add(sGPU2);

    /* call Merge function */
    _Merge(smallList, tGPU, 0);
    tUserGPU = Merge(*smallList, 0);

    /* check results */
    gpuTest = _CheckData(tGPU, answer, tUnitNum) && _CheckData(&tUserGPU, answer, tUnitNum);

    /* destroy variables */
    delete s1;
    delete s2;
    delete t;
    delete sGPU1;
    delete sGPU2;
    delete tGPU;
    delete[] sDimSize;
    delete[] tDimSize;
    delete smallList;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s1;
    delete s2;
    delete t;
    delete[] sDimSize;
    delete[] tDimSize;
    delete smallList;

    return cpuTest;
#endif // USE_CUDA
}

/* 
case 4: merge small tensors into a big tensor. 
In this case, 2 * (2, 4) -> (2, 8), whereToMerge=1.
*/
bool TestMerge4()
{
    /* create list */
    TensorList * smallList = new TensorList();

    /* a small tensor of size (2, 4) */
    int sOrder = 2;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 2;
    sDimSize[1] = 4;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    DTYPE sData1[2][4] = { {0.0F, 1.0F, 2.0F, 3.0F},
                           {4.0F, 5.0F, 6.0F, 7.0F} };
    DTYPE sData2[2][4] = { {0.0F, -1.0F, -2.0F, -3.0F},
                           {-4.0F, -5.0F, -6.0F, -7.0F} };

    /* a target tensor of size (4, 4) */
    int tOrder = 2;
    int * tDimSize = new int[tOrder];
    tDimSize[0] = 2;
    tDimSize[1] = 8;

    int tUnitNum = 1;
    for (int i = 0; i < tOrder; i++)
        tUnitNum *= tDimSize[i];

    DTYPE answer[2][8] = { {0.0F, 1.0F, 2.0F, 3.0F, 0.0F, -1.0F, -2.0F, -3.0F},
                           {4.0F, 5.0F, 6.0F, 7.0F, -4.0F, -5.0F, -6.0F, -7.0F} };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s1 = NewTensorV2(sOrder, sDimSize);
    XTensor * s2 = NewTensorV2(sOrder, sDimSize);
    XTensor * t = NewTensorV2(tOrder, tDimSize);
    XTensor tUser;

    /* initialize variables */
    s1->SetData(sData1, sUnitNum);
    s2->SetData(sData2, sUnitNum);
    t->SetZeroAll();

    /* add tensors to list */
    smallList->Add(s1);
    smallList->Add(s2);

    /* call Merge function */
    _Merge(smallList, t, 1);
    tUser = Merge(*smallList, 1);

    /* check results */
    cpuTest = _CheckData(t, answer, tUnitNum) && _CheckData(&tUser, answer, tUnitNum);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* clear list */
    smallList->Clear();

    /* create tensors */
    XTensor * sGPU1 = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * sGPU2 = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU = NewTensorV2(tOrder, tDimSize);
    XTensor tUserGPU;

    /* initialize variables */
    sGPU1->SetData(sData1, sUnitNum);
    sGPU2->SetData(sData2, sUnitNum);
    tGPU->SetZeroAll();

    /* add tensors to list*/
    smallList->Add(sGPU1);
    smallList->Add(sGPU2);

    /* call Merge function */
    _Merge(smallList, tGPU, 1);
    tUserGPU = Merge(*smallList, 1);

    /* check results */
    gpuTest = _CheckData(tGPU, answer, tUnitNum) && _CheckData(&tUserGPU, answer, tUnitNum);

    /* destroy variables */
    delete s1;
    delete s2;
    delete t;
    delete sGPU1;
    delete sGPU2;
    delete tGPU;
    delete[] sDimSize;
    delete[] tDimSize;
    delete smallList;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s1;
    delete s2;
    delete t;
    delete[] sDimSize;
    delete[] tDimSize;
    delete smallList;

    return cpuTest;
#endif // USE_CUDA
}

/* other cases */
/*
    TODO!!
*/

/* test for Merge Function */
bool TestMerge()
{
    XPRINT(0, stdout, "[TEST MERGE] transform a tensor by merging it alone with a dimension or merge small tensors into a big tensor\n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestMerge1();

    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 1 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 1 passed!\n");

    /* case 2 test */
    caseFlag = TestMerge2();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 2 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 2 passed!\n");

    /* case 3 test */
    caseFlag = TestMerge3();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 3 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 3 passed!\n");

    /* case 4 test */
    caseFlag = TestMerge4();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 4 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 4 passed!\n");

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
