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
* $Created by: Lin Ye (email: linye2015@outlook.com) 2018-06-13
*/

#include "../core/utilities/CheckData.h"
#include "TSplit.h"

namespace nts { // namespace nt(NiuTrans.Tensor)

/* 
case 1: transform a tensor by splitting it, e.g., (N, M) -> (N/3, M, 3)
In this case, (4, 3) -> (2, 2, 3), whereToSplit=0, splitNum=2.
*/
bool TestSplit1()
{
    /* a source tensor of size (4, 3) */
    int sOrder = 2;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 4;
    sDimSize[1] = 3;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    /* a target tensor of size (2, 2, 3) */
    int tOrder = 3;
    int * tDimSize = new int[tOrder];
    tDimSize[0] = 2;
    tDimSize[1] = 2;
    tDimSize[2] = 3;

    int tUnitNum = 1;
    for (int i = 0; i < tOrder; i++)
        tUnitNum *= tDimSize[i];

    DTYPE sData[4][3] = { {0.0F, 1.0F, 2.0F},
                          {3.0F, 4.0F, 5.0F},
                          {0.1F, 1.1F, 2.1F},
                          {3.1F, 4.1F, 5.1F} };
    DTYPE answer[2][2][3] = { { {0.0F, 1.0F, 2.0F},
                                {3.0F, 4.0F, 5.0F} },
                              { {0.1F, 1.1F, 2.1F},
                                {3.1F, 4.1F, 5.1F} } };
  
    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(sOrder, sDimSize);
    XTensor * t = NewTensorV2(tOrder, tDimSize);
    XTensor tUser;

    /* initialize variables */
    s->SetData(sData, sUnitNum);
    t->SetZeroAll();

    /* call split function */
    _Split(s, t, 0, 2);
    tUser = Split(*s, 0, 2);

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

    /* call sum function */
    _Split(sGPU, tGPU, 0, 2);
    tUserGPU = Split(*sGPU, 0, 2);

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
case 2: transform a tensor by splitting it, e.g., (N, M) -> (N/3, M, 3)
In this case, (3, 4) -> (2, 3, 2), whereToSplit=1, splitNum=2.
*/
bool TestSplit2()
{
    /* a source tensor of size (3, 4) */
    int sOrder = 2;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 3;
    sDimSize[1] = 4;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    /* a target tensor of size (2, 3, 2) */
    int tOrder = 3;
    int * tDimSize = new int[tOrder];
    tDimSize[0] = 2;
    tDimSize[1] = 3;
    tDimSize[2] = 2;

    int tUnitNum = 1;
    for (int i = 0; i < tOrder; i++)
        tUnitNum *= tDimSize[i];

    DTYPE sData[3][4] = { {0.0F, 1.0F, 2.0F, 3.0F},
                          {4.0F, 5.0F, 0.1F, 1.1F},
                          {2.1F, 3.1F, 4.1F, 5.1F} };
    DTYPE answer[2][3][2] = { { {0.0F, 1.0F},
                                {4.0F, 5.0F},
                                {2.1F, 3.1F} },
                              { {2.0F, 3.0F},
                                {0.1F, 1.1F},
                                {4.1F, 5.1F} } };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(sOrder, sDimSize);
    XTensor * t = NewTensorV2(tOrder, tDimSize);
    XTensor tUser;

    /* initialize variables */
    s->SetData(sData, sUnitNum);
    t->SetZeroAll();;

    /* call split function */
    _Split(s, t, 1, 2);
    tUser = Split(*s, 1, 2);

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

    /* call sum function */
    _Split(sGPU, tGPU, 1, 2);
    tUserGPU = Split(*sGPU, 1, 2);

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
case 3: split a big tensor into small tensors
In this case, (3, 4) -> 2 * (3, 2) , whereToSplit=1, splitNum=2.
*/
bool TestSplit3()
{
    /* create list */
    TensorList * tList = new TensorList();
    TensorList tUserList;

    /* a source tensor of size (3, 4) */
    int sOrder = 2;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 3;
    sDimSize[1] = 4;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    /* a target tensor of size (3, 2) */
    int tOrder1 = 2;
    int * tDimSize1 = new int[tOrder1];
    tDimSize1[0] = 3;
    tDimSize1[1] = 2;

    int tUnitNum1 = 1;
    for (int i = 0; i < tOrder1; i++)
        tUnitNum1 *= tDimSize1[i];

    /* a target tensor of size (3 * 2) */
    int tOrder2 = 2;
    int * tDimSize2 = new int[tOrder2];
    tDimSize2[0] = 3;
    tDimSize2[1] = 2;

    int tUnitNum2 = 1;
    for (int i = 0; i < tOrder2; i++)
        tUnitNum2 *= tDimSize2[i];

    DTYPE sData[3][4] = { {0.0F, 1.0F, 2.0F, 3.0F},
                          {4.0F, 5.0F, 0.1F, 1.1F},
                          {2.1F, 3.1F, 4.1F, 5.1F} };
    DTYPE answer1[3][2] = { {0.0F, 1.0F},
                            {4.0F, 5.0F},
                            {2.1F, 3.1F} };
    DTYPE answer2[3][2] = { {2.0F, 3.0F},
                            {0.1F, 1.1F},
                            {4.1F, 5.1F} };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(sOrder, sDimSize);
    XTensor * t1 = NewTensorV2(tOrder1, tDimSize1);
    XTensor * t2 = NewTensorV2(tOrder2, tDimSize2);
    XTensor * t3 = NewTensorV2(tOrder2, tDimSize2);
    XTensor * t4 = NewTensorV2(tOrder2, tDimSize2);

    /* initialize variables */
    s->SetData(sData, sUnitNum);
    t1->SetZeroAll();
    t2->SetZeroAll();

    /* add tensors to list */
    tList->Add(t1);
    tList->Add(t2);

    tUserList.Add(t3);
    tUserList.Add(t4);

    /* call split function */
    _Split(s, tList, 1, 2);
    Split(*s, tUserList, 1, 2);

    /* check results */
    cpuTest = _CheckData(t1, answer1, tUnitNum1) && _CheckData((XTensor *)tUserList.Get(0), answer1, tUnitNum1) &&
              _CheckData(t2, answer2, tUnitNum2) && _CheckData((XTensor *)tUserList.Get(1), answer2, tUnitNum2);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* clear list */
    tList->Clear();
    tUserList.Clear();

    /* create tensor */
    XTensor * sGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU1 = NewTensorV2(tOrder1, tDimSize1, X_FLOAT, 1.0F, 0);
    XTensor * tGPU2 = NewTensorV2(tOrder2, tDimSize2, X_FLOAT, 1.0F, 0);
    XTensor * tGPU3 = NewTensorV2(tOrder2, tDimSize2, X_FLOAT, 1.0F, 0);
    XTensor * tGPU4 = NewTensorV2(tOrder2, tDimSize2, X_FLOAT, 1.0F, 0);

    /* Initialize variables */
    sGPU->SetData(sData, sUnitNum);
    tGPU1->SetZeroAll();
    tGPU2->SetZeroAll();

    /* add tensors to list */
    tList->Add(tGPU1);
    tList->Add(tGPU2);

    tUserList.Add(tGPU3);
    tUserList.Add(tGPU4);

    /* call Split function */
    _Split(sGPU, tList, 1, 2);
    Split(*sGPU, tUserList, 1, 2);

    /* check results */
    gpuTest = _CheckData(tGPU1, answer1, tUnitNum1) && _CheckData((XTensor *)tUserList.Get(0), answer1, tUnitNum1) &&
              _CheckData(tGPU2, answer2, tUnitNum2) && _CheckData((XTensor *)tUserList.Get(1), answer2, tUnitNum2);

    /* destroy variables */
    delete s;
    delete t1;
    delete t2;
    delete t3;
    delete t4;
    delete sGPU;
    delete tGPU1;
    delete tGPU2;
    delete tGPU3;
    delete tGPU4;
    delete[] sDimSize;
    delete[] tDimSize1;
    delete[] tDimSize2;
    delete tList;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete t1;
    delete t2;
    delete t3;
    delete t4;
    delete[] sDimSize;
    delete[] tDimSize1;
    delete[] tDimSize2;
    delete tList;

    return cpuTest;
#endif // USE_CUDA
}

/* other cases */
/*
TODO!!
*/

/* test for Split Function */
bool TestSplit()
{
    XPRINT(0, stdout, "[TEST SPLIT] split a big tensor into small tensors \n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestSplit1();

    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 1 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 1 passed!\n");

    /* case 2 test */
    caseFlag = TestSplit2();

    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 2 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 2 passed!\n");

    /* case 3 test */
    caseFlag = TestSplit3();

    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 3 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 3 passed!\n");

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
