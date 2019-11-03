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
* $Created by: Xu Chen (email: hello_master1954@163.com) 2018-06-15
*/

#include "../XTensor.h"
#include "../core/utilities/CheckData.h"
#include "TMatrixMulBatched.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* 
case 1: matrix multiplication of the two tensors. 
In this case, a=(2, 3), b=(2, 3) -> c=(2, 2), transposedA=X_NOTRANS, transposedB=X_NOTRANS.
*/
bool TestMatrixMulBatched1()
{
    /* a source tensor of size (2, 3) */
    int sOrder1 = 2;
    int * sDimSize1 = new int[sOrder1];
    sDimSize1[0] = 2;
    sDimSize1[1] = 3;

    int sUnitNum1 = 1;
    for (int i = 0; i < sOrder1; i++)
        sUnitNum1 *= sDimSize1[i];

    /* a source tensor of size (3, 2) */
    int sOrder2 = 2;
    int * sDimSize2 = new int[sOrder2];
    sDimSize2[0] = 3;
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

    DTYPE sData1[2][3] = { {1.0F, 2.0F, 3.0F},
                           {-4.0F, 5.0F, 6.0F} };
    DTYPE sData2[3][2] = { {0.0F, -1.0F},
                           {1.0F, 2.0F}, 
                           {2.0F, 1.0F} };
    DTYPE answer[2][2] = { {8.0F, 6.0F}, 
                           {17.0F, 20.0F} };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s1 = NewTensorV2(sOrder1, sDimSize1);
    XTensor * s2 = NewTensorV2(sOrder2, sDimSize2);
    XTensor * t = NewTensorV2(tOrder, tDimSize);
    XTensor tUser;

    /* initialize variables */
    s1->SetData(sData1, sUnitNum1);
    s2->SetData(sData2, sUnitNum2);
    t->SetZeroAll();

    /* call MatrixMulBatched function */
    _MatrixMulBatched(s1, X_NOTRANS, s2, X_NOTRANS, t);
    tUser = MatrixMulBatched(*s1, X_NOTRANS, *s2, X_NOTRANS);
    
    /* check results */
    cpuTest = _CheckData(t, answer, tUnitNum) && _CheckData(&tUser, answer, tUnitNum);
    
#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensor */
    XTensor * sGPU1 = NewTensorV2(sOrder1, sDimSize1, X_FLOAT, 1.0F, 0);
    XTensor * sGPU2 = NewTensorV2(sOrder2, sDimSize2, X_FLOAT, 1.0F, 0);
    XTensor * tGPU = NewTensorV2(tOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor tUserGPU;

    /* Initialize variables */
    sGPU1->SetData(sData1, sUnitNum1);
    sGPU2->SetData(sData2, sUnitNum2);
    tGPU->SetZeroAll();
    
    /* call MatrixMulBatched function */
    _MatrixMulBatched(sGPU1, X_NOTRANS, sGPU2, X_NOTRANS, tGPU);
    tUserGPU = MatrixMulBatched(*sGPU1, X_NOTRANS, *sGPU2, X_NOTRANS);
    
    /* check results */
    gpuTest = _CheckData(tGPU, answer, tUnitNum) && _CheckData(&tUserGPU, answer, tUnitNum);
    
    /* destroy variables */
    delete s1;
    delete s2;
    delete t;
    delete sGPU1;
    delete sGPU2;
    delete tGPU;
    delete[] sDimSize1;
    delete[] sDimSize2;
    delete[] tDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s1;
    delete s2;
    delete t;
    delete[] sDimSize1;
    delete[] sDimSize2;
    delete[] tDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/*
case 2: matrix multiplication of the two tensors. 
In this case, a=(2, 2, 3), b=(2, 3, 2) -> c=(2, 2, 2), transposedA=X_NOTRANS, transposedB=X_NOTRANS.
*/
bool TestMatrixMulBatched2()
{
    /* a source tensor of size (2, 2, 3) */
    int sOrder1 = 3;
    int * sDimSize1 = new int[sOrder1];
    sDimSize1[0] = 2;
    sDimSize1[1] = 2;
    sDimSize1[2] = 3;

    int sUnitNum1 = 1;
    for (int i = 0; i < sOrder1; i++)
        sUnitNum1 *= sDimSize1[i];

    /* a source tensor of size (2, 3, 2) */
    int sOrder2 = 3;
    int * sDimSize2 = new int[sOrder2];
    sDimSize2[0] = 2;
    sDimSize2[1] = 3;
    sDimSize2[2] = 2;

    int sUnitNum2 = 1;
    for (int i = 0; i < sOrder2; i++)
        sUnitNum2 *= sDimSize2[i];

    /* a target tensor of size (2, 2, 2) */
    int tOrder = 3;
    int * tDimSize = new int[tOrder];
    tDimSize[0] = 2;
    tDimSize[1] = 2;
    tDimSize[2] = 2;

    int tUnitNum = 1;
    for (int i = 0; i < tOrder; i++)
        tUnitNum *= tDimSize[i];

    DTYPE sData1[2][2][3] = { { {0.0F, -1.0F, 2.0F},
                                {2.0F, 1.0F, 3.0F} },
                              { {1.0F, 2.0F, 4.0F}, 
                                {3.0F, 1.0F, 2.0F} } };
    DTYPE sData2[2][3][2] = { { {1.0F, 2.0F},
                                {-4.0F, 3.0F},
                                {2.0F, 6.0F} },
                              { {1.0F, 2.0F},
                                {3.0F, 4.0F},
                                {5.0F, 6.0F} } };
    DTYPE answer[2][2][2] = { { {8.0F, 9.0F}, 
                                {4.0F, 25.0F} },
                              { {27.0F, 34.0F}, 
                                {16.0F, 22.0F} } };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s1 = NewTensorV2(sOrder1, sDimSize1);
    XTensor * s2 = NewTensorV2(sOrder2, sDimSize2);
    XTensor * t = NewTensorV2(tOrder, tDimSize);
    XTensor tUser;

    /* initialize variables */
    s1->SetData(sData1, sUnitNum1);
    s2->SetData(sData2, sUnitNum2);
    t->SetZeroAll();

    /* call MatrixMulBatched function */
    _MatrixMulBatched(s1, X_NOTRANS, s2, X_NOTRANS, t);
    tUser = MatrixMulBatched(*s1, X_NOTRANS, *s2, X_NOTRANS);

    /* check results */
    cpuTest = _CheckData(t, answer, tUnitNum) && _CheckData(&tUser, answer, tUnitNum);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensor */
    XTensor * sGPU1 = NewTensorV2(sOrder1, sDimSize1, X_FLOAT, 1.0F, 0);
    XTensor * sGPU2 = NewTensorV2(sOrder2, sDimSize2, X_FLOAT, 1.0F, 0);
    XTensor * tGPU = NewTensorV2(tOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor tUserGPU;

    /* Initialize variables */
    sGPU1->SetData(sData1, sUnitNum1);
    sGPU2->SetData(sData2, sUnitNum2);
    tGPU->SetZeroAll();

    /* call MatrixMulBatched function */
    _MatrixMulBatched(sGPU1, X_NOTRANS, sGPU2, X_NOTRANS, tGPU);
    tUserGPU = MatrixMulBatched(*sGPU1, X_NOTRANS, *sGPU2, X_NOTRANS);

    /* check results */
    gpuTest = _CheckData(tGPU, answer, tUnitNum) && _CheckData(&tUserGPU, answer, tUnitNum);

    /* destroy variables */
    delete s1;
    delete s2;
    delete t;
    delete sGPU1;
    delete sGPU2;
    delete tGPU;
    delete[] sDimSize1;
    delete[] sDimSize2;
    delete[] tDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s1;
    delete s2;
    delete t;
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

/* test for TestMatrixMulBatched Function */
bool TestMatrixMulBatched()
{
    XPRINT(0, stdout, "[TEST MATRIXMULBATCHED] matrix multiplication of the two tensors \n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestMatrixMulBatched1();

    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 1 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 1 passed!\n");

    /* case 2 test */
    caseFlag = TestMatrixMulBatched2();

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
