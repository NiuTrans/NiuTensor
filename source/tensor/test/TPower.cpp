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
* $Created by: Lin Ye (email: linye2015@outlook.com) 2018-06-15
*/

#include "../core/math/Binary.h"
#include "../core/utilities/CheckData.h"
#include "../XUtility.h"
#include "TPower.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* 
case 1: get the power(a, p) 
In this case, p=2.
*/
bool TestPower1()
{
    /* a tensor of size (3, 2) */
    int aOrder = 2;
    int * aDimSize = new int[aOrder];
    aDimSize[0] = 3;
    aDimSize[1] = 2;

    int aUnitNum = 1;
    for (int i = 0; i < aOrder; i++)
        aUnitNum *= aDimSize[i];

    DTYPE aData[3][2] = { {1.0F, 2.0F},
                          {3.0F, 4.0F},
                          {5.0F, 6.0F} };
    DTYPE answer[3][2] = { {1.0F, 4.0F},
                           {9.0F, 16.0F},
                           {25.0F, 36.0F} };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * a = NewTensorV2(aOrder, aDimSize);
    XTensor * b = NewTensorV2(aOrder, aDimSize);
    XTensor * aMe = NewTensorV2(aOrder, aDimSize);
    XTensor bUser;

    /* initialize variables */
    a->SetData(aData, aUnitNum);
    aMe->SetData(aData, aUnitNum);

    /* call Power function */
    _Power(a, b, 2.0F);
    _PowerMe(aMe, 2.0F);
    bUser = Power(*a, 2.0F);

    /* check results */
    cpuTest = _CheckData(b, answer, aUnitNum, 1e-4F) &&
              _CheckData(aMe, answer, aUnitNum, 1e-4F) &&
              _CheckData(&bUser, answer, aUnitNum, 1e-4F);
    
#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensor */
    XTensor * aGPU = NewTensorV2(aOrder, aDimSize, X_FLOAT, 1.0F, 0);
    XTensor * bGPU = NewTensorV2(aOrder, aDimSize, X_FLOAT, 1.0F, 0);
    XTensor * aMeGPU = NewTensorV2(aOrder, aDimSize, X_FLOAT, 1.0F, 0);
    XTensor bUserGPU;

    /* Initialize variables */
    aGPU->SetData(aData, aUnitNum);
    aMeGPU->SetData(aData, aUnitNum);

    /* call power function */
    _Power(aGPU, bGPU, 2.0F);
    _PowerMe(aMeGPU, 2.0F);
    bUserGPU = Power(*aGPU, 2.0F);

    /* check results */
    gpuTest = _CheckData(bGPU, answer, aUnitNum, 1e-4F) &&
              _CheckData(aMeGPU, answer, aUnitNum, 1e-4F) &&
              _CheckData(&bUserGPU, answer, aUnitNum, 1e-4F);
    
    /* destroy variables */
    delete a;
    delete b;
    delete aMe;
    delete aGPU;
    delete bGPU;
    delete aMeGPU;
    delete[] aDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete a;
    delete b;
    delete aMe;
    delete[] aDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* 
case 2: get the power(a, p)
In this case, p=1.
*/
bool TestPower2()
{
    /* a tensor of size (3, 2) */
    int aOrder = 2;
    int * aDimSize = new int[aOrder];
    aDimSize[0] = 3;
    aDimSize[1] = 2;

    int aUnitNum = 1;
    for (int i = 0; i < aOrder; i++)
        aUnitNum *= aDimSize[i];

    DTYPE aData[3][2] = { {0.0F, 1.0F},
                          {2.0F, 3.0F},
                          {4.0F, 5.0F} };
    DTYPE answer[3][2] = { {0.0F, 1.0F},
                           {2.0F, 3.0F},
                           {4.0F, 5.0F} };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * a = NewTensorV2(aOrder, aDimSize);
    XTensor * b = NewTensorV2(aOrder, aDimSize);
    XTensor * aMe = NewTensorV2(aOrder, aDimSize);
    XTensor bUser;

    /* initialize variables */
    a->SetData(aData, aUnitNum);
    aMe->SetData(aData, aUnitNum);

    /* call Power function */
    _Power(a, b, 1.0F);
    _PowerMe(aMe, 1.0F);
    bUser = Power(*a, 1.0F);

    /* check results */
    cpuTest = _CheckData(b, answer, aUnitNum, 1e-4F) &&
              _CheckData(aMe, answer, aUnitNum, 1e-4F) &&
              _CheckData(&bUser, answer, aUnitNum, 1e-4F);
    
#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensor */
    XTensor * aGPU = NewTensorV2(aOrder, aDimSize, X_FLOAT, 1.0F, 0);
    XTensor * bGPU = NewTensorV2(aOrder, aDimSize, X_FLOAT, 1.0F, 0);
    XTensor * aMeGPU = NewTensorV2(aOrder, aDimSize, X_FLOAT, 1.0F, 0);
    XTensor bUserGPU;

    /* Initialize variables */
    aGPU->SetData(aData, aUnitNum);
    aMeGPU->SetData(aData, aUnitNum);

    /* call Power function */
    _Power(aGPU, bGPU, 1.0F);
    _PowerMe(aMeGPU, 1.0F);
    bUserGPU = Power(*aGPU, 1.0F);

    /* check results */
    gpuTest = _CheckData(bGPU, answer, aUnitNum, 1e-4F) &&
              _CheckData(aMeGPU, answer, aUnitNum, 1e-4F) &&
              _CheckData(&bUserGPU, answer, aUnitNum, 1e-4F);

    /* destroy variables */
    delete a;
    delete b;
    delete aMe;
    delete aGPU;
    delete bGPU;
    delete aMeGPU;
    delete[] aDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete a;
    delete b;
    delete aMe;
    delete[] aDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* 
case 3: get the power(a, p)
In this case, p=0.
*/
bool TestPower3()
{
    /* a tensor of size (3, 2) */
    int aOrder = 2;
    int * aDimSize = new int[aOrder];
    aDimSize[0] = 3;
    aDimSize[1] = 2;

    int aUnitNum = 1;
    for (int i = 0; i < aOrder; i++)
        aUnitNum *= aDimSize[i];

    DTYPE aData[3][2] = { {1.0F, 1.0F},
                          {2.0F, 3.0F},
                          {4.0F, 5.0F} };
    DTYPE answer[3][2] = { {1.0F, 1.0F},
                           {1.0F, 1.0F},
                           {1.0F, 1.0F} };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * a = NewTensorV2(aOrder, aDimSize);
    XTensor * b = NewTensorV2(aOrder, aDimSize);
    XTensor * aMe = NewTensorV2(aOrder, aDimSize);
    XTensor bUser;

    /* initialize variables */
    a->SetData(aData, aUnitNum);
    aMe->SetData(aData, aUnitNum);

    /* call Power function */
    _Power(a, b, 0.0F);
    _PowerMe(aMe, 0.0F);
    bUser = Power(*a, 0.0F);

    /* check results */
    cpuTest = _CheckData(b, answer, aUnitNum, 1e-4F) &&
              _CheckData(aMe, answer, aUnitNum, 1e-4F) &&
              _CheckData(&bUser, answer, aUnitNum, 1e-4F);
    
#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensor */
    XTensor * aGPU = NewTensorV2(aOrder, aDimSize, X_FLOAT, 1.0F, 0);
    XTensor * bGPU = NewTensorV2(aOrder, aDimSize, X_FLOAT, 1.0F, 0);
    XTensor * aMeGPU = NewTensorV2(aOrder, aDimSize, X_FLOAT, 1.0F, 0);
    XTensor bUserGPU;

    /* Initialize variables */
    aGPU->SetData(aData, aUnitNum);
    aMeGPU->SetData(aData, aUnitNum);

    /* call Power function */
    _Power(aGPU, bGPU, 0.0F);
    _PowerMe(aMeGPU, 0.0F);
    bUserGPU = Power(*aGPU, 0.0F);

    /* check results */
    gpuTest = _CheckData(bGPU, answer, aUnitNum, 1e-4F) &&
              _CheckData(aMeGPU, answer, aUnitNum, 1e-4F) &&
              _CheckData(&bUserGPU, answer, aUnitNum, 1e-4F);

    /* destroy variables */
    delete a;
    delete b;
    delete aMe;
    delete aGPU;
    delete bGPU;
    delete aMeGPU;
    delete[] aDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete a;
    delete b;
    delete aMe;
    delete[] aDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* other cases */
/*
TODO!!
*/

/* test for Power Function */
bool TestPower()
{
    XPRINT(0, stdout, "[TEST POWER] get the power(a, p) \n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestPower1();

    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 1 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 1 passed!\n");

    /* case 2 test */
    caseFlag = TestPower2();

    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 2 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 2 passed!\n");

    /* case 3 test */
    caseFlag = TestPower3();

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
