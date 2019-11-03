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
* $Created by: Xu Chen (email: hello_master1954@163.com) 2018-07-12
*/

#include "../core/utilities/CheckData.h"
#include "../core/movement/CopyValues.h"
#include "TTranspose.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
case 1: test Transpose function.
tensor transposition of dimensions i and j 
*/
bool TestTranspose1()
{
    /* a tensor of size (3, 2) */
    int aOrder = 2;
    int * aDimSize = new int[aOrder];
    aDimSize[0] = 3;
    aDimSize[1] = 2;

    int aUnitNum = 1;
    for (int i = 0; i < aOrder; i++)
        aUnitNum *= aDimSize[i];

    /* a tensor of size (2, 3) */
    int bOrder = 2;
    int * bDimSize = new int[bOrder];
    bDimSize[0] = 2;
    bDimSize[1] = 3;

    int bUnitNum = 1;
    for (int i = 0; i < bOrder; i++)
        bUnitNum *= bDimSize[i];

    DTYPE aData[3][2] = { {1.0F, 2.0F}, 
                          {3.0F, 4.0F},
                          {5.0F, 6.0F} };
    DTYPE answer[2][3] = { {1.0F, 3.0F, 5.0F},
                           {2.0F, 4.0F, 6.0F} };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * a = NewTensorV2(aOrder, aDimSize);
    XTensor * b = NewTensorV2(bOrder, bDimSize);
    XTensor bUser;

    /* initialize variables */
    a->SetData(aData, aUnitNum);

    /* call Transpose function */
    _Transpose(a, b, 0, 1);
    bUser = Transpose(*a, 0, 1);

    /* check results */
    cpuTest = _CheckData(b, answer, aUnitNum, 1e-4F)
              && _CheckData(&bUser, answer, aUnitNum, 1e-4F);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensor */
    XTensor * aGPU = NewTensorV2(aOrder, aDimSize, X_FLOAT, 1.0F, 0);
    XTensor * bGPU = NewTensorV2(bOrder, bDimSize, X_FLOAT, 1.0F, 0);
    XTensor bUserGPU;

    /* Initialize variables */
    aGPU->SetData(aData, aUnitNum);

    /* call Transpose function */
    _Transpose(aGPU, bGPU, 0, 1);
    bUserGPU = Transpose(*aGPU, 0, 1);

    /* check results */
    gpuTest = _CheckData(bGPU, answer, aUnitNum, 1e-4F)
              && _CheckData(&bUserGPU, answer, aUnitNum, 1e-4F);

    /* destroy variables */
    delete a;
    delete b;
    delete aGPU;
    delete bGPU;
    delete[] aDimSize;
    delete[] bDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete a;
    delete b;
    delete[] aDimSize;
    delete[] bDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/*
case 2: test Transpose function.
tensor transposition of dimensions i and j 
*/
bool TestTranspose2()
{
    /* a tensor of size (4, 3, 2) */
    int aOrder = 3;
    int * aDimSize = new int[aOrder];
    aDimSize[0] = 4;
    aDimSize[1] = 3;
    aDimSize[2] = 2;

    int aUnitNum = 1;
    for (int i = 0; i < aOrder; i++)
        aUnitNum *= aDimSize[i];

    /* a tensor of size (2, 3, 4) */
    int bOrder = 3;
    int * bDimSize = new int[bOrder];
    bDimSize[0] = 2;
    bDimSize[1] = 3;
    bDimSize[2] = 4;

    int bUnitNum = 1;
    for (int i = 0; i < bOrder; i++)
        bUnitNum *= bDimSize[i];

    DTYPE aData[4][3][2] = { { {1.0F, 2.0F}, 
                               {3.0F, 4.0F},
                               {5.0F, 6.0F} },
                             { {2.0F, 4.0F}, 
                               {4.0F, 7.0F},
                               {6.0F, 8.0F} },
                             { {1.0F, 2.0F}, 
                               {3.0F, 4.0F},
                               {5.0F, 6.0F} },
                             { {2.0F, 4.0F}, 
                               {4.0F, 7.0F},
                               {6.0F, 8.0F} },};
    DTYPE answer[2][3][4] = { { {1.0F, 2.0F, 1.0F, 2.0F},
                                {2.0F, 4.0F, 2.0F, 4.0F},
                                {3.0F, 4.0F, 3.0F, 4.0F} },
                              { {4.0F, 7.0F, 4.0F, 7.0F},
                                {5.0F, 6.0F, 5.0F, 6.0F},
                                {6.0F, 8.0F, 6.0F, 8.0F} } };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * a = NewTensorV2(aOrder, aDimSize);
    XTensor * b = NewTensorV2(bOrder, bDimSize);
    XTensor bUser;

    /* initialize variables */
    a->SetData(aData, aUnitNum);

    /* call Transpose function */
    _Transpose(a, b, 0, 2);
    bUser = Transpose(*a, 0, 2);

    /* check results */
    cpuTest = _CheckData(b, answer, aUnitNum, 1e-4F)
              && _CheckData(&bUser, answer, aUnitNum, 1e-4F);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensor */
    XTensor * aGPU = NewTensorV2(aOrder, aDimSize, X_FLOAT, 1.0F, 0);
    XTensor * bGPU = NewTensorV2(bOrder, bDimSize, X_FLOAT, 1.0F, 0);
    XTensor bUserGPU;

    /* Initialize variables */
    aGPU->SetData(aData, aUnitNum);

    /* call Transpose function */
    _Transpose(aGPU, bGPU, 0, 2);
    bUserGPU = Transpose(*aGPU, 0, 2);

    /* check results */
    gpuTest = _CheckData(bGPU, answer, aUnitNum, 1e-4F)
              && _CheckData(&bUserGPU, answer, aUnitNum, 1e-4F);

    /* destroy variables */
    delete a;
    delete b;
    delete aGPU;
    delete bGPU;
    delete[] aDimSize;
    delete[] bDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete a;
    delete b;
    delete[] aDimSize;
    delete[] bDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* other cases */
/*
TODO!!
*/

/* test for Transpose Function */
bool TestTranspose()
{
    XPRINT(0, stdout, "[TEST TRANSPOSE] tensor transposition with specified dimensions \n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestTranspose1();

    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 1 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 1 passed!\n");
    
    /* case 2 test */
    caseFlag = TestTranspose2();

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
