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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-09-18
 */

#include "../core/math/Unary.h"
#include "../core/utilities/CheckData.h"
#include "TCos.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
case 1: test Cos function.
Set every entry to its cosine value.
*/
bool TestCos1()
{
    /* a tensor of size (3, 2) */
    int order = 2;
    int * dimSize = new int[order];
    dimSize[0] = 3;
    dimSize[1] = 2;

    int unitNum = 1;
    for (int i = 0; i < order; i++)
        unitNum *= dimSize[i];

    DTYPE aData[3][2] = { {1.0F, 2.0F}, 
                          {-1.0F, -2.0F},
                          {0.0F, 0.5F} };
    DTYPE answer[3][2] = { {0.5403F, -0.4161F},
                           {0.5403F, -0.4161F},
                           {1.0F, 0.8776F} };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * a = NewTensorV2(order, dimSize);
    XTensor * b = NewTensorV2(order, dimSize);
    XTensor * aMe = NewTensorV2(order, dimSize);
    XTensor bUser;

    /* initialize variables */
    a->SetData(aData, unitNum);
    aMe->SetData(aData, unitNum);

    /* call Cos function */
    _Cos(a, b);
    _CosMe(aMe);
    bUser = Cos(*a);

    /* check results */
    cpuTest = _CheckData(b, answer, unitNum, 1e-4F) && _CheckData(aMe, answer, unitNum, 1e-4F) && _CheckData(&bUser, answer, unitNum, 1e-4F);
    
#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensor */
    XTensor * aGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * bGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * aMeGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor bUserGPU;

    /* Initialize variables */
    aGPU->SetData(aData, unitNum);
    aMeGPU->SetData(aData, unitNum);

    /* call Cos function */
    _Cos(aGPU, bGPU);
    _CosMe(aMeGPU);
    bUserGPU = Cos(*aGPU);

    /* check results */
    gpuTest = _CheckData(bGPU, answer, unitNum, 1e-4F) && _CheckData(aMeGPU, answer, unitNum, 1e-4F) && _CheckData(&bUserGPU, answer, unitNum, 1e-4F);

    /* destroy variables */
    delete a;
    delete b;
    delete aMe;
    delete aGPU;
    delete bGPU;
    delete aMeGPU;
    delete[] dimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete a;
    delete b;
    delete aMe;
    delete[] dimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* other cases */
/*
TODO!!
*/

/* test for Cos Function */
bool TestCos()
{
    XPRINT(0, stdout, "[TEST Cos] set every entry to its cosine value \n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestCos1();

    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 1 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 1 passed!\n");

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
