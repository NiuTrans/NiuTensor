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
* $Created by: Xu Chen (email: hello_master1954@163.com) 2018-07-06
*/

#include "../core/utilities/CheckData.h"
#include "../core/utilities/SetAscendingOrder.h"
#include "TSetAscendingOrder.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* case 1: set the cell to the ascending order along a given dimension. */
bool TestSetAscendingOrder1()
{
    /* a input tensor of size (2, 4) */
    int sOrder = 2;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 2;
    sDimSize[1] = 4;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    int answer[2][4] = { {0, 1, 2, 3},
                         {0, 1, 2, 3} };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(sOrder, sDimSize, X_INT);

    /* initialize variables */
    s->SetZeroAll();

    /* call SetAscendingOrder function */
    SetAscendingOrder(*s, 1);
    
    /* check results */
    cpuTest = _CheckData(s, answer, sUnitNum);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensorV2(sOrder, sDimSize, X_INT, 1.0F, 0);

    /* initialize variables */
    sGPU->SetZeroAll();

    /* call SetAscendingOrder function */
    SetAscendingOrder(*sGPU, 1);

    /* check results */
    gpuTest = _CheckData(sGPU, answer, sUnitNum);

    /* destroy variables */
    delete s;
    delete sGPU;
    delete[] sDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete[] sDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* other cases */
/*
TODO!!
*/

/* test for SetAscendingOrder Function */
bool TestSetAscendingOrder()
{
    XPRINT(0, stdout, "[TEST SetAscendingOrder] set the cell to the ascending order along a given dimension \n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestSetAscendingOrder1();
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
