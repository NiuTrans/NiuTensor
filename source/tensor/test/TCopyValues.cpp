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
* $Created by: Xu Chen (email: hello_master1954@163.com) 2018-06-27
*/

#include "../XUtility.h"
#include "../core/utilities/CheckData.h"
#include "TCopyValues.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* case 1: copy tensor s to tensor t */
bool TestCopyValues1()
{
    /* a input tensor of size (2, 4) */
    int sOrder = 2;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 2;
    sDimSize[1] = 4;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    DTYPE sData[2][4] = { {0.0F, 1.0F, 2.0F, 3.0F},
                          {4.0F, 5.0F, 6.0F, 7.0F} };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(sOrder, sDimSize);
    XTensor * t = NewTensorV2(sOrder, sDimSize);
    XTensor tUser;

    /* initialize variables */
    s->SetData(sData, sUnitNum);
    t->SetZeroAll();

    /* call CopyValues function */
    _CopyValues(s, t);
    tUser = CopyValues(*s);

    /* check results */
    cpuTest = _CheckData(t, sData, sUnitNum) && _CheckData(&tUser, sData, sUnitNum);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor tUserGPU;

    /* initialize variables */
    sGPU->SetData(sData, sUnitNum);
    tGPU->SetData(sData, sUnitNum);

    /* call CopyValues function */
    _CopyValues(sGPU, tGPU);
    tUserGPU = CopyValues(*sGPU);

    /* check results */
    gpuTest = _CheckData(tGPU, sData, sUnitNum) && _CheckData(&tUser, sData, sUnitNum);

    /* destroy variables */
    delete s;
    delete t;
    delete sGPU;
    delete tGPU;
    delete[] sDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete t;
    delete[] sDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* other cases */
/*
TODO!!
*/

/* test for CopyValues Function */
bool TestCopyValues()
{
    XPRINT(0, stdout, "[TEST CopyValues] copy tensor s to tensor t \n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestCopyValues1();
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
