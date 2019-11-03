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
* $Created by: Xu Chen (email: hello_master1954@163.com) 2018-07-04
*/

#include "../core/utilities/CheckData.h"
#include "TSelect.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* 
case 1: test SelectRange function.
It can generate a tensor with seleccted data in range[low,high] along the given dimension.
In this case, (2, 2, 4) -> (2, 2, 2), dim = 2, low = 1, high = 3.
*/
bool TestSelect1()
{
    /* a input tensor of size (2, 2, 4) */
    int sOrder = 3;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 2;
    sDimSize[1] = 2;
    sDimSize[2] = 4;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    /* a output tensor of size (2, 2, 2) */
    int tOrder = 3;
    int * tDimSize = new int[tOrder];
    tDimSize[0] = 2;
    tDimSize[1] = 2;
    tDimSize[2] = 2;

    int tUnitNum = 1;
    for (int i = 0; i < tOrder; i++)
        tUnitNum *= tDimSize[i];

    DTYPE sData[2][2][4] = { { {0.0F, 1.0F, 2.0F, 3.0F},
                               {4.0F, 5.0F, 6.0F, 7.0F} },
                             { {1.0F, 2.0F, 3.0F, 4.0F},
                               {5.0F, 6.0F, 7.0F, 8.0F} } };
    DTYPE answer[2][2][2] = { { {1.0F, 2.0F},
                                {5.0F, 6.0F} },
                              { {2.0F, 3.0F},
                                {6.0F, 7.0F} } };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(sOrder, sDimSize);
    XTensor * t = NewTensorV2(tOrder, tDimSize);
    XTensor tUser;

    /* initialize variables */
    s->SetData(sData, sUnitNum);
    t->SetZeroAll();

    /* call SelectRange function */
    _SelectRange(s, t, 2, 1, 3);
    tUser = SelectRange(*s, 2, 1, 3);

    /* check results */
    cpuTest = _CheckData(t, answer, tUnitNum) && _CheckData(&tUser, answer, tUnitNum);
    
#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU = NewTensorV2(tOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor tUserGPU;

    /* initialize variables */
    sGPU->SetData(sData, sUnitNum);
    tGPU->SetZeroAll();

    /* call SelectRange function */
    _SelectRange(sGPU, tGPU, 2, 1, 3);
    tUserGPU = SelectRange(*sGPU, 2, 1, 3);

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

/* other cases */
/*
TODO!!
*/

/* test for Select Function */
bool TestSelect()
{
    XPRINT(0, stdout, "[TEST Select] generate a tensor with seleccted data in range[low,high] along the given dimension \n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestSelect1();
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
