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

#include "../core/utilities/CheckData.h"
#include "TGather.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* 
case 1: gather indexed sub-tensors 
In this case, (3, 3) -> (2, 3), dim = 0, 
srcIndex = [0, 2]
*/
bool TestGather1()
{
    /* a input tensor of size (3, 3) */
    int sOrder = 2;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 3;
    sDimSize[1] = 3;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    /* a output tensor of size (2, 3) */
    int tOrder = 2;
    int * tDimSize = new int[tOrder];
    tDimSize[0] = 2;
    tDimSize[1] = 3;

    int tUnitNum = 1;
    for (int i = 0; i < tOrder; i++)
        tUnitNum *= tDimSize[i];
        
    /* a index tensor of size (2) */
    int indexOrder = 1;
    int * indexDimSize = new int[indexOrder];
    indexDimSize[0] = 2;

    int indexUnitNum = 1;
    for (int i = 0; i < indexOrder; i++)
        indexUnitNum *= indexDimSize[i];

    DTYPE sData[3][3] = { {0.0F, -1.0F, 2.0F},
                          {2.0F, 1.0F, 3.0F},
                          {1.0F, 2.0F, 4.0F} };

    DTYPE answer[2][3] = { {0.0F, -1.0F, 2.0F},
                           {1.0F, 2.0F, 4.0F} };

    int dim = 0;
    int indexSize = 2;
    int srcIndex[2] = {0, 2};

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(sOrder, sDimSize);
    XTensor * t = NewTensorV2(tOrder, tDimSize);
    XTensor * index = NewTensorV2(indexOrder, indexDimSize, X_INT);
    XTensor tUser;

    /* initialize variables */
    s->SetData(sData, sUnitNum);
    t->SetZeroAll();
    index->SetData(srcIndex, indexSize);

    /* call Gather function */
    _Gather(s, t, index);
    tUser = Gather(*s, *index);

    /* check results */
    cpuTest = _CheckData(t, answer, tUnitNum) &&
              _CheckData(&tUser, answer, tUnitNum);
    
#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU = NewTensorV2(sOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor * indexGPU = NewTensorV2(indexOrder, indexDimSize, X_INT, 1.0F, 0);
    XTensor tUserGPU;

    /* initialize variables */
    sGPU->SetData(sData, sUnitNum);
    tGPU->SetZeroAll();
    indexGPU->SetData(srcIndex, indexSize);

    /* call Gather function */
    _Gather(sGPU, tGPU, indexGPU);
    tUserGPU = Gather(*sGPU, *indexGPU);

    /* check results */
    gpuTest = _CheckData(tGPU, answer, tUnitNum) &&
              _CheckData(&tUserGPU, answer, tUnitNum);

    /* destroy variables */
    delete s;
    delete t;
    delete index;
    delete sGPU;
    delete tGPU;
    delete indexGPU;
    delete[] sDimSize;
    delete[] tDimSize;
    delete[] indexDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete t;
    delete index;
    delete[] sDimSize;
    delete[] tDimSize;
    delete[] indexDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* other cases */
/*
TODO!!
*/

/* test for Gather Function */
bool TestGather()
{
    XPRINT(0, stdout, "[TEST Gather] gather indexed sub-tensors \n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestGather1();
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
