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
* $Created by: LI Yinqiao (li.yin.qiao.2012@hotmail.com) 2018-04-30
*/

#include "../core/utilities/CheckData.h"
#include "TSort.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* case 1: sort the tensor along a given dimension */
bool TestSort1()
{
    /* a tensor of size (2, 4) */
    int order = 2;
    int * dimSize = new int[order];
    dimSize[0] = 2;
    dimSize[1] = 4;

    int unitNum = 1;
    for (int i = 0; i < order; i++)
        unitNum *= dimSize[i];

    DTYPE aData[2][4] = { {0.0F, 1.0F, 2.0F, 3.0F},
                          {4.0F, 5.0F, 6.0F, 7.0F} };
    DTYPE answer[2][4] = { {4.0F, 5.0F, 6.0F, 7.0F},
                           {0.0F, 1.0F, 2.0F, 3.0F} };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * a = NewTensorV2(order, dimSize);
    XTensor * b = NewTensorV2(order, dimSize);
    XTensor * aMe = NewTensorV2(order, dimSize);
    XTensor * index = NewTensorV2(order, dimSize, X_INT);
    XTensor bUser(order, dimSize, X_FLOAT, 1.0F, -1, NULL);

    /* initialize variables */
    a->SetData(aData, unitNum);
    aMe->SetData(aData, unitNum);
    index->SetZeroAll();

    /* call Sort function */
    _Sort(a, b, index, 0);
    _SortMe(aMe, index, 0);
    Sort(*a, bUser, *index, 0);

    cpuTest = _CheckData(b, answer, unitNum) &&
              _CheckData(aMe, answer, unitNum) &&
              _CheckData(&bUser, answer, unitNum);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensor */
    XTensor * aGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * bGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * aMeGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * indexGPU = NewTensorV2(order, dimSize, X_INT, 1.0F, 0);
    XTensor bUserGPU(order, dimSize, X_FLOAT, 1.0F, 0, NULL);

    /* Initialize variables */
    aGPU->SetData(aData, unitNum);
    aMeGPU->SetData(aData, unitNum);
    indexGPU->SetZeroAll();

    /* call sum function */
    _Sort(aGPU, bGPU, indexGPU, 0);
    _SortMe(aMeGPU, indexGPU, 0);
    Sort(*aGPU, bUserGPU, *indexGPU, 0);

    /* check results */
    gpuTest = _CheckData(bGPU, answer, unitNum) &&
              _CheckData(aMeGPU, answer, unitNum) &&
              _CheckData(&bUserGPU, answer, unitNum);

    /* destroy variables */
    delete a;
    delete b;
    delete aMe;
    delete index;
    delete aGPU;
    delete bGPU;
    delete aMeGPU;
    delete indexGPU;
    delete[] dimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete a;
    delete b;
    delete aMe;
    delete index;
    delete[] dimSize;

    return cpuTest;
#endif // USE_CUDA
}

bool TestSort2()
{
    /* a tensor of size (2, 4) */
    int order = 2;
    int * dimSize = new int[order];
    dimSize[0] = 2;
    dimSize[1] = 4;

    int unitNum = 1;
    for (int i = 0; i < order; i++)
        unitNum *= dimSize[i];

    DTYPE aData[2][4] = { {0.0F, 1.0F, 2.0F, 3.0F},
                          {4.0F, 5.0F, 6.0F, 7.0F} };
    DTYPE answer[2][4] = { {3.0F, 2.0F, 1.0F, 0.0F},
                           {7.0F, 6.0F, 5.0F, 4.0F} };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * a = NewTensorV2(order, dimSize);
    XTensor * b = NewTensorV2(order, dimSize);
    XTensor * aMe = NewTensorV2(order, dimSize);
    XTensor * index = NewTensorV2(order, dimSize, X_INT);
    XTensor bUser(order, dimSize, X_FLOAT, 1.0F, -1, NULL);

    /* initialize variables */
    a->SetData(aData, unitNum);
    aMe->SetData(aData, unitNum);
    index->SetZeroAll();
    
    /* call Sort function */
    _Sort(a, b, index, 1);
    _SortMe(aMe, index, 1);
    Sort(*a, bUser, *index, 1);

    /* check results */
    cpuTest = _CheckData(b, answer, unitNum) &&
              _CheckData(aMe, answer, unitNum) &&
              _CheckData(&bUser, answer, unitNum);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensor */
    XTensor * aGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * bGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * aMeGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * indexGPU = NewTensorV2(order, dimSize, X_INT, 1.0F, 0);
    XTensor bUserGPU(order, dimSize, X_FLOAT, 1.0F, 0, NULL);

    /* Initialize variables */
    aGPU->SetData(aData, unitNum);
    aMeGPU->SetData(aData, unitNum);
    indexGPU->SetZeroAll();

    /* call sum function */
    _Sort(aGPU, bGPU, indexGPU, 1);
    _SortMe(aMeGPU, indexGPU, 1);
    Sort(*aGPU, bUserGPU, *indexGPU, 1);

    /* check results */
    gpuTest = _CheckData(bGPU, answer, unitNum) &&
              _CheckData(aMeGPU, answer, unitNum) &&
              _CheckData(&bUserGPU, answer, unitNum);

    /* destroy variables */
    delete a;
    delete b;
    delete aMe;
    delete index;
    delete aGPU;
    delete bGPU;
    delete aMeGPU;
    delete indexGPU;
    delete[] dimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete a;
    delete b;
    delete aMe;
    delete index;
    delete[] dimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* other cases */
/*
TODO!!
*/

/* test for Sort Function */
bool TestSort()
{
    XPRINT(0, stdout, "[TEST SORT] sort the tensor along a given dimension \n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestSort1();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 1 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 1 passed!\n");

    /* case 2 test */
    caseFlag = TestSort2();
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
