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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-09-25
 */

#include "TSpread.h"
#include "../core/getandset/SetData.h"
#include "../core/movement/Spread.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
case 1: test _Spread function.
spread a collection tensor to source tensor.
*/
bool TestSpread1()
{
    /* a input tensor of size (2, 4, 3) */
    int sOrder = 3;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 4;
    sDimSize[1] = 4;
    sDimSize[2] = 3;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];
    
    /* a data tensor of size (2, 4, 3) */
    int dataOrder = 3;
    int * dataDimSize = new int[dataOrder];
    dataDimSize[0] = 2;
    dataDimSize[1] = 4;
    dataDimSize[2] = 3;

    int dataUnitNum = 1;
    for (int i = 0; i < dataOrder; i++)
        dataUnitNum *= dataDimSize[i];
    
    int srcIndex[2] = {0, 1};
    int tgtIndex[2] = {0, 1};


    DTYPE data[2][4][3] = { { {1.0F, 1.0F, 1.0F},
                              {0.0F, 1.0F, 2.0F},
                              {1.0F, 1.0F, 1.0F},
                              {1.0F, 1.0F, 1.0F} },
                            { {1.0F, 1.0F, 1.0F},
                              {3.0F, 4.0F, 5.0F},
                              {1.0F, 1.0F, 1.0F},
                              {1.0F, 1.0F, 1.0F} } };

    DTYPE answer[4][4][3] = { { {1.0F, 1.0F, 1.0F},
                                {0.0F, 1.0F, 2.0F},
                                {1.0F, 1.0F, 1.0F},
                                {1.0F, 1.0F, 1.0F} },
                              { {1.0F, 1.0F, 1.0F},
                                {3.0F, 4.0F, 5.0F},
                                {1.0F, 1.0F, 1.0F},
                                {1.0F, 1.0F, 1.0F} },
                              { {0.0F, 0.0F, 0.0F}, 
                                {0.0F, 0.0F, 0.0F}, 
                                {0.0F, 0.0F, 0.0F} },
                              { {0.0F, 0.0F, 0.0F}, 
                                {0.0F, 0.0F, 0.0F}, 
                                {0.0F, 0.0F, 0.0F} },
    };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensor(sOrder, sDimSize);
    XTensor * modify = NewTensor(dataOrder, dataDimSize);

    /* Initialize variables */
    _SetDataFixedFloat(s, 0.0F);
    modify->SetData(data, dataUnitNum);

    /* call _Spread function */
    _Spread(s, modify, 0, srcIndex, 2, tgtIndex);
    
    /* check results */
    cpuTest = s->CheckData(answer, sUnitNum, 1e-5F);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensor(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * modifyGPU = NewTensor(dataOrder, dataDimSize, X_FLOAT, 1.0F, 0);

    /* Initialize variables */
    _SetDataFixedFloat(sGPU, 0.0F);
    modifyGPU->SetData(data, dataUnitNum);
    
    /* call _Spread function */
    _Spread(sGPU, modifyGPU, 0, srcIndex, 2, tgtIndex);
    
    gpuTest = sGPU->CheckData(answer, sUnitNum, 1e-5F);

    /* destroy variables */
    delete s;
    delete modify;
    delete sGPU;
    delete modifyGPU;
    delete[] sDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete[] sDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* 
case 2: test _SpreadForGather function 
spread a collection tensor to source tensor
*/
bool TestSpread2()
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

    DTYPE sData[3][3] = { {0.0F, 0.0F, 2.0F},
                          {2.0F, 1.0F, 3.0F},
                          {2.0F, 2.0F, 4.0F} };

    DTYPE tData[2][3] = { {0.0F, -1.0F, 2.0F},
                          {1.0F, 2.0F, 0.0F} };

    DTYPE answer[3][3] = { {0.0F, -1.0F, 4.0F},
                           {2.0F, 1.0F, 3.0F},
                           {3.0F, 4.0F, 4.0F} };

    int dim = 0;
    int indexSize = 2;
    int srcIndex[2] = {0, 2};
    int tgtIndex[2] = {0, 1};

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s1 = NewTensor(sOrder, sDimSize);
    XTensor * s2 = NewTensor(sOrder, sDimSize);
    XTensor * t = NewTensor(tOrder, tDimSize);
    XTensor * sIndex = NewTensor(indexOrder, indexDimSize, X_INT);
    XTensor * cIndex = NewTensor(indexOrder, indexDimSize, X_INT);

    /* initialize variables */
    s1->SetData(sData, sUnitNum);
    s2->SetData(sData, sUnitNum);
    t->SetData(tData, tUnitNum);
    sIndex->SetData(srcIndex, indexSize);
    cIndex->SetData(tgtIndex, indexSize);

    /* call _SpreadForGather function */
    _SpreadForCopyIndexed(s1, t, dim, sIndex, cIndex, 1);
    _SpreadForGather(s2, t, sIndex);

    /* check results */
    cpuTest = s1->CheckData(answer, tUnitNum) &&
              s2->CheckData(answer, tUnitNum);
    
#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU1 = NewTensor(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * sGPU2 = NewTensor(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU = NewTensor(sOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor * sIndexGPU = NewTensor(indexOrder, indexDimSize, X_INT, 1.0F, 0);
    XTensor * cIndexGPU = NewTensor(indexOrder, indexDimSize, X_INT, 1.0F, 0);

    /* initialize variables */
    sGPU1->SetData(sData, sUnitNum);
    sGPU2->SetData(sData, sUnitNum);
    tGPU->SetData(tData, tUnitNum);
    sIndexGPU->SetData(srcIndex, indexSize);
    cIndexGPU->SetData(tgtIndex, indexSize);

    /* call _SpreadForGather function */
    _SpreadForCopyIndexed(sGPU1, tGPU, dim, sIndex, cIndex, 1);
    _SpreadForGather(sGPU2, tGPU, sIndexGPU);

    /* check results */
    gpuTest = sGPU1->CheckData(answer, tUnitNum) && 
              sGPU2->CheckData(answer, tUnitNum);

    /* destroy variables */
    delete s1;
    delete s2;
    delete t;
    delete sIndex;
    delete cIndex;
    delete sGPU1;
    delete sGPU2;
    delete tGPU;
    delete sIndexGPU;
    delete cIndexGPU;
    delete[] sDimSize;
    delete[] tDimSize;
    delete[] indexDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s1;
    delete s2;
    delete t;
    delete sIndex;
    delete cIndex;
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

/* test for Spread Function */
bool TestSpread()
{
    XPRINT(0, stdout, "[TEST Spread] spread a collection tensor to source tensor \n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestSpread1();
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
