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

#include "../core/getandset/SetData.h"
#include "../core/movement/Spread.h"
#include "../core/utilities/CheckData.h"
#include "TSpread.h"

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
    XTensor * s = NewTensorV2(sOrder, sDimSize);
    XTensor * modify = NewTensorV2(dataOrder, dataDimSize);

    /* Initialize variables */
    _SetDataFixedFloat(s, 0.0F);
    modify->SetData(data, dataUnitNum);

    /* call _Spread function */
    _Spread(s, modify, 0, srcIndex, 2, tgtIndex);
    
    /* check results */
    cpuTest = _CheckData(s, answer, sUnitNum, 1e-5F);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * modifyGPU = NewTensorV2(dataOrder, dataDimSize, X_FLOAT, 1.0F, 0);

    /* Initialize variables */
    _SetDataFixedFloat(sGPU, 0.0F);
    modifyGPU->SetData(data, dataUnitNum);
    
    /* call _Spread function */
    _Spread(sGPU, modifyGPU, 0, srcIndex, 2, tgtIndex);
    
    gpuTest = _CheckData(sGPU, answer, sUnitNum, 1e-5F);

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
    XTensor * s1 = NewTensorV2(sOrder, sDimSize);
    XTensor * s2 = NewTensorV2(sOrder, sDimSize);
    XTensor * t = NewTensorV2(tOrder, tDimSize);
    XTensor * sIndex = NewTensorV2(indexOrder, indexDimSize, X_INT);
    XTensor * tIndex = NewTensorV2(indexOrder, indexDimSize, X_INT);

    /* initialize variables */
    s1->SetData(sData, sUnitNum);
    s2->SetData(sData, sUnitNum);
    t->SetData(tData, tUnitNum);
    sIndex->SetData(srcIndex, indexSize);
    tIndex->SetData(tgtIndex, indexSize);

    /* call _SpreadForGather function */
    _SpreadForCopyIndexed(s1, t, dim, sIndex, tIndex, 1);
    _SpreadForGather(s2, t, sIndex);

    /* check results */
    cpuTest = _CheckData(s1, answer, sUnitNum) &&
              _CheckData(s2, answer, sUnitNum);
    
#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU1 = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * sGPU2 = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU = NewTensorV2(sOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor * sIndexGPU = NewTensorV2(indexOrder, indexDimSize, X_INT, 1.0F, 0);
    XTensor * tIndexGPU = NewTensorV2(indexOrder, indexDimSize, X_INT, 1.0F, 0);

    /* initialize variables */
    sGPU1->SetData(sData, sUnitNum);
    sGPU2->SetData(sData, sUnitNum);
    tGPU->SetData(tData, tUnitNum);
    sIndexGPU->SetData(srcIndex, indexSize);
    tIndexGPU->SetData(tgtIndex, indexSize);

    /* call _SpreadForGather function */
    _SpreadForCopyIndexed(sGPU1, tGPU, dim, sIndexGPU, tIndexGPU, 1);
    _SpreadForGather(sGPU2, tGPU, sIndexGPU);

    /* check results */
    gpuTest = _CheckData(sGPU1, answer, sUnitNum) &&
              _CheckData(sGPU2, answer, sUnitNum);

    /* destroy variables */
    delete s1;
    delete s2;
    delete t;
    delete sIndex;
    delete tIndex;
    delete sGPU1;
    delete sGPU2;
    delete tGPU;
    delete sIndexGPU;
    delete tIndexGPU;
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
    delete tIndex;
    delete[] sDimSize;
    delete[] tDimSize;
    delete[] indexDimSize;

    return cpuTest;
#endif // USE_CUDA
}


/* 
case 3: test _SpreadForGather and _SpreadForCopyIndexed function 
spread a collection tensor to source tensor
*/
bool TestSpread3()
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
    tDimSize[0] = 3;
    tDimSize[1] = 2;

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

    DTYPE tData[3][2] = { {0.0F, -1.0F}, 
                          {2.0F, 1.0F},
                          {2.0F, 0.0F} };

    DTYPE answer[3][3] = { {-1.0F, 0.0F, 2.0F},
                           {3.0F, 1.0F, 5.0F},
                           {2.0F, 2.0F, 6.0F} };

    int dim = 1;
    int indexSize = 2;
    int srcIndex[2] = {0, 2};
    int tgtIndex[2] = {1, 0};

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s1 = NewTensorV2(sOrder, sDimSize);
    XTensor * s2 = NewTensorV2(sOrder, sDimSize);
    XTensor * t = NewTensorV2(tOrder, tDimSize);
    XTensor * sIndex = NewTensorV2(indexOrder, indexDimSize, X_INT);
    XTensor * tIndex = NewTensorV2(indexOrder, indexDimSize, X_INT);

    /* initialize variables */
    s1->SetData(sData, sUnitNum);
    s2->SetData(sData, sUnitNum);
    t->SetData(tData, tUnitNum);
    sIndex->SetData(srcIndex, indexSize);
    tIndex->SetData(tgtIndex, indexSize);

    /* call _SpreadForGather function */
    _SpreadForCopyIndexed(s1, t, dim, sIndex, tIndex, 1);
    _SpreadForCopyIndexed(s2, t, dim, sIndex, tIndex, 1);

    /* check results */
    cpuTest = _CheckData(s1, answer, sUnitNum) &&
              _CheckData(s2, answer, sUnitNum);
    
#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU1 = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * sGPU2 = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU = NewTensorV2(sOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor * sIndexGPU = NewTensorV2(indexOrder, indexDimSize, X_INT, 1.0F, 0);
    XTensor * tIndexGPU = NewTensorV2(indexOrder, indexDimSize, X_INT, 1.0F, 0);

    /* initialize variables */
    sGPU1->SetData(sData, sUnitNum);
    sGPU2->SetData(sData, sUnitNum);
    tGPU->SetData(tData, tUnitNum);
    sIndexGPU->SetData(srcIndex, indexSize);
    tIndexGPU->SetData(tgtIndex, indexSize);

    /* call _SpreadForGather function */
    _SpreadForCopyIndexed(sGPU1, tGPU, dim, sIndexGPU, tIndexGPU, 1);
    _SpreadForCopyIndexed(sGPU2, tGPU, dim, sIndexGPU, tIndexGPU, 1);

    /* check results */
    gpuTest = _CheckData(sGPU1, answer, sUnitNum) &&
              _CheckData(sGPU2, answer, sUnitNum);

    /* destroy variables */
    delete s1;
    delete s2;
    delete t;
    delete sIndex;
    delete tIndex;
    delete sGPU1;
    delete sGPU2;
    delete tGPU;
    delete sIndexGPU;
    delete tIndexGPU;
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
    delete tIndex;
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
        
    /* case 1 test */
    caseFlag = TestSpread2();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 2 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 2 passed!\n");
        
    /* case 1 test */
    caseFlag = TestSpread3();
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
