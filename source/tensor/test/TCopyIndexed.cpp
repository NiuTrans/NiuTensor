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

#include "../core/utilities/CheckData.h"
#include "TCopyIndexed.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* 
case 1: copy indexed sub-tensors 
In this case, (3, 2, 3) -> (3, 2, 2), dim = 2, indexSize = 2, 
srcIndex = [0, 2], tgtIndex = [0, 1], copyNum = 1.
*/
bool TestCopyIndexed1()
{
    /* a input tensor of size (3, 2, 3) */
    int sOrder = 3;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 3;
    sDimSize[1] = 2;
    sDimSize[2] = 3;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    /* a output tensor of size (3, 2, 2) */
    int tOrder = 3;
    int * tDimSize = new int[tOrder];
    tDimSize[0] = 3;
    tDimSize[1] = 2;
    tDimSize[2] = 2;

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

    DTYPE sData[3][2][3] = { { {0.0F, -1.0F, 2.0F},
                               {2.0F, 1.0F, 3.0F} },
                             { {1.0F, 2.0F, 4.0F}, 
                               {3.0F, 1.0F, 2.0F}},
                             { {-1.0F, 3.0F, 2.0F}, 
                               {1.0F, -1.0F, 0.0F} } };

    DTYPE answer[3][2][2] = { { {0.0F, 2.0F},
                                {2.0F, 3.0F} },
                              { {1.0F, 4.0F}, 
                                {3.0F, 2.0F}},
                              { {-1.0F, 2.0F}, 
                                {1.0F, 0.0F} } };
    int dim = 2;
    int indexSize = 2;
    int srcIndex[2] = {0, 2};
    int tgtIndex[2] = {0, 1};
    int copyNum = 1;

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(sOrder, sDimSize);
    XTensor * t1 = NewTensorV2(tOrder, tDimSize);
    XTensor * t2 = NewTensorV2(tOrder, tDimSize);
    XTensor * sIndex = NewTensorV2(indexOrder, indexDimSize, X_INT);
    XTensor * tIndex = NewTensorV2(indexOrder, indexDimSize, X_INT);
    XTensor tUser;

    /* initialize variables */
    s->SetData(sData, sUnitNum);
    t1->SetZeroAll();
    t2->SetZeroAll();
    sIndex->SetData(srcIndex, indexUnitNum);
    tIndex->SetData(tgtIndex, indexUnitNum);

    /* call CopyIndexed function */
    _CopyIndexed(s, t1, dim, srcIndex, indexSize, tgtIndex, copyNum);
    _CopyIndexed(s, t2, dim, sIndex, tIndex, copyNum);
    tUser = CopyIndexed(*s, dim, *sIndex, *tIndex, copyNum);

    /* check results */
    cpuTest = _CheckData(t1, answer, tUnitNum) && 
              _CheckData(t2, answer, tUnitNum) &&
              _CheckData(&tUser, answer, tUnitNum);
    
#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU1 = NewTensorV2(sOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU2 = NewTensorV2(sOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor * sIndexGPU = NewTensorV2(indexOrder, indexDimSize, X_INT, 1.0F, 0);
    XTensor * tIndexGPU = NewTensorV2(indexOrder, indexDimSize, X_INT, 1.0F, 0);
    XTensor tUserGPU;

    /* initialize variables */
    sGPU->SetData(sData, sUnitNum);
    tGPU1->SetZeroAll();
    tGPU2->SetZeroAll();
    sIndexGPU->SetData(srcIndex, indexUnitNum);
    tIndexGPU->SetData(tgtIndex, indexUnitNum);

    /* call CopyIndexed function */
    _CopyIndexed(sGPU, tGPU1, dim, srcIndex, indexSize, tgtIndex, copyNum);
    _CopyIndexed(sGPU, tGPU2, dim, sIndexGPU, tIndexGPU, copyNum);
    tUserGPU = CopyIndexed(*sGPU, dim, *sIndexGPU, *tIndexGPU, copyNum);

    /* check results */
    gpuTest = _CheckData(tGPU1, answer, tUnitNum) &&
              _CheckData(tGPU2, answer, tUnitNum) &&
              _CheckData(&tUserGPU, answer, tUnitNum);

    /* destroy variables */
    delete s;
    delete t1;
    delete t2;
    delete sIndex;
    delete tIndex;
    delete sGPU;
    delete tGPU1;
    delete tGPU2;
    delete sIndexGPU;
    delete tIndexGPU;
    delete[] sDimSize;
    delete[] tDimSize;
    delete[] indexDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete t1;
    delete t2;
    delete sIndex;
    delete tIndex;
    delete[] sDimSize;
    delete[] tDimSize;
    delete[] indexDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* 
case 2: copy indexed sub-tensors 
In this case, (3, 2, 3) -> (3, 2, 2), dim = 2, indexSize = 2, 
srcIndex = [0, 2], tgtIndex = [1, 0], copyNum = 1.
*/
bool TestCopyIndexed2()
{
    /* a input tensor of size (3, 2, 3) */
    int sOrder = 3;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 3;
    sDimSize[1] = 2;
    sDimSize[2] = 3;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    /* a output tensor of size (3, 2, 2) */
    int tOrder = 3;
    int * tDimSize = new int[tOrder];
    tDimSize[0] = 3;
    tDimSize[1] = 2;
    tDimSize[2] = 2;

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

    DTYPE sData[3][2][3] = { { {0.0F, -1.0F, 2.0F},
                               {2.0F, 1.0F, 3.0F} },
                             { {1.0F, 2.0F, 4.0F}, 
                               {3.0F, 1.0F, 2.0F}},
                             { {-1.0F, 3.0F, 2.0F}, 
                               {1.0F, -1.0F, 0.0F} } };

    DTYPE answer[3][2][2] = { { {2.0F, 0.0F},
                                {3.0F, 2.0F} },
                              { {4.0F, 1.0F}, 
                                {2.0F, 3.0F}},
                              { {2.0F, -1.0F}, 
                                {0.0F, 1.0F} } };
    int dim = 2;
    int indexSize = 2;
    int srcIndex[2] = {0, 2};
    int tgtIndex[2] = {1, 0};
    int copyNum = 1;

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(sOrder, sDimSize);
    XTensor * t1 = NewTensorV2(tOrder, tDimSize);
    XTensor * t2 = NewTensorV2(tOrder, tDimSize);
    XTensor * sIndex = NewTensorV2(indexOrder, indexDimSize, X_INT);
    XTensor * tIndex = NewTensorV2(indexOrder, indexDimSize, X_INT);
    XTensor tUser;

    /* initialize variables */
    s->SetData(sData, sUnitNum);
    t1->SetZeroAll();
    t2->SetZeroAll();
    sIndex->SetData(srcIndex, indexUnitNum);
    tIndex->SetData(tgtIndex, indexUnitNum);

    /* call CopyIndexed function */
    _CopyIndexed(s, t1, dim, srcIndex, indexSize, tgtIndex, copyNum);
    _CopyIndexed(s, t2, dim, sIndex, tIndex, copyNum);
    tUser = CopyIndexed(*s, dim, *sIndex, *tIndex);
    
    /* check results */
    cpuTest = _CheckData(t1, answer, tUnitNum) &&
              _CheckData(t2, answer, tUnitNum) &&
              _CheckData(&tUser, answer, tUnitNum);
    
#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU1 = NewTensorV2(sOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU2 = NewTensorV2(sOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor * sIndexGPU = NewTensorV2(indexOrder, indexDimSize, X_INT, 1.0F, 0);
    XTensor * tIndexGPU = NewTensorV2(indexOrder, indexDimSize, X_INT, 1.0F, 0);
    XTensor tUserGPU;

    /* initialize variables */
    sGPU->SetData(sData, sUnitNum);
    tGPU1->SetZeroAll();
    tGPU2->SetZeroAll();
    sIndexGPU->SetData(srcIndex, indexUnitNum);
    tIndexGPU->SetData(tgtIndex, indexUnitNum);

    /* call CopyIndexed function */
    _CopyIndexed(sGPU, tGPU1, dim, srcIndex, indexSize, tgtIndex, copyNum);
    _CopyIndexed(sGPU, tGPU2, dim, sIndexGPU, tIndexGPU, copyNum);
    tUserGPU = CopyIndexed(*sGPU, dim, *sIndexGPU, *tIndexGPU, copyNum);

    /* check results */
    gpuTest = _CheckData(tGPU1, answer, tUnitNum) &&
              _CheckData(tGPU2, answer, tUnitNum) &&
              _CheckData(&tUserGPU, answer, tUnitNum);

    /* destroy variables */
    delete s;
    delete t1;
    delete t2;
    delete sIndex;
    delete tIndex;
    delete sGPU;
    delete tGPU1;
    delete tGPU2;
    delete sIndexGPU;
    delete tIndexGPU;
    delete[] sDimSize;
    delete[] tDimSize;
    delete[] indexDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete t1;
    delete t2;
    delete sIndex;
    delete tIndex;
    delete[] sDimSize;
    delete[] tDimSize;
    delete[] indexDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* 
case 3: copy indexed sub-tensors 
In this case, (3, 2, 3) -> (3, 2, 2), dim = 2, indexSize = 1, 
srcIndex = [0], tgtIndex = [0], copyNum = 2.
*/
bool TestCopyIndexed3()
{
    /* a input tensor of size (3, 2, 3) */
    int sOrder = 3;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 3;
    sDimSize[1] = 2;
    sDimSize[2] = 3;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    /* a output tensor of size (3, 2, 2) */
    int tOrder = 3;
    int * tDimSize = new int[tOrder];
    tDimSize[0] = 3;
    tDimSize[1] = 2;
    tDimSize[2] = 2;

    int tUnitNum = 1;
    for (int i = 0; i < tOrder; i++)
        tUnitNum *= tDimSize[i];
    
    /* a index tensor of size (1) */
    int indexOrder = 1;
    int * indexDimSize = new int[indexOrder];
    indexDimSize[0] = 1;

    int indexUnitNum = 1;
    for (int i = 0; i < indexOrder; i++)
        indexUnitNum *= indexDimSize[i];

    DTYPE sData[3][2][3] = { { {0.0F, -1.0F, 2.0F},
                               {2.0F, 1.0F, 3.0F} },
                             { {1.0F, 2.0F, 4.0F}, 
                               {3.0F, 1.0F, 2.0F}},
                             { {-1.0F, 3.0F, 2.0F}, 
                               {1.0F, -1.0F, 0.0F} } };

    DTYPE answer[3][2][2] = { { {0.0F, -1.0F},
                                {2.0F, 1.0F} },
                              { {1.0F, 2.0F}, 
                                {3.0F, 1.0F}},
                              { {-1.0F, 3.0F}, 
                                {1.0F, -1.0F} } };
    int dim = 2;
    int indexSize = 1;
    int srcIndex[1] = {0};
    int tgtIndex[1] = {0};
    int copyNum = 2;

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(sOrder, sDimSize);
    XTensor * t1 = NewTensorV2(tOrder, tDimSize);
    XTensor * t2 = NewTensorV2(tOrder, tDimSize);
    XTensor * sIndex = NewTensorV2(indexOrder, indexDimSize, X_INT);
    XTensor * tIndex = NewTensorV2(indexOrder, indexDimSize, X_INT);
    XTensor tUser;

    /* initialize variables */
    s->SetData(sData, sUnitNum);
    t1->SetZeroAll();
    t2->SetZeroAll();
    sIndex->SetData(srcIndex, indexUnitNum);
    tIndex->SetData(tgtIndex, indexUnitNum);

    /* call CopyIndexed function */
    _CopyIndexed(s, t1, dim, srcIndex, indexSize, tgtIndex, copyNum);
    _CopyIndexed(s, t2, dim, sIndex, tIndex, copyNum);
    tUser = CopyIndexed(*s, dim, *sIndex, *tIndex, copyNum);
    
    /* check results */
    cpuTest = _CheckData(t1, answer, tUnitNum) &&
              _CheckData(t2, answer, tUnitNum) &&
              _CheckData(&tUser, answer, tUnitNum);
    
#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU1 = NewTensorV2(sOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU2 = NewTensorV2(sOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor * sIndexGPU = NewTensorV2(indexOrder, indexDimSize, X_INT, 1.0F, 0);
    XTensor * tIndexGPU = NewTensorV2(indexOrder, indexDimSize, X_INT, 1.0F, 0);
    XTensor tUserGPU;

    /* initialize variables */
    sGPU->SetData(sData, sUnitNum);
    tGPU1->SetZeroAll();
    tGPU2->SetZeroAll();
    sIndexGPU->SetData(srcIndex, indexUnitNum);
    tIndexGPU->SetData(tgtIndex, indexUnitNum);

    /* call CopyIndexed function */
    _CopyIndexed(sGPU, tGPU1, dim, srcIndex, indexSize, tgtIndex, copyNum);
    _CopyIndexed(sGPU, tGPU2, dim, sIndexGPU, tIndexGPU, copyNum);
    tUserGPU = CopyIndexed(*sGPU, dim, *sIndexGPU, *tIndexGPU, copyNum);

    /* check results */
    gpuTest = _CheckData(tGPU1, answer, tUnitNum) &&
              _CheckData(tGPU2, answer, tUnitNum) &&
              _CheckData(&tUserGPU, answer, tUnitNum);

    /* destroy variables */
    delete s;
    delete t1;
    delete t2;
    delete sIndex;
    delete tIndex;
    delete sGPU;
    delete tGPU1;
    delete tGPU2;
    delete sIndexGPU;
    delete tIndexGPU;
    delete[] sDimSize;
    delete[] tDimSize;
    delete[] indexDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete t1;
    delete t2;
    delete sIndex;
    delete tIndex;
    delete[] sDimSize;
    delete[] tDimSize;
    delete[] indexDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* 
case 4: copy indexed sub-tensors 
In this case, (3, 2, 3) -> (3, 2, 2), dim = 2, indexSize = 2, 
srcIndex = [0, 2], tgtIndex = [0, 1], copyNum = 1.
*/
bool TestCopyIndexed4()
{
    /* a input tensor of size (3, 2, 3) */
    int sOrder = 3;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 3;
    sDimSize[1] = 2;
    sDimSize[2] = 3;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    /* a output tensor of size (3, 2, 2) */
    int tOrder = 3;
    int * tDimSize = new int[tOrder];
    tDimSize[0] = 3;
    tDimSize[1] = 2;
    tDimSize[2] = 2;

    int tUnitNum = 1;
    for (int i = 0; i < tOrder; i++)
        tUnitNum *= tDimSize[i];

    /* a index tensor of size(2) */
    int indexOrder = 1;
    int * indexDimSize = new int[indexOrder];
    indexDimSize[0] = 2;

    int indexUnitNum = 1;
    for (int i = 0; i < indexOrder; i++)
        indexUnitNum *= indexDimSize[i];

    DTYPE sData[3][2][3] = { { {0.0F, -1.0F, 2.0F},
                               {2.0F, 1.0F, 3.0F} },
                             { {1.0F, 2.0F, 4.0F}, 
                               {3.0F, 1.0F, 2.0F}},
                             { {-1.0F, 3.0F, 2.0F}, 
                               {1.0F, -1.0F, 0.0F} } };

    DTYPE answer[3][2][2] = { { {0.0F, 2.0F},
                                {2.0F, 3.0F} },
                              { {1.0F, 4.0F}, 
                                {3.0F, 2.0F}},
                              { {-1.0F, 2.0F}, 
                                {1.0F, 0.0F} } };
    int dim = 2;
    int indexSize = 2;
    int srcIndex[2] = {0, 2};
    int tgtIndex[2] = {0, 1};
    int copyNum = 1;

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(sOrder, sDimSize);
    XTensor * t1 = NewTensorV2(tOrder, tDimSize);
    XTensor * t2 = NewTensorV2(tOrder, tDimSize);
    XTensor * sIndex = NewTensorV2(indexOrder, indexDimSize, X_INT);
    XTensor * tIndex = NewTensorV2(indexOrder, indexDimSize, X_INT);
    XTensor tUser;

    /* initialize variables */
    s->SetData(sData, sUnitNum);
    t1->SetZeroAll();
    t2->SetZeroAll();
    sIndex->SetData(srcIndex, indexUnitNum);
    tIndex->SetData(tgtIndex, indexUnitNum);

    /* call CopyIndexed function */
    _CopyIndexed(s, t1, dim, srcIndex, indexSize, tgtIndex, copyNum);
    _CopyIndexed(s, t2, dim, sIndex, tIndex, copyNum);
    tUser = CopyIndexed(*s, dim, *sIndex, *tIndex, copyNum);

    /* check results */
    cpuTest = _CheckData(t1, answer, tUnitNum) && 
              _CheckData(t2, answer, tUnitNum) &&
              _CheckData(&tUser, answer, tUnitNum);
    
#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU1 = NewTensorV2(sOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU2 = NewTensorV2(sOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor * sIndexGPU = NewTensorV2(indexOrder, indexDimSize, X_INT, 1.0F, 0);
    XTensor * tIndexGPU = NewTensorV2(indexOrder, indexDimSize, X_INT, 1.0F, 0);
    XTensor tUserGPU;

    /* initialize variables */
    sGPU->SetData(sData, sUnitNum);
    tGPU1->SetZeroAll();
    tGPU2->SetZeroAll();
    sIndexGPU->SetData(srcIndex, indexUnitNum);
    tIndexGPU->SetData(tgtIndex, indexUnitNum);

    /* call CopyIndexed function */
    _CopyIndexed(sGPU, tGPU1, dim, srcIndex, indexSize, tgtIndex, copyNum);
    _CopyIndexed(sGPU, tGPU2, dim, sIndexGPU, tIndexGPU, copyNum);
    tUserGPU = CopyIndexed(*sGPU, dim, *sIndexGPU, *tIndexGPU, copyNum);

    /* check results */
    gpuTest = _CheckData(tGPU1, answer, tUnitNum) && 
              _CheckData(tGPU2, answer, tUnitNum) &&
              _CheckData(&tUserGPU, answer, tUnitNum);

    /* destroy variables */
    delete s;
    delete t1;
    delete t2;
    delete sIndex;
    delete tIndex;
    delete sGPU;
    delete tGPU1;
    delete tGPU2;
    delete sIndexGPU;
    delete tIndexGPU;
    delete[] sDimSize;
    delete[] tDimSize;
    delete[] indexDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete t1;
    delete t2;
    delete sIndex;
    delete tIndex;
    delete[] sDimSize;
    delete[] tDimSize;
    delete[] indexDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* 
case 5: copy indexed sub-tensors 
In this case, (3, 2, 3) -> (3, 2, 2), dim = 2, indexSize = 1, 
srcIndex = [0, 1], tgtIndex = [0, 2], copyNum = 2.
*/
bool TestCopyIndexed5()
{
    /* a input tensor of size (3, 2, 3) */
    int sOrder = 3;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 3;
    sDimSize[1] = 2;
    sDimSize[2] = 3;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    /* a output tensor of size (3, 2, 2) */
    int tOrder = 3;
    int * tDimSize = new int[tOrder];
    tDimSize[0] = 3;
    tDimSize[1] = 2;
    tDimSize[2] = 4;

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

    DTYPE sData[3][2][3] = { { {0.0F, -1.0F, 2.0F},
                               {2.0F, 1.0F, 3.0F} },
                             { {1.0F, 2.0F, 4.0F}, 
                               {3.0F, 1.0F, 2.0F}},
                             { {-1.0F, 3.0F, 2.0F}, 
                               {1.0F, -1.0F, 0.0F} } };

    DTYPE answer[3][2][4] = { { {0.0F, -1.0F, -1.0F, 2.0F},
                                {2.0F, 1.0F, 1.0F, 3.0F} },
                              { {1.0F, 2.0F, 2.0F, 4.0F}, 
                                {3.0F, 1.0F, 1.0F, 2.0F}},
                              { {-1.0F, 3.0F, 3.0F, 2.0F}, 
                                {1.0F, -1.0F, -1.0F, 0.0F} } };
    int dim = 2;
    int indexSize = 2;
    int srcIndex[2] = {0, 1};
    int tgtIndex[2] = {0, 2};
    int copyNum = 2;

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(sOrder, sDimSize);
    XTensor * t1 = NewTensorV2(tOrder, tDimSize);
    XTensor * t2 = NewTensorV2(tOrder, tDimSize);
    XTensor * sIndex = NewTensorV2(indexOrder, indexDimSize, X_INT);
    XTensor * tIndex = NewTensorV2(indexOrder, indexDimSize, X_INT);
    XTensor tUser;

    /* initialize variables */
    s->SetData(sData, sUnitNum);
    t1->SetZeroAll();
    t2->SetZeroAll();
    sIndex->SetData(srcIndex, indexUnitNum);
    tIndex->SetData(tgtIndex, indexUnitNum);

    /* call CopyIndexed function */
    _CopyIndexed(s, t1, dim, srcIndex, indexSize, tgtIndex, copyNum);
    _CopyIndexed(s, t2, dim, sIndex, tIndex, copyNum);
    tUser = CopyIndexed(*s, dim, *sIndex, *tIndex, copyNum);

    /* check results */
    cpuTest = _CheckData(t1, answer, tUnitNum) &&
              _CheckData(t2, answer, tUnitNum) &&
              _CheckData(&tUser, answer, tUnitNum);
    
#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU1 = NewTensorV2(sOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU2 = NewTensorV2(sOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor * sIndexGPU = NewTensorV2(indexOrder, indexDimSize, X_INT, 1.0F, 0);
    XTensor * tIndexGPU = NewTensorV2(indexOrder, indexDimSize, X_INT, 1.0F, 0);
    XTensor tUserGPU;

    /* initialize variables */
    sGPU->SetData(sData, sUnitNum);
    tGPU1->SetZeroAll();
    tGPU2->SetZeroAll();
    sIndexGPU->SetData(srcIndex, indexUnitNum);
    tIndexGPU->SetData(tgtIndex, indexUnitNum);

    /* call CopyIndexed function */
    _CopyIndexed(sGPU, tGPU1, dim, srcIndex, indexSize, tgtIndex, copyNum);
    _CopyIndexed(sGPU, tGPU2, dim, sIndexGPU, tIndexGPU, copyNum);
    tUserGPU = CopyIndexed(*sGPU, dim, *sIndexGPU, *tIndexGPU, copyNum);

    /* check results */
    gpuTest = _CheckData(tGPU1, answer, tUnitNum) &&
              _CheckData(tGPU2, answer, tUnitNum) &&
              _CheckData(&tUserGPU, answer, tUnitNum);

    /* destroy variables */
    delete s;
    delete t1;
    delete t2;
    delete sIndex;
    delete tIndex;
    delete sGPU;
    delete tGPU1;
    delete tGPU2;
    delete sIndexGPU;
    delete tIndexGPU;
    delete[] sDimSize;
    delete[] tDimSize;
    delete[] indexDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete t1;
    delete t2;
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

/* test for CopyIndexed Function */
bool TestCopyIndexed()
{
    XPRINT(0, stdout, "[TEST CopyIndexed] copy indexed sub-tensors \n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestCopyIndexed1();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 1 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 1 passed!\n");
    
    /* case 2 test */
    caseFlag = TestCopyIndexed2();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 2 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 2 passed!\n");
        
    /* case 3 test */
    caseFlag = TestCopyIndexed3();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 3 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 3 passed!\n");
            
    /* case 4 test */
    caseFlag = TestCopyIndexed4();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 4 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 4 passed!\n");

    /* case 5 test */
    caseFlag = TestCopyIndexed5();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 5 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 5 passed!\n");

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
