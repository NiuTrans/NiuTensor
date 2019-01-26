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

#include "TGather.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* 
case 1: gather indexed sub-tensors 
In this case, (3, 2, 3) -> (3, 2, 2), dim = 2, 
srcIndex = [0, 2], indexSize = 2
*/
bool TestGather1()
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

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensor(sOrder, sDimSize);
    XTensor * t = NewTensor(tOrder, tDimSize);

    /* initialize variables */
    s->SetData(sData, sUnitNum);
    t->SetZeroAll();

    /* call Gather function */
    _Gather(s, t, dim, srcIndex, indexSize);

    /* check results */
    cpuTest = t->CheckData(answer, tUnitNum);
    
#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensor(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU = NewTensor(sOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor tUserGPU;

    /* initialize variables */
    sGPU->SetData(sData, sUnitNum);
    tGPU->SetZeroAll();

    /* call Gather function */
    _Gather(sGPU, tGPU, dim, srcIndex, indexSize);

    /* check results */
    gpuTest = tGPU->CheckData(answer, tUnitNum);

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

/* 
case 2: gather indexed sub-tensors 
In this case, (3, 2, 3) -> (3, 1, 3), dim = 1, 
srcIndex = [0], indexSize = 1
*/
bool TestGather2()
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

    /* a output tensor of size (3, 1, 3) */
    int tOrder = 3;
    int * tDimSize = new int[tOrder];
    tDimSize[0] = 3;
    tDimSize[1] = 1;
    tDimSize[2] = 3;

    int tUnitNum = 1;
    for (int i = 0; i < tOrder; i++)
        tUnitNum *= tDimSize[i];

    DTYPE sData[3][2][3] = { { {0.0F, -1.0F, 2.0F},
                               {2.0F, 1.0F, 3.0F} },
                             { {1.0F, 2.0F, 4.0F}, 
                               {3.0F, 1.0F, 2.0F}},
                             { {-1.0F, 3.0F, 2.0F}, 
                               {1.0F, -1.0F, 0.0F} } };

    DTYPE answer[3][1][3] = { { {0.0F, -1.0F, 2.0F} },
                              { {1.0F, 2.0F, 4.0F} } , 
                              { {-1.0F, 3.0F, 2.0F} } };
    int dim = 1;
    int indexSize = 1;
    int srcIndex[2] = {0};

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensor(sOrder, sDimSize);
    XTensor * t = NewTensor(tOrder, tDimSize);

    /* initialize variables */
    s->SetData(sData, sUnitNum);
    t->SetZeroAll();

    /* call Gather function */
    _Gather(s, t, dim, srcIndex, indexSize);
    
    /* check results */
    cpuTest = t->CheckData(answer, tUnitNum);
    
#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensor(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU = NewTensor(sOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor tUserGPU;

    /* initialize variables */
    sGPU->SetData(sData, sUnitNum);
    tGPU->SetZeroAll();

    /* call Gather function */
    _Gather(sGPU, tGPU, dim, srcIndex, indexSize);

    /* check results */
    gpuTest = tGPU->CheckData(answer, tUnitNum);

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

/* 
case 3: gather indexed sub-tensors 
In this case, (3, 3) -> (2, 3), dim = 0, 
srcIndex = [0, 2]
*/
bool TestGather3()
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
    XTensor * s = NewTensor(sOrder, sDimSize);
    XTensor * t = NewTensor(tOrder, tDimSize);
    XTensor * index = NewTensor(indexOrder, indexDimSize, X_INT);
    XTensor tUser;

    /* initialize variables */
    s->SetData(sData, sUnitNum);
    t->SetZeroAll();
    index->SetData(srcIndex, indexSize);

    /* call Gather function */
    _Gather(s, t, dim, srcIndex, indexSize);
    tUser = Gather(*s, *index);

    /* check results */
    cpuTest = t->CheckData(answer, tUnitNum) &&
              tUser.CheckData(answer, tUnitNum);
    
#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensor(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU = NewTensor(sOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor * indexGPU = NewTensor(indexOrder, indexDimSize, X_INT, 1.0F, 0);
    XTensor tUserGPU;

    /* initialize variables */
    sGPU->SetData(sData, sUnitNum);
    tGPU->SetZeroAll();
    indexGPU->SetData(srcIndex, indexSize);

    /* call Gather function */
    _Gather(sGPU, tGPU, dim, srcIndex, indexSize);
    tUserGPU = Gather(*sGPU, *indexGPU);

    /* check results */
    gpuTest = tGPU->CheckData(answer, tUnitNum) && 
              tUserGPU.CheckData(answer, tUnitNum);

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
    
    /* case 2 test */
    caseFlag = TestGather2();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 2 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 2 passed!\n");
         
    /* case 2 test */
    caseFlag = TestGather3();
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
