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

#include "TTopK.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* 
case 1: get the top-k items along a given dimension.
In this case, 
(2, 4) -> (2, 4), dim = 0, k = 2
(2, 4) -> (2, 4), dim = 1, k = 4
*/
bool TestTopK1()
{
    /* a input tensor of size (2, 4) */
    int sOrder = 2;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 2;
    sDimSize[1] = 4;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    /* a output tensor of size (2, 4) */
    int tOrder = 2;
    int * tDimSize = new int[tOrder];
    tDimSize[0] = 2;
    tDimSize[1] = 4;

    int tUnitNum = 1;
    for (int i = 0; i < tOrder; i++)
        tUnitNum *= tDimSize[i];

    DTYPE sData[2][4] = { {5.0F, 1.0F, 2.0F, 8.0F},
                          {4.0F, 3.0F, 7.0F, 6.0F} };

    DTYPE tAnswer1[2][4] = { {5.0F, 3.0F, 7.0F, 8.0F},
                             {4.0F, 1.0F, 2.0F, 6.0F} };
    int indexAnswer1[2][4] = { {0, 1, 1, 0},
                               {1, 0, 0, 1} };

    DTYPE tAnswer2[2][4] = { {8.0F, 5.0F, 2.0F, 1.0F},
                             {7.0F, 6.0F, 4.0F, 3.0F} };
    int indexAnswer2[2][4] = { {3, 0, 2, 1},
                               {2, 3, 0, 1} };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensor(sOrder, sDimSize);
    XTensor * t1 = NewTensor(tOrder, tDimSize);
    XTensor * t2 = NewTensor(tOrder, tDimSize);
    XTensor * index1 = NewTensor(tOrder, tDimSize, X_INT);
    XTensor * index2 = NewTensor(tOrder, tDimSize, X_INT);

    XTensor sUser = XTensor(sOrder, sDimSize, X_FLOAT, 1.0F, -1, NULL);
    XTensor tUser1 = XTensor(tOrder, tDimSize, X_FLOAT, 1.0F, -1, NULL);
    XTensor tUser2 = XTensor(tOrder, tDimSize, X_FLOAT, 1.0F, -1, NULL);
    XTensor indexUser1 = NewTensor(tOrder, tDimSize, X_INT, 1.0F, -1, NULL);
    XTensor indexUser2 = NewTensor(tOrder, tDimSize, X_INT, 1.0F, -1, NULL);

    /* initialize variables */
    s->SetData(sData, sUnitNum);
    t1->SetZeroAll();
    t2->SetZeroAll();
    index1->SetZeroAll();
    index2->SetZeroAll();

    sUser.SetData(sData, sUnitNum);
    tUser1.SetZeroAll();
    tUser2.SetZeroAll();
    indexUser1.SetZeroAll();
    indexUser2.SetZeroAll();

    /* call TopK function */
    int dim = 0;
    int k = sDimSize[dim];
    _TopK(s, t1, index1, dim, k);
    TopK(sUser, tUser1, indexUser1, dim, k);

    dim = 1;
    k = sDimSize[dim];
    _TopK(s, t2, index2, dim, k);
    TopK(sUser, tUser2, indexUser2, dim, k);

    /* check results */
    cpuTest = t1->CheckData(tAnswer1, tUnitNum) && tUser1.CheckData(tAnswer1, tUnitNum)
           && t2->CheckData(tAnswer2, tUnitNum) && tUser2.CheckData(tAnswer2, tUnitNum)
           && index1->CheckData(indexAnswer1, tUnitNum) && indexUser1.CheckData(indexAnswer1, tUnitNum)
           && index2->CheckData(indexAnswer2, tUnitNum) && indexUser2.CheckData(indexAnswer2, tUnitNum);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensor(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU1 = NewTensor(tOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU2 = NewTensor(tOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor * indexGPU1 = NewTensor(tOrder, tDimSize, X_INT, 1.0F, 0);
    XTensor * indexGPU2 = NewTensor(tOrder, tDimSize, X_INT, 1.0F, 0);
    
    XTensor sUserGPU = XTensor(sOrder, sDimSize, X_FLOAT, 1.0F, 0, NULL);
    XTensor tUserGPU1 = XTensor(tOrder, tDimSize, X_FLOAT, 1.0F, 0, NULL);
    XTensor tUserGPU2 = XTensor(tOrder, tDimSize, X_FLOAT, 1.0F, 0, NULL);
    XTensor indexUserGPU1 = NewTensor(tOrder, tDimSize, X_INT, 1.0F, 0, NULL);
    XTensor indexUserGPU2 = NewTensor(tOrder, tDimSize, X_INT, 1.0F, 0, NULL);

    /* initialize variables */
    sGPU->SetData(sData, sUnitNum);
    tGPU1->SetZeroAll();
    tGPU2->SetZeroAll();
    indexGPU1->SetZeroAll();
    indexGPU2->SetZeroAll();

    sUserGPU.SetData(sData, sUnitNum);
    tUserGPU1.SetZeroAll();
    tUserGPU2.SetZeroAll();
    indexUserGPU1.SetZeroAll();
    indexUserGPU2.SetZeroAll();

    /* call TopK function */
    dim = 0;
    k = sDimSize[dim];
    _TopK(sGPU, tGPU1, indexGPU1, dim, k);
    TopK(sUserGPU, tUserGPU1, indexUserGPU1, dim, k);
    
    dim = 1;
    k = sDimSize[dim];
    _TopK(sGPU, tGPU2, indexGPU2, dim, k);
    TopK(sUserGPU, tUserGPU2, indexUserGPU2, dim, k);
    
    /* check results */
    gpuTest = tGPU1->CheckData(tAnswer1, tUnitNum) && tUserGPU1.CheckData(tAnswer1, tUnitNum)
              && tGPU2->CheckData(tAnswer2, tUnitNum) && tUserGPU2.CheckData(tAnswer2, tUnitNum)
              && indexGPU1->CheckData(indexAnswer1, tUnitNum) && indexUserGPU1.CheckData(indexAnswer1, tUnitNum)
              && indexGPU2->CheckData(indexAnswer2, tUnitNum) && indexUserGPU2.CheckData(indexAnswer2, tUnitNum);

    /* destroy variables */
    delete s;
    delete t1;
    delete t2;
    delete index1;
    delete index2;
    delete sGPU;
    delete tGPU1;
    delete tGPU2;
    delete indexGPU1;
    delete indexGPU2;
    delete[] sDimSize;
    delete[] tDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete t1;
    delete t2;
    delete index1;
    delete index2;
    delete[] sDimSize;
    delete[] tDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/*
case 2: get the top-k items along a given dimension.
In this case, (2, 4) -> (2, 2), dim = 1, k = 2.
*/
bool TestTopK2()
{
    /* a input tensor of size (2, 4) */
    int sOrder = 2;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 2;
    sDimSize[1] = 4;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    /* a output tensor of size (2, 2) */
    int tOrder = 2;
    int * tDimSize = new int[tOrder];
    tDimSize[0] = 2;
    tDimSize[1] = 2;

    int tUnitNum = 1;
    for (int i = 0; i < tOrder; i++)
        tUnitNum *= tDimSize[i];

    DTYPE sData[2][4] = { {5.0F, 1.0F, 2.0F, 8.0F},
                          {4.0F, 3.0F, 7.0F, 6.0F} };
    DTYPE tAnswer[2][2] = { {8.0F, 5.0F},
                            {7.0F, 6.0F} };
    int indexAnswer[2][2] = { {3, 0},
                              {2, 3} };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensor(sOrder, sDimSize);
    XTensor * t = NewTensor(tOrder, tDimSize);
    XTensor * index = NewTensor(tOrder, tDimSize, X_INT);
    
    XTensor sUser = XTensor(sOrder, sDimSize, X_FLOAT, 1.0F, -1, NULL);
    XTensor tUser = XTensor(tOrder, tDimSize, X_FLOAT, 1.0F, -1, NULL);
    XTensor indexUser = NewTensor(tOrder, tDimSize, X_INT, 1.0F, -1, NULL);

    /* initialize variables */
    s->SetData(sData, sUnitNum);
    t->SetZeroAll();
    index->SetZeroAll();

    sUser.SetData(sData, sUnitNum);
    tUser.SetZeroAll();
    indexUser.SetZeroAll();

    /* call TopK function */
    int dim = 1;
    int k = tDimSize[dim];
    _TopK(s, t, index, dim, k);
    TopK(sUser, tUser, indexUser, dim, k);

    /* check results */
    cpuTest = t->CheckData(tAnswer, tUnitNum) && tUser.CheckData(tAnswer, tUnitNum)
              && index->CheckData(indexAnswer, tUnitNum) && indexUser.CheckData(indexAnswer, tUnitNum);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensor(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU = NewTensor(tOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor * indexGPU = NewTensor(tOrder, tDimSize, X_INT, 1.0F, 0);
    
    XTensor sUserGPU = XTensor(sOrder, sDimSize, X_FLOAT, 1.0F, 0, NULL);
    XTensor tUserGPU = XTensor(tOrder, tDimSize, X_FLOAT, 1.0F, 0, NULL);
    XTensor indexUserGPU = NewTensor(tOrder, tDimSize, X_INT, 1.0F, 0, NULL);

    /* initialize variables */
    sGPU->SetData(sData, sUnitNum);
    tGPU->SetZeroAll();
    indexGPU->SetZeroAll();

    sUserGPU.SetData(sData, sUnitNum);
    tUserGPU.SetZeroAll();
    indexUserGPU.SetZeroAll();

    /* call TopK function */
    dim = 1;
    k = tDimSize[dim];
    _TopK(sGPU, tGPU, indexGPU, dim, k);
    TopK(sUserGPU, tUserGPU, indexUserGPU, dim, k);

    /* check results */
    gpuTest = tGPU->CheckData(tAnswer, tUnitNum) && tUserGPU.CheckData(tAnswer, tUnitNum)
              && indexGPU->CheckData(indexAnswer, tUnitNum) && indexUserGPU.CheckData(indexAnswer, tUnitNum);

    /* destroy variables */
    delete s;
    delete t;
    delete index;
    delete sGPU;
    delete tGPU;
    delete indexGPU;
    delete[] sDimSize;
    delete[] tDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete t;
    delete index;
    delete[] sDimSize;
    delete[] tDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* other cases */
/*
TODO!!
*/

/* test for TopK Function */
bool TestTopK()
{
    XPRINT(0, stdout, "[TEST TopK] get the top-k items along a given dimension\n");
    bool returnFlag = true, caseFlag = true;
    
    /* case 1 test */
    caseFlag = TestTopK1();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 1 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 1 passed!\n");
    
    /* case 2 test */
    caseFlag = TestTopK2();
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
