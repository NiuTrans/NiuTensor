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
    XTensor * s = NewTensorV2(sOrder, sDimSize);
    XTensor * t1 = NewTensorV2(tOrder, tDimSize);
    XTensor * t2 = NewTensorV2(tOrder, tDimSize);
    XTensor * index1 = NewTensorV2(tOrder, tDimSize, X_INT);
    XTensor * index2 = NewTensorV2(tOrder, tDimSize, X_INT);

    XTensor sUser = XTensor(sOrder, sDimSize, X_FLOAT, 1.0F, -1, NULL);
    XTensor tUser1 = XTensor(tOrder, tDimSize, X_FLOAT, 1.0F, -1, NULL);
    XTensor tUser2 = XTensor(tOrder, tDimSize, X_FLOAT, 1.0F, -1, NULL);
    XTensor indexUser1 = NewTensorV2(tOrder, tDimSize, X_INT, 1.0F, -1, NULL);
    XTensor indexUser2 = NewTensorV2(tOrder, tDimSize, X_INT, 1.0F, -1, NULL);

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
        
    for (int i = 0; i < tDimSize[1]; ++i)
    {
        for (int j = 0; j < tDimSize[0]; ++j)
        {
            float tmpData = ((float *)t1->data)[i + tDimSize[1] * j];
            int tmpIndex = ((int *)index1->data)[i + tDimSize[1] * j];
            float tmpDataUser = ((float *)tUser1.data)[i + tDimSize[1] * j];
            int tmpIndexUser = ((int *)indexUser1.data)[i + tDimSize[1] * j];
            bool flag = false;
            bool flagUser = false;
            for (int k = 0; k < tDimSize[0]; ++k)
            {
                float* ans = tAnswer1[0];
                int* ansIndex = indexAnswer1[0];
                if (tmpData == ans[i + tDimSize[1] * k] && tmpIndex == ansIndex[i + tDimSize[1] * k])
                {
                    flag = true;
                }
                if (tmpDataUser == ans[i + tDimSize[1] * k] && tmpIndexUser == ansIndex[i + tDimSize[1] * k])
                {
                    flagUser = true;
                }
            }
            cpuTest = cpuTest&&flag&&flagUser;
        }
    }

    for (int i = 0; i < tDimSize[0]; ++i)
    {
        for (int j = 0; j < tDimSize[1]; ++j)
        {
            float tmpData = ((float *)t2->data)[i * tDimSize[1] + j];
            int tmpIndex = ((int *)index2->data)[i * tDimSize[1] + j];
            float tmpDataUser = ((float *)tUser2.data)[i * tDimSize[1] + j];
            int tmpIndexUser = ((int *)indexUser2.data)[i * tDimSize[1] + j];
            bool flag = false;
            bool flagUser = false;
            for (int k = 0; k < tDimSize[1]; ++k)
            {
                float* ans = tAnswer2[0];
                int* ansIndex = indexAnswer2[0];
                if (tmpData == ans[i * tDimSize[1] + k] && tmpIndex == ansIndex[i * tDimSize[1] + k])
                {
                    flag = true;
                }
                if (tmpDataUser == ans[i * tDimSize[1] + k] && tmpIndexUser == ansIndex[i * tDimSize[1] + k])
                {
                    flagUser = true;
                }
            }
            cpuTest = cpuTest&&flag&&flagUser;
        }
    }



#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU1 = NewTensorV2(tOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU2 = NewTensorV2(tOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor * indexGPU1 = NewTensorV2(tOrder, tDimSize, X_INT, 1.0F, 0);
    XTensor * indexGPU2 = NewTensorV2(tOrder, tDimSize, X_INT, 1.0F, 0);
    
    XTensor sUserGPU = XTensor(sOrder, sDimSize, X_FLOAT, 1.0F, 0, NULL);
    XTensor tUserGPU1 = XTensor(tOrder, tDimSize, X_FLOAT, 1.0F, 0, NULL);
    XTensor tUserGPU2 = XTensor(tOrder, tDimSize, X_FLOAT, 1.0F, 0, NULL);
    XTensor indexUserGPU1 = NewTensorV2(tOrder, tDimSize, X_INT, 1.0F, 0, NULL);
    XTensor indexUserGPU2 = NewTensorV2(tOrder, tDimSize, X_INT, 1.0F, 0, NULL);

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
    float* checkData = new float[tUnitNum];
    int* checkIndex = new int[tUnitNum];
    float* checkDataUser = new float[tUnitNum];
    int* checkIndexUser = new int[tUnitNum];

    cudaMemcpy(checkData, tGPU1->data, sizeof(DTYPE)*tUnitNum,cudaMemcpyDeviceToHost);
    cudaMemcpy(checkIndex, indexGPU1->data, sizeof(int)*tUnitNum, cudaMemcpyDeviceToHost);
    cudaMemcpy(checkDataUser, tUserGPU1.data, sizeof(DTYPE)*tUnitNum, cudaMemcpyDeviceToHost);
    cudaMemcpy(checkIndexUser, indexUserGPU1.data, sizeof(int)*tUnitNum, cudaMemcpyDeviceToHost);

    for (int i = 0; i < tDimSize[1]; ++i)
    {
        for (int j = 0; j < tDimSize[0]; ++j)
        {
            float tmpData = ((float *)checkData)[i + tDimSize[1] * j];
            int tmpIndex = ((int *)checkIndex)[i + tDimSize[1] * j];
            float tmpDataUser = ((float *)checkDataUser)[i + tDimSize[1] * j];
            int tmpIndexUser = ((int *)checkIndexUser)[i + tDimSize[1] * j];
            bool flag = false;
            bool flagUser = false;
            for (int k = 0; k < tDimSize[0]; ++k)
            {
                float* ans = tAnswer1[0];
                int* ansIndex = indexAnswer1[0];
                if (tmpData == ans[i + tDimSize[1] * k] && tmpIndex == ansIndex[i + tDimSize[1] * k])
                {
                    flag = true;
                }
                if (tmpDataUser == ans[i + tDimSize[1] * k] && tmpIndexUser == ansIndex[i + tDimSize[1] * k])
                {
                    flagUser = true;
                }
            }
            gpuTest = gpuTest&&flag&&flagUser;
        }
    }

    cudaMemcpy(checkData, tGPU2->data, sizeof(DTYPE)*tUnitNum, cudaMemcpyDeviceToHost);
    cudaMemcpy(checkIndex, indexGPU2->data, sizeof(int)*tUnitNum, cudaMemcpyDeviceToHost);
    cudaMemcpy(checkDataUser, tUserGPU2.data, sizeof(DTYPE)*tUnitNum, cudaMemcpyDeviceToHost);
    cudaMemcpy(checkIndexUser, indexUserGPU2.data, sizeof(int)*tUnitNum, cudaMemcpyDeviceToHost);

    for (int i = 0; i < tDimSize[0]; ++i)
    {
        for (int j = 0; j < tDimSize[1]; ++j)
        {
            float tmpData = ((float *)checkData)[i * tDimSize[1] + j];
            int tmpIndex = ((int *)checkIndex)[i * tDimSize[1] + j];
            float tmpDataUser = ((float *)checkDataUser)[i * tDimSize[1] + j];
            int tmpIndexUser = ((int *)checkIndexUser)[i * tDimSize[1] + j];
            bool flag = false;
            bool flagUser = false;
            for (int k = 0; k < tDimSize[1]; ++k)
            {
                float* ans = tAnswer2[0];
                int* ansIndex = indexAnswer2[0];
                if (tmpData == ans[i * tDimSize[1] + k] && tmpIndex == ansIndex[i * tDimSize[1] + k])
                {
                    flag = true;
                }
                if (tmpDataUser == ans[i * tDimSize[1] + k] && tmpIndexUser == ansIndex[i * tDimSize[1] + k])
                {
                    flagUser = true;
                }
            }
            gpuTest = gpuTest&&flag&&flagUser;
        }
    }

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
    delete[] checkData;
    delete[] checkIndex;
    delete[] checkDataUser;
    delete[] checkIndexUser;

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
    XTensor * s = NewTensorV2(sOrder, sDimSize);
    XTensor * t = NewTensorV2(tOrder, tDimSize);
    XTensor * index = NewTensorV2(tOrder, tDimSize, X_INT);
    
    XTensor sUser = XTensor(sOrder, sDimSize, X_FLOAT, 1.0F, -1, NULL);
    XTensor tUser = XTensor(tOrder, tDimSize, X_FLOAT, 1.0F, -1, NULL);
    XTensor indexUser = NewTensorV2(tOrder, tDimSize, X_INT, 1.0F, -1, NULL);

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

    for (int i = 0; i < tDimSize[0]; ++i)
    {
        for (int j = 0; j < tDimSize[1]; ++j)
        {
            float tmpData = ((float *)t->data)[i * tDimSize[1] + j];
            int tmpIndex = ((int *)index->data)[i * tDimSize[1] + j];
            float tmpDataUser = ((float *)tUser.data)[i * tDimSize[1] + j];
            int tmpIndexUser = ((int *)indexUser.data)[i * tDimSize[1] + j];
            bool flag = false;
            bool flagUser = false;
            for (int k = 0; k < tDimSize[1]; ++k)
            {
                float* ans = tAnswer[0];
                int* ansIndex = indexAnswer[0];
                if (tmpData == ans[i * tDimSize[1] + k] && tmpIndex == ansIndex[i * tDimSize[1] + k])
                {
                    flag = true;
                }
                if (tmpDataUser == ans[i * tDimSize[1] + k] && tmpIndexUser == ansIndex[i * tDimSize[1] + k])
                {
                    flagUser = true;
                }
            }
            cpuTest = cpuTest&&flag&&flagUser;
        }
    }

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU = NewTensorV2(tOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor * indexGPU = NewTensorV2(tOrder, tDimSize, X_INT, 1.0F, 0);
    
    XTensor sUserGPU = XTensor(sOrder, sDimSize, X_FLOAT, 1.0F, 0, NULL);
    XTensor tUserGPU = XTensor(tOrder, tDimSize, X_FLOAT, 1.0F, 0, NULL);
    XTensor indexUserGPU = NewTensorV2(tOrder, tDimSize, X_INT, 1.0F, 0, NULL);

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
    float* checkData = new float[tUnitNum];
    int* checkIndex = new int[tUnitNum];
    float* checkDataUser = new float[tUnitNum];
    int* checkIndexUser = new int[tUnitNum];

    cudaMemcpy(checkData, tGPU->data, sizeof(DTYPE)*tUnitNum, cudaMemcpyDeviceToHost);
    cudaMemcpy(checkIndex, indexGPU->data, sizeof(int)*tUnitNum, cudaMemcpyDeviceToHost);
    cudaMemcpy(checkDataUser, tUserGPU.data, sizeof(DTYPE)*tUnitNum, cudaMemcpyDeviceToHost);
    cudaMemcpy(checkIndexUser, indexUserGPU.data, sizeof(int)*tUnitNum, cudaMemcpyDeviceToHost);

    for (int i = 0; i < tDimSize[0]; ++i)
    {
        for (int j = 0; j < tDimSize[1]; ++j)
        {
            float tmpData = ((float *)checkData)[i * tDimSize[1] + j];
            int tmpIndex = ((int *)checkIndex)[i * tDimSize[1] + j];
            float tmpDataUser = ((float *)checkDataUser)[i * tDimSize[1] + j];
            int tmpIndexUser = ((int *)checkIndexUser)[i * tDimSize[1] + j];
            bool flag = false;
            bool flagUser = false;
            for (int k = 0; k < tDimSize[1]; ++k)
            {
                float* ans = tAnswer[0];
                int* ansIndex = indexAnswer[0];
                if (tmpData == ans[i * tDimSize[1] + k] && tmpIndex == ansIndex[i * tDimSize[1] + k])
                {
                    flag = true;
                }
                if (tmpDataUser == ans[i * tDimSize[1] + k] && tmpIndexUser == ansIndex[i * tDimSize[1] + k])
                {
                    flagUser = true;
                }
            }
            gpuTest = gpuTest&&flag&&flagUser;
        }
    }

    /* destroy variables */
    delete s;
    delete t;
    delete index;
    delete sGPU;
    delete tGPU;
    delete indexGPU;
    delete[] sDimSize;
    delete[] tDimSize;
    delete[] checkData;
    delete[] checkIndex;
    delete[] checkDataUser;
    delete[] checkIndexUser;

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
