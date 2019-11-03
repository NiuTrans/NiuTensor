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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-07-06
 */

#include "../core/utilities/CheckData.h"
#include "../core/getandset/SetData.h"
#include "TSetData.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* 
case 1: test SetDataRand function.
set the tensor items by a uniform distribution in range [lower, upper]. 
*/
bool TestSetData1()
{
    /* a input tensor of size (2, 4) */
    int sOrder = 2;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 2;
    sDimSize[1] = 4;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    DTYPE answer[2][4] = {0};

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(sOrder, sDimSize);

    /* call SetDataRand function */
    s->SetDataRand(0.0, 1.0);
    
    /* check results */
    cpuTest = _CheckData(s, answer, sUnitNum, 1.0F);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);

    /* call SetDataRand function */
    sGPU->SetDataRand(0.0, 1.0);
    
    gpuTest = _CheckData(sGPU, answer, sUnitNum, 1.0F);

    /* destroy variables */
    delete s;
    delete sGPU;
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
case 2: test SetDataIndexed function.
modify data items along with a given dimension.
*/
bool TestSetData2()
{
    /* a input tensor of size (2, 4) */
    int sOrder = 2;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 2;
    sDimSize[1] = 4;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    /* a data tensor of size (4) for GPU test */
    int dataOrder = 1;
    int * dataDimSize = new int[dataOrder];
    dataDimSize[0] = 4;

    int dataUnitNum = 1;
    for (int i = 0; i < dataOrder; i++)
        dataUnitNum *= dataDimSize[i];

    DTYPE data[4] = {0.0F, 1.0F, 2.0F, 3.0F};
    DTYPE answer[2][4] = { {1.0F, 1.0F, 1.0F, 1.0F}, 
                           {0.0F, 1.0F, 2.0F, 3.0F} };
    
    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(sOrder, sDimSize);
    XTensor * modify = NewTensorV2(dataOrder, dataDimSize);

    /* Initialize variables */
    _SetDataFixedFloat(s, 1.0F);
    modify->SetData(data, dataUnitNum);

    /* call SetDataIndexed function */
    _SetDataIndexed(s, modify, 0, 1);

    /* check results */
    cpuTest = _CheckData(s, answer, sUnitNum, 1e-5F);
    
#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * modifyGPU = NewTensorV2(dataOrder, dataDimSize, X_FLOAT, 1.0F, 0);

    /* Initialize variables */
    _SetDataFixedFloat(sGPU, 1.0F);
    modifyGPU->SetData(data, dataUnitNum);

    /* call SetDataIndexed function */
    _SetDataIndexed(sGPU, modifyGPU, 0, 1);
    
    gpuTest = _CheckData(sGPU, answer, sUnitNum, 1e-5F);

    /* destroy variables */
    delete s;
    delete modify;
    delete sGPU;
    delete modifyGPU;
    delete[] sDimSize;
    delete[] dataDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete modify;
    delete[] sDimSize;
    delete[] dataDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/*
case 3: test SetDataIndexed function.
modify data items along with a given dimension.
*/
bool TestSetData3()
{
    /* a input tensor of size (2, 4, 3) */
    int sOrder = 3;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 2;
    sDimSize[1] = 4;
    sDimSize[2] = 3;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];
    
    /* a data tensor of size (2, 3) for GPU test */
    int dataOrder = 2;
    int * dataDimSize = new int[dataOrder];
    dataDimSize[0] = 2;
    dataDimSize[1] = 3;

    int dataUnitNum = 1;
    for (int i = 0; i < dataOrder; i++)
        dataUnitNum *= dataDimSize[i];

    DTYPE data[2][3] = { {0.0F, 1.0F, 2.0F},
                         {3.0F, 4.0F, 5.0F} };

    DTYPE answer[2][4][3] = { { {1.0F, 1.0F, 1.0F},
                                {0.0F, 1.0F, 2.0F},
                                {1.0F, 1.0F, 1.0F},
                                {1.0F, 1.0F, 1.0F} },
                              { {1.0F, 1.0F, 1.0F},
                                {3.0F, 4.0F, 5.0F},
                                {1.0F, 1.0F, 1.0F},
                                {1.0F, 1.0F, 1.0F} } };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(sOrder, sDimSize);
    XTensor * modify = NewTensorV2(dataOrder, dataDimSize);

    /* Initialize variables */
    _SetDataFixedFloat(s, 1.0F);
    modify->SetData(data, dataUnitNum);

    /* call SetDataIndexed function */
    _SetDataFixedFloat(s, 1.0F);
    _SetDataIndexed(s, modify, 1, 1);
    
    /* check results */
    cpuTest = _CheckData(s, answer, sUnitNum, 1e-5F);
    
#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * modifyGPU = NewTensorV2(dataOrder, dataDimSize, X_FLOAT, 1.0F, 0);

    /* Initialize variables */
    _SetDataFixedFloat(sGPU, 1.0F);
    modifyGPU->SetData(data, dataUnitNum);
    
    /* call SetDataIndexed function */
    _SetDataIndexed(sGPU, modifyGPU, 1, 1);
    
    gpuTest = _CheckData(sGPU, answer, sUnitNum, 1e-5F);

    /* destroy variables */
    delete s;
    delete modify;
    delete sGPU;
    delete modifyGPU;
    delete[] sDimSize;
    delete[] dataDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete modify;
    delete[] sDimSize;
    delete[] dataDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/*
case 4: test SetDataDim function.
set data items along with a given dimension (and keep the remaining items unchanged)
*/
bool TestSetData4()
{
    /* a input tensor of size (3, 3) */
    int order = 2;
    int * dimSize = new int[order];
    dimSize[0] = 3;
    dimSize[1] = 3;

    int unitNum = 1;
    for (int i = 0; i < order; i++)
        unitNum *= dimSize[i];

    DTYPE sData[3][3] = { {1.0F, 2.0F, 3.0F},
                          {4.0F, 5.0F, 6.0F},
                          {7.0F, 8.0F, 9.0F} };
    DTYPE answer[3][3] = { {1.0F, 2.0F, 3.0F},
                           {0.0F, 0.0F, 0.0F},
                           {7.0F, 8.0F, 9.0F} };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(order, dimSize);

    /* initialize variables */
    s->SetData(sData, unitNum);

    /* call _SetDataDim function */
    _SetDataDim(s, 1, 1, 0, 0);

    /* check results */
    cpuTest = _CheckData(s, answer, unitNum, 1e-4F);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);

    /* initialize variables */
    sGPU->SetData(sData, unitNum);

    /* call _SetDataDim function */
    _SetDataDim(sGPU, 1, 1, 0, 0);

    gpuTest = _CheckData(sGPU, answer, unitNum, 1e-4F);

    /* destroy variables */
    delete s;
    delete sGPU;
    delete[] dimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete[] dimSize;

    return cpuTest;
#endif // USE_CUDA
}

/*
case 5: test SetDataDim function.
set data items along with a given dimension (and keep the remaining items unchanged)
*/
bool TestSetData5()
{
    /* a input tensor of size (2, 4, 3) */
    int order = 3;
    int * dimSize = new int[order];
    dimSize[0] = 2;
    dimSize[1] = 4;
    dimSize[2] = 3;

    int unitNum = 1;
    for (int i = 0; i < order; i++)
        unitNum *= dimSize[i];

    DTYPE data[2][4][3] = { { {1.0F, 1.0F, 1.0F},
                              {0.0F, 1.0F, 2.0F},
                              {1.0F, 1.0F, 1.0F},
                              {1.0F, 1.0F, 1.0F} },
                            { {1.0F, 1.0F, 1.0F},
                              {3.0F, 4.0F, 5.0F},
                              {1.0F, 1.0F, 1.0F},
                              {1.0F, 1.0F, 1.0F} } };

    DTYPE answer[2][4][3] = { { {1.0F, 1.0F, 1.0F},
                                {0.0F, 1.0F, 2.0F},
                                {5.0F, 5.0F, 5.0F},
                                {1.0F, 1.0F, 1.0F} },
                              { {1.0F, 1.0F, 1.0F},
                                {3.0F, 4.0F, 5.0F},
                                {5.0F, 5.0F, 5.0F},
                                {1.0F, 1.0F, 1.0F} } };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(order, dimSize);

    /* initialize variables */
    s->SetData(data, unitNum);

    /* call _SetDataDim function */
    _SetDataDim(s, 2, 1, 1, 5.0F);

    /* check results */
    cpuTest = _CheckData(s, answer, unitNum, 1e-4F);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);

    /* initialize variables */
    sGPU->SetData(data, unitNum);

    /* call _SetDataDim function */
    _SetDataDim(sGPU, 2, 1, 1, 5.0F);

    gpuTest = _CheckData(sGPU, answer, unitNum, 1e-4F);

    /* destroy variables */
    delete s;
    delete sGPU;
    delete[] dimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete[] dimSize;

    return cpuTest;
#endif // USE_CUDA
}

/*
case 6: test SetDataRange function.
generate data items with a range by start, end and the step
*/
bool TestSetData6()
{
    /* a input tensor of size (5) */
    int order = 1;
    int * dimSize = new int[order];
    dimSize[0] = 5;

    int unitNum = 1;
    for (int i = 0; i < order; i++)
        unitNum *= dimSize[i];

    DTYPE answer[5] = {5.2F, 3.2F, 1.2F, -0.8F, -2.8F};

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(order, dimSize);

    /* initialize variables */
    s->SetZeroAll();

    /* call _SetDataRange function */
    _SetDataRange(s, 5.2, -3.2, -2);

    /* check results */
    cpuTest = _CheckData(s, answer, unitNum, 1e-4F);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);

    /* initialize variables */
    sGPU->SetZeroAll();

    /* call _SetDataRange function */
    _SetDataRange(sGPU, 5.2, -3.2, -2);

    gpuTest = _CheckData(sGPU, answer, unitNum, 1e-4F);

    /* destroy variables */
    delete s;
    delete sGPU;
    delete[] dimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete[] dimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* other cases */
/*
TODO!!
*/

/* test for SetData Function */
bool TestSetData()
{
    XPRINT(0, stdout, "[TEST SetData] set the data of tensor \n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestSetData1();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 1 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 1 passed!\n");

    /* case 2 test */
    caseFlag = TestSetData2();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 2 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 2 passed!\n");
    
    /* case 3 test */
    caseFlag = TestSetData3();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 3 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 3 passed!\n");
        
    /* case 4 test */
    caseFlag = TestSetData4();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 4 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 4 passed!\n");
            
    /* case 5 test */
    caseFlag = TestSetData5();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 5 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 5 passed!\n");

    /* case 6 test */
    caseFlag = TestSetData6();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 6 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 6 passed!\n");

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
