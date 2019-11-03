/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2018, Natural Language Processing Lab, Northestern University.
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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-07-12
 */

#include "../core/arithmetic/MatrixMul.h"
#include "../core/utilities/CheckData.h"
#include "TConvertDataType.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
case 1: test ConvertDataType function.
In this case, the flaot32 data type is converted to int32 data type.

*/
bool TestConvertDataType1()
{
    /* a tensor of size (3, 2) */
    int aOrder = 2;
    int * aDimSize = new int[aOrder];
    aDimSize[0] = 3;
    aDimSize[1] = 2;

    int aUnitNum = 1;
    for (int i = 0; i < aOrder; i++)
        aUnitNum *= aDimSize[i];

    DTYPE aData[3][2] = { {1.0F, 2.0F}, 
                          {0.5F, 4.0F},
                          {5.0F, 6.0F} };
    int answer[3][2] = { {1, 2},
                         {0, 4},
                         {5, 6} };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * a = NewTensorV2(aOrder, aDimSize);
    XTensor * b = NewTensorV2(aOrder, aDimSize, X_INT);

    /* initialize variables */
    a->SetData(aData, aUnitNum);
    b->SetZeroAll();

    /* call ConvertDataType function */
    _ConvertDataType(a, b);

    /* check results */
    cpuTest = _CheckData(b, answer, aUnitNum);
    
#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensor */
    XTensor * aGPU = NewTensorV2(aOrder, aDimSize, X_FLOAT, 1.0F, 0);
    XTensor * bGPU = NewTensorV2(aOrder, aDimSize, X_INT, 1.0F, 0);

    /* Initialize variables */
    aGPU->SetData(aData, aUnitNum);

    /* call ConvertDataType function */
    _ConvertDataType(aGPU, bGPU);

    /* check results */
    gpuTest = _CheckData(bGPU, answer, aUnitNum);

    /* destroy variables */
    delete a;
    delete b;
    delete aGPU;
    delete bGPU;
    delete[] aDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete a;
    delete b;
    delete[] aDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/*
case 2: test ConvertDataType function.
In this case, the int32 data type is converted to float32 data type.
*/
bool TestConvertDataType2()
{
    /* a tensor of size (3, 2) */
    int aOrder = 2;
    int * aDimSize = new int[aOrder];
    aDimSize[0] = 3;
    aDimSize[1] = 2;

    int aUnitNum = 1;
    for (int i = 0; i < aOrder; i++)
        aUnitNum *= aDimSize[i];

    int aData[3][2] = { {1, 2}, 
                        {0, 4},
                        {5, 6} };
    DTYPE answer[3][2] = { {1.0F, 2.0F}, 
                           {0.0F, 4.0F},
                           {5.0F, 6.0F} };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * a = NewTensorV2(aOrder, aDimSize, X_INT);
    XTensor * b = NewTensorV2(aOrder, aDimSize);

    /* initialize variables */
    a->SetData(aData, aUnitNum);
    b->SetZeroAll();

    /* call ConvertDataType function */
    _ConvertDataType(a, b);

    /* check results */
    cpuTest = _CheckData(b, answer, aUnitNum, 1e-4F);
    
#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensor */
    XTensor * aGPU = NewTensorV2(aOrder, aDimSize, X_INT, 1.0F, 0);
    XTensor * bGPU = NewTensorV2(aOrder, aDimSize, X_FLOAT, 1.0F, 0);

    /* Initialize variables */
    aGPU->SetData(aData, aUnitNum);

    /* call ConvertDataType function */
    _ConvertDataType(aGPU, bGPU);

    /* check results */
    gpuTest = _CheckData(bGPU, answer, aUnitNum, 1e-4F);

    /* destroy variables */
    delete a;
    delete b;
    delete aGPU;
    delete bGPU;
    delete[] aDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete a;
    delete b;
    delete[] aDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/*
case 3: test ConvertDataType function.
In this case, the float data type is converted to float16 data type.
*/
bool TestConvertDataType3()
{
    int order = 2;

    /* a tensor of size (3, 2) */
    int * dimSize1 = new int[order];
    dimSize1[0] = 3;
    dimSize1[1] = 2;

    int unitNum1 = 1;
    for (int i = 0; i < order; i++)
        unitNum1 *= dimSize1[i];
        
    /* a tensor of size (3, 2) */
    int * dimSize2 = new int[order];
    dimSize2[0] = 2;
    dimSize2[1] = 3;

    int unitNum2 = 1;
    for (int i = 0; i < order; i++)
        unitNum2 *= dimSize2[i];
        
    /* a tensor of size (3, 3) */
    int * dimSize3 = new int[order];
    dimSize3[0] = 3;
    dimSize3[1] = 3;

    int unitNum3 = 1;
    for (int i = 0; i < order; i++)
        unitNum3 *= dimSize3[i];

    DTYPE data1[3][2] = { {1.0F, -2.0F},
                          {0.5F, -4.0F},
                          {0.0F, 6.0F} };
    
    DTYPE data2[2][3] = { {1.0F, 2.0F, 3.0F},
                          {0.0F, 4.0F, 5.0F} };
    
    DTYPE answer[3][3] = { {1.0F, -6.0F, -7.0F},
                           {0.5F, -15.0F, -18.5F}, 
                           {0.0F, 24.0F, 30.0F} };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * a = NewTensorV2(order, dimSize1, X_FLOAT, 1.0F, -1);
    XTensor * b = NewTensorV2(order, dimSize1, X_FLOAT16, 1.0F, -1);
    XTensor * c = NewTensorV2(order, dimSize1, X_FLOAT, 1.0F, -1);

    /* initialize variables */
    a->SetData(data1, unitNum1);

    /* call ConvertDataType function (We have not implemented this yet...)  */
    //_ConvertDataType(a, b);
    //_ConvertDataType(b, c);
    
    /* check results */
    //cpuTest = _CheckData(a, data1, unitNum1, 1e-4F);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensor */
    XTensor * aGPU = NewTensorV2(order, dimSize1, X_FLOAT, 1.0F, 0);
    XTensor * bGPU = NewTensorV2(order, dimSize2, X_FLOAT, 1.0F, 0);
    XTensor * cGPU = NewTensorV2(order, dimSize1, X_FLOAT16, 1.0F, 0);
    XTensor * dGPU = NewTensorV2(order, dimSize2, X_FLOAT16, 1.0F, 0);
    XTensor * eGPU = NewTensorV2(order, dimSize3, X_FLOAT16, 1.0F, 0);
    XTensor * fGPU = NewTensorV2(order, dimSize3, X_FLOAT, 1.0F, 0);

    /* Initialize variables */
    aGPU->SetData(data1, unitNum1);
    bGPU->SetData(data2, unitNum2);

    /* call ConvertDataType function */
    _ConvertDataType(aGPU, cGPU);
    _ConvertDataType(bGPU, dGPU);

    _MatrixMul(cGPU, X_NOTRANS, dGPU, X_NOTRANS, eGPU);
    _ConvertDataType(eGPU, fGPU);

    /* check results */
    gpuTest = _CheckData(fGPU, answer, unitNum3, 1e-4F);

    /* destroy variables */
    delete a;
    delete b;
    delete c;
    delete aGPU;
    delete bGPU;
    delete cGPU;
    delete[] dimSize1;
    delete[] dimSize2;
    delete[] dimSize3;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete a;
    delete b;
    delete c;
    delete[] dimSize1;
    delete[] dimSize2;
    delete[] dimSize3;

    return cpuTest;
#endif // USE_CUDA
}

/* other cases */
/*
TODO!!
*/

/* test for ConvertDataType Function */
bool TestConvertDataType()
{
    XPRINT(0, stdout, "[TEST ConvertDataType] convert data type \n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestConvertDataType1();

    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 1 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 1 passed!\n");

    /* case 2 test */
    caseFlag = TestConvertDataType2();

    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 2 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 2 passed!\n");
    
    /* case 3 test */
    caseFlag = TestConvertDataType3();

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
