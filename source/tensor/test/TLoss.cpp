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
* $Created by: LI Yinqiao (email: li.yin.qiao.2012@hotmail.com) 2018-04-30
*/

#include<math.h>
#include "../core/math/ScaleAndShift.h"
#include "TLoss.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* 
case 1: test LossCompute function.
In this case, Loss function name = SQUAREDERROR.
loss = sum_{i} 0.5*(t_i - y_i)^2, 
where t_i is the gold standard and y_i is the model output.
*/
bool TestLoss1()
{
    /* a tensor of size (10, 1) */
    int order = 2;
    int * dimSize = new int[order];
    dimSize[0] = 10;
    dimSize[1] = 1;

    int unitNum = 1;
    for (int i = 0; i < order; i++)
        unitNum *= dimSize[i];

    /* CPU test */
    bool cpuTest = true;

    DTYPE answer = 5.0F;
    DTYPE error;

    /* create tensors */
    XTensor * output = NewTensorV2(order, dimSize);
    XTensor * gold = NewTensorV2(order, dimSize);

    /* initialize variables */
    output->SetZeroAll();
    gold->SetZeroAll();
    _ScaleAndShiftMe(output, 1, 1);
    _ScaleAndShiftMe(gold, 1, 2);

    /* call LossCompute function */
    error = _LossCompute(gold, output, SQUAREDERROR, false, 0, 0, dimSize[0], 0);
    
    /* check results */
    cpuTest = (fabs(error - answer) < 1e-4);
    
#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensor */
    XTensor * outputGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * goldGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);

    /* Initialize variables */
    outputGPU->SetZeroAll();
    goldGPU->SetZeroAll();
    _ScaleAndShiftMe(outputGPU, 1, 1);
    _ScaleAndShiftMe(goldGPU, 1, 2);

    /* call LossCompute function */
    error = _LossCompute(goldGPU, outputGPU, SQUAREDERROR, false, 0, 0, dimSize[0], 0);
    
    /* check results */
    gpuTest = (fabs(error - answer) < 1e-4);

    /* destroy variables */
    delete output;
    delete gold;
    delete outputGPU;
    delete goldGPU;
    delete[] dimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete output;
    delete gold;
    delete[] dimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* 
case 2: test LossCompute function.
In this case, Loss function name = CROSSENTROPY.
loss = sum_{i} (-t_i * log(y_i))
where t_i is the gold standard and y_i is the model output.
*/
bool TestLoss2()
{
    /* a tensor of size (10, 1) */
    int order = 2;
    int * dimSize = new int[order];
    dimSize[0] = 10;
    dimSize[1] = 1;

    int unitNum = 1;
    for (int i = 0; i < order; i++)
        unitNum *= dimSize[i];

    /* CPU test */
    bool cpuTest = true;

    DTYPE answer = 0.0F;
    DTYPE error;

    /* create tensors */
    XTensor * output = NewTensorV2(order, dimSize);
    XTensor * gold = NewTensorV2(order, dimSize);

    /* initialize variables */
    output->SetZeroAll();
    gold->SetZeroAll();
    _ScaleAndShiftMe(output, 1, 1);
    _ScaleAndShiftMe(gold, 1, 2);

    /* call LossCompute function */
    error = _LossCompute(gold, output, CROSSENTROPY, false, 0, 0, dimSize[0], 0);
    
    /* check results */
    cpuTest = (fabs(error - answer) < 1e-4);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensor */
    XTensor * outputGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * goldGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);

    /* Initialize variables */
    outputGPU->SetZeroAll();
    goldGPU->SetZeroAll();
    _ScaleAndShiftMe(outputGPU, 1, 1);
    _ScaleAndShiftMe(goldGPU, 1, 2);

    /* call LossCompute function */
    error = _LossCompute(goldGPU, outputGPU, CROSSENTROPY, false, 0, 0, dimSize[0], 0);
    
    /* check results */
    gpuTest = (fabs(error - answer) < 1e-4);

    /* destroy variables */
    delete output;
    delete gold;
    delete outputGPU;
    delete goldGPU;
    delete[] dimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete output;
    delete gold;
    delete[] dimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* 
case 3: test LossCompute function.
In this case, Loss function name = ONEHOTERROR.
loss = sum_{i} e_i
where e_i = 0.5*(t_i - y_i)^2 if t_i = 1, e_i = 0 otherwise.
*/
bool TestLoss3()
{
    /* a tensor of size (10, 1) */
    int order = 2;
    int * dimSize = new int[order];
    dimSize[0] = 5;
    dimSize[1] = 1;

    int unitNum = 1;
    for (int i = 0; i < order; i++)
        unitNum *= dimSize[i];
    DTYPE outputData[5][1] = { {0.5F},
                               {0.5F},
                               {0.5F},
                               {0.5F},
                               {0.5F} };
    DTYPE goldData[5][1] = { {1.0F},
                             {1.0F},
                             {0.0F},
                             {0.0F},
                             {0.0F} };

    /* CPU test */
    bool cpuTest = true;

    DTYPE answer = 0.25F;
    DTYPE error;

    /* create tensors */
    XTensor * output = NewTensorV2(order, dimSize);
    XTensor * gold = NewTensorV2(order, dimSize);

    /* initialize variables */
    output->SetData(outputData, unitNum);
    gold->SetData(goldData, unitNum);

    /* call LossCompute function */
    error = _LossCompute(gold, output, ONEHOTERROR, false, 0, 0, dimSize[0], 0);
    
    /* check results */
    cpuTest = (fabs(error - answer) < 1e-4);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensor */
    XTensor * outputGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * goldGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);

    /* Initialize variables */
    outputGPU->SetData(outputData, unitNum);
    goldGPU->SetData(goldData, unitNum);

    /* call LossCompute function */
    error = _LossCompute(goldGPU, outputGPU, ONEHOTERROR, false, 0, 0, dimSize[0], 0);
    
    /* check results */
    gpuTest = (fabs(error - answer) < 1e-4);

    /* destroy variables */
    delete output;
    delete gold;
    delete outputGPU;
    delete goldGPU;
    delete[] dimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete output;
    delete gold;
    delete[] dimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* other cases */
/*
TODO!!
*/

/* test for Loss Function */
bool TestLoss()
{
    XPRINT(0, stdout, "[TEST Loss] compute the loss \n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestLoss1();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 1 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 1 passed!\n");

    /* case 2 test */
    caseFlag = TestLoss2();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 2 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 2 passed!\n");
        
    caseFlag = TestLoss3();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 3 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 3 passed!\n");

    ///* other cases test */
    ///*
    //TODO!!
    //*/

    if (returnFlag) {
        XPRINT(0, stdout, ">> All Passed!\n");
    }
    else
        XPRINT(0, stdout, ">> Failed!\n");

    XPRINT(0, stdout, "\n");

    return returnFlag;
}

} // namespace nts(NiuTrans.Tensor)
