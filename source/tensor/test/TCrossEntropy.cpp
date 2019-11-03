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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-09-17
 */

#include <math.h>
#include "../core/utilities/CheckData.h"
#include "../loss/CrossEntropy.h"
#include "../core/math/ScaleAndShift.h"
#include "TCrossEntropy.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* 
case 1: test CrossEntropy function.
loss = sum_{i} (-t_i * log(y_i))
where t_i is the gold standard and y_i is the model output.
*/
bool TestCrossEntropy1()
{
    /* a tensor of size (1, 4) */
    int order = 2;
    int * dimSize = new int[order];
    dimSize[0] = 1;
    dimSize[1] = 4;

    int unitNum = 1;
    for (int i = 0; i < order; i++)
        unitNum *= dimSize[i];

    DTYPE outputData[4] = {0.25F, 0.25F, 0.25F, 0.25F};
    DTYPE goldData[4] = {0.5F, 0.5F, 0.0F, 0.0F};
    DTYPE answer = 1.3863F;
    DTYPE error1;
    DTYPE error2;
    
    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * output = NewTensorV2(order, dimSize);
    XTensor * gold = NewTensorV2(order, dimSize);
    XTensor * loss = NewTensor1DV2(1);

    /* initialize variables */
    output->SetData(outputData, unitNum);
    gold->SetData(goldData, unitNum);

    /* call CrossEntropy function */
    _CrossEntropyFast(output, gold, loss);
    error2 = _CrossEntropy(output, gold, REDUCE_SUM);
    error1 = loss->Get1D(0);

    /* check results */
    cpuTest = (fabs(error1 - answer) < 1e-4F && 
               fabs(error2 - answer) < 1e-4F);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensor */
    XTensor * outputGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * goldGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * lossGPU = NewTensor1DV2(1, X_FLOAT, 0);

    /* Initialize variables */
    outputGPU->SetData(outputData, unitNum);
    goldGPU->SetData(goldData, unitNum);

    /* call CrossEntropy function */
    _CrossEntropyFast(outputGPU, goldGPU, lossGPU);
    error1 = lossGPU->Get1D(0);
    error2 = _CrossEntropy(outputGPU, goldGPU, REDUCE_SUM);

    /* check results */
    gpuTest = (fabs(error1 - answer) < 1e-4F && 
               fabs(error2 - answer) < 1e-4F);

    /* destroy variables */
    delete output;
    delete gold;
    delete loss;
    delete outputGPU;
    delete goldGPU;
    delete lossGPU;

    delete[] dimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete output;
    delete gold;
    delete loss;
    delete[] dimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* 
case 2: test CrossEntropy function.
loss = sum_{i} (-t_i * log(y_i))
where t_i is the gold standard and y_i is the model output.
*/
bool TestCrossEntropy2()
{
    /* a tensor of size (4, 10) */
    int order = 2;
    int * dimSize = new int[order];
    dimSize[0] = 4;
    dimSize[1] = 10;

    int unitNum = 1;
    for (int i = 0; i < order; i++)
        unitNum *= dimSize[i];

    DTYPE outputData[4][10] = { {0.5F, 2.6F, 0.3F, 1.7F, 0.6F, 
                                 0.1F, 0.7F, 1.3F, 0.4F, 0.6F}, 
                                {0.5F, 1.6F, 0.2F, 1.1F, 0.3F, 
                                 0.8F, 2.2F, 0.1F, 0.1F, 0.8F},
                                {0.2F, 0.5F, 1.1F, 1.2F, 0.6F, 
                                 0.1F, 0.2F, 0.7F, 0.5F, 0.7F},
                                {0.2F, 1.7F, 0.6F, 1.5F, 0.8F, 
                                 0.1F, 0.8F, 0.1F, 0.6F, 0.2F} };
    DTYPE answer1 = 4.3275F;
    DTYPE answer2 = 1.0818F;
    DTYPE error1;
    DTYPE error2;
    DTYPE error3;
    DTYPE error4;
    
    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * output = NewTensorV2(order, dimSize);
    XTensor * gold = NewTensorV2(order, dimSize);

    /* initialize variables */
    output->SetData(outputData, unitNum);
    gold->SetZeroAll();
    gold->Set2D(1.0F, 0, 9);
    gold->Set2D(1.0F, 1, 7);
    gold->Set2D(1.0F, 2, 2);
    gold->Set2D(1.0F, 3, 9);

    /* call CrossEntropy function */
    error1 = _CrossEntropy(output, gold, REDUCE_SUM);
    error2 = _CrossEntropy(output, gold, REDUCE_MEAN);
    error3 = _CrossEntropyFast(output, gold, REDUCE_SUM);
    error4 = _CrossEntropyFast(output, gold, REDUCE_MEAN);
    
    /* check results */
    cpuTest = (fabs(error1 - answer1) < 1e-4F &&
               fabs(error2 - answer2) < 1e-4F && 
               fabs(error3 - answer1) < 1e-4F &&
               fabs(error4 - answer2) < 1e-4F);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensor */
    XTensor * outputGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * goldGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);

    /* Initialize variables */
    outputGPU->SetData(outputData, unitNum);
    goldGPU->SetZeroAll();
    goldGPU->Set2D(1.0F, 0, 9);
    goldGPU->Set2D(1.0F, 1, 7);
    goldGPU->Set2D(1.0F, 2, 2);
    goldGPU->Set2D(1.0F, 3, 9);

    /* call CrossEntropy function */
    error1 = _CrossEntropy(outputGPU, goldGPU, REDUCE_SUM);
    error2 = _CrossEntropy(outputGPU, goldGPU, REDUCE_MEAN);
    error3 = _CrossEntropyFast(outputGPU, goldGPU, REDUCE_SUM);
    error4 = _CrossEntropyFast(outputGPU, goldGPU, REDUCE_MEAN);

    /* check results */
    gpuTest = (fabs(error1 - answer1) < 1e-4F &&
               fabs(error2 - answer2) < 1e-4F && 
               fabs(error3 - answer1) < 1e-4F &&
               fabs(error4 - answer2) < 1e-4F);

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
case 3: test CrossEntropy function.
loss = sum_{i} (-t_i * log(y_i))
where t_i is the gold standard and y_i is the model output.
In this case, I compute the cross entropy with weight.
*/
bool TestCrossEntropy3()
{
    /* a output tensor of size (4, 4) */
    int order = 2;
    int * dimSize = new int[order];
    dimSize[0] = 4;
    dimSize[1] = 4;

    int unitNum = 1;
    for (int i = 0; i < order; i++)
        unitNum *= dimSize[i];
        
    /* a weight tensor of size (4) */
    int wOrder = 1;
    int * wDimSize = new int[wOrder];
    wDimSize[0] = 4;

    int wUnitNum = 1;
    for (int i = 0; i < wOrder; i++)
        wUnitNum *= wDimSize[i];

    DTYPE outputData[4][4] = { {0.3F, 0.2F, 0.3F, 0.2F}, 
                               {0.1F, 0.4F, 0.2F, 0.3F}, 
                               {0.7F, 0.1F, 0.1F, 0.1F}, 
                               {0.5F, 0.1F, 0.2F, 0.2F} };
    DTYPE weightData[4] = {2.0F, 1.0F, 5.0F, 0.0F};
    DTYPE answer[4] = {2.4079F, 0.9163F, 11.5129F, 0.0F};
    
    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * output = NewTensorV2(order, dimSize);
    XTensor * gold = NewTensorV2(order, dimSize);
    XTensor * loss = NewTensor1DV2(4);
    XTensor * weight = NewTensorV2(wOrder, wDimSize);

    /* initialize variables */
    output->SetData(outputData, unitNum);
    weight->SetData(weightData, wUnitNum);
    gold->SetZeroAll();
    gold->Set2D(1.0F, 0, 0);
    gold->Set2D(1.0F, 1, 1);
    gold->Set2D(1.0F, 2, 2);
    gold->Set2D(1.0F, 3, 3);

    /* call CrossEntropy function */
    _CrossEntropyFast(output, gold, loss, weight);

    /* check results */
    cpuTest = _CheckData(loss, answer, 4, 1e-4F);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensor */
    XTensor * outputGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * goldGPU = NewTensorV2(order, dimSize, X_FLOAT, 1.0F, 0);
    XTensor * lossGPU = NewTensor1DV2(4, X_FLOAT, 0);
    XTensor * weightGPU = NewTensorV2(wOrder, wDimSize, X_FLOAT, 1.0F, 0);

    /* Initialize variables */
    outputGPU->SetData(outputData, unitNum);
    weightGPU->SetData(weightData, wUnitNum);
    goldGPU->SetZeroAll();
    goldGPU->Set2D(1.0F, 0, 0);
    goldGPU->Set2D(1.0F, 1, 1);
    goldGPU->Set2D(1.0F, 2, 2);
    goldGPU->Set2D(1.0F, 3, 3);
        
    /* call CrossEntropy function */
    _CrossEntropyFast(outputGPU, goldGPU, lossGPU, weightGPU);

    /* check results */
    gpuTest = _CheckData(lossGPU, answer, 4, 1e-4F);

    /* destroy variables */
    delete output;
    delete gold;
    delete loss;
    delete weight;
    delete outputGPU;
    delete goldGPU;
    delete lossGPU;
    delete weightGPU;
    delete[] dimSize;
    delete[] wDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete output;
    delete gold;
    delete loss;
    delete weight;
    delete[] dimSize;
    delete[] wDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* 
case 4: test CrossEntropy function.
loss = sum_{i} (-t_i * log(y_i))
where t_i is the gold standard and y_i is the model output.
*/
bool TestCrossEntropy4()
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

    /* call CrossEntropy function */
    error = _CrossEntropyFast(output, gold);
    
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

    /* call CrossEntropy function */
    error = _CrossEntropyFast(outputGPU, goldGPU);
    
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

/* test for CrossEntropy Function */
bool TestCrossEntropy()
{
    XPRINT(0, stdout, "[TEST CrossEntropy] compute the cross entropy loss and backward gradient \n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestCrossEntropy1();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 1 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 1 passed!\n");
    
    /* case 2 test */
    caseFlag = TestCrossEntropy2();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 2 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 2 passed!\n");
        
    /* case 3 test */
    caseFlag = TestCrossEntropy3();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 3 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 3 passed!\n");        
    
    /* case 4 test */
    caseFlag = TestCrossEntropy4();
    if (!caseFlag) {
        returnFlag = false;
        XPRINT(0, stdout, ">> case 4 failed!\n");
    }
    else
        XPRINT(0, stdout, ">> case 4 passed!\n");

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
