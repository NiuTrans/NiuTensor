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
* $Created by: Lin Ye (email: linye2015@outlook.com) 2018-06-20
*/

#include "../core/utilities/CheckData.h"
#include "TNormalize.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
case 1: normalized the data with normal distribution 
For an input x, y = a * (x-mean)/sqrt(variance+\epsilon) + b.
where a and b are the scalar and bias respectively, 
and \epsilon is the adjustment parameter.
*/
bool TestNormalize1()
{
    /* a source tensor of size (2, 3) */
    int sOrder = 2;
    int * sDimSize = new int[sOrder];
    sDimSize[0] = 2;
    sDimSize[1] = 3;

    int sUnitNum = 1;
    for (int i = 0; i < sOrder; i++)
        sUnitNum *= sDimSize[i];

    /* a target tensor of size (2, 3) */
    int tOrder = 2;
    int * tDimSize = new int[tOrder];
    tDimSize[0] = 2;
    tDimSize[1] = 3;

    int tUnitNum = 1;
    for (int i = 0; i < tOrder; i++)
        tUnitNum *= tDimSize[i];

    /* a mean tensor of size (3) */
    int meanOrder = 1;
    int * meanDimSize = new int[meanOrder];
    meanDimSize[0] = 3;

    int meanUnitNum = 1;
    for (int i = 0; i < meanOrder; i++)
        meanUnitNum *= meanDimSize[i];

    /* a variance tensor of size (3) */
    int varOrder = 1;
    int * varDimSize = new int[varOrder];
    varDimSize[0] = 3;

    int varUnitNum = 1;
    for (int i = 0; i < varOrder; i++)
        varUnitNum *= varDimSize[i];

    /* a scalar tensor of size (2, 3) */
    int aOrder = 2;
    int * aDimSize = new int[aOrder];
    aDimSize[0] = 2;
    aDimSize[1] = 3;

    int aUnitNum = 1;
    for (int i = 0; i < aOrder; i++)
        aUnitNum *= aDimSize[i];

    /* a bias tensor of size (2, 3) */
    int bOrder = 2;
    int * bDimSize = new int[bOrder];
    bDimSize[0] = 2;
    bDimSize[1] = 3;

    int bUnitNum = 1;
    for (int i = 0; i < bOrder; i++)
        bUnitNum *= bDimSize[i];

    DTYPE sData[2][3] = { {1.0F, 2.0F, 3.0F},
                          {1.5F, 2.5F, 3.5F} };
    DTYPE meanData[3] = {1.0F, 1.5F, 2.0F};
    DTYPE varData[3] = {1.0F, 1.0F, 4.0F};
    DTYPE aData[2][3] = { {1.0F, 1.0F, 1.0F},
                          {1.0F, 1.0F, 1.0F} };
    DTYPE answer[2][3] = { {0.0F, 0.5F, 0.5F},
                           {0.5F, 1.0F, 0.75F} };

    /* CPU test */
    bool cpuTest = true;

    /* create tensors */
    XTensor * s = NewTensorV2(sOrder, sDimSize);
    XTensor * t = NewTensorV2(tOrder, tDimSize);
    XTensor * mean = NewTensorV2(meanOrder, meanDimSize);
    XTensor * var = NewTensorV2(varOrder, varDimSize);
    XTensor * a = NewTensorV2(aOrder, aDimSize);
    XTensor * b = NewTensorV2(bOrder, bDimSize);
    XTensor * tMe = NewTensorV2(sOrder, sDimSize);
    XTensor tUser;

    /* initialize variables */
    s->SetData(sData, sUnitNum);
    tMe->SetData(sData, sUnitNum);
    mean->SetData(meanData, meanUnitNum);
    var->SetData(varData, varUnitNum);
    a->SetData(aData, aUnitNum);
    b->SetZeroAll();
    t->SetZeroAll();

    /* call normalize function */
    _Normalize(s, t, 0, mean, var, a, b, 0.0F);
    _NormalizeMe(tMe, 0, mean, var, a, b, 0.0F);
    tUser = Normalize(*s, 0, *mean, *var, *a, *b, 0.0F);
    
    /* check results */
    cpuTest = _CheckData(t, answer, tUnitNum, 1e-4F)
        && _CheckData(tMe, answer, tUnitNum, 1e-4F) && _CheckData(&tUser, answer, tUnitNum, 1e-4F);

#ifdef USE_CUDA
    /* GPU test */
    bool gpuTest = true;

    /* create tensors */
    XTensor * sGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor * meanGPU = NewTensorV2(meanOrder, meanDimSize, X_FLOAT, 1.0F, 0);
    XTensor * varGPU = NewTensorV2(varOrder, varDimSize, X_FLOAT, 1.0F, 0);
    XTensor * aGPU = NewTensorV2(aOrder, aDimSize, X_FLOAT, 1.0F, 0);
    XTensor * bGPU = NewTensorV2(bOrder, bDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tGPU = NewTensorV2(tOrder, tDimSize, X_FLOAT, 1.0F, 0);
    XTensor * tMeGPU = NewTensorV2(sOrder, sDimSize, X_FLOAT, 1.0F, 0);
    XTensor tUserGPU;

    /* initialize variables */
    sGPU->SetData(sData, sUnitNum);
    tMeGPU->SetData(sData, sUnitNum);
    meanGPU->SetData(meanData, meanUnitNum);
    varGPU->SetData(varData, varUnitNum);
    aGPU->SetData(aData, aUnitNum);
    bGPU->SetZeroAll();
    tGPU->SetZeroAll();

    /* call Normalize function */
    _Normalize(sGPU, tGPU, 0, meanGPU, varGPU, aGPU, bGPU, 0.0F);
    _NormalizeMe(tMeGPU, 0, meanGPU, varGPU, aGPU, bGPU, 0.0F);
    tUserGPU = Normalize(*sGPU, 0, *meanGPU, *varGPU, *aGPU, *bGPU, 0.0F);

    /* check results */
    gpuTest = _CheckData(tGPU, answer, tUnitNum, 1e-4F)
        && _CheckData(tMeGPU, answer, tUnitNum, 1e-4F) && _CheckData(&tUserGPU, answer, tUnitNum, 1e-4F);

    /* destroy variables */
    delete s;
    delete tMe;
    delete t;
    delete mean;
    delete var;
    delete a;
    delete b;
    delete sGPU;
    delete tMeGPU;
    delete tGPU;
    delete meanGPU;
    delete varGPU;
    delete aGPU;
    delete bGPU;
    delete[] sDimSize;
    delete[] tDimSize;
    delete[] meanDimSize;
    delete[] varDimSize;
    delete[] aDimSize;
    delete[] bDimSize;

    return cpuTest && gpuTest;
#else
    /* destroy variables */
    delete s;
    delete tMe;
    delete t;
    delete mean;
    delete var;
    delete a;
    delete b;
    delete[] sDimSize;
    delete[] tDimSize;
    delete[] meanDimSize;
    delete[] varDimSize;
    delete[] aDimSize;
    delete[] bDimSize;

    return cpuTest;
#endif // USE_CUDA
}

/* other cases */
/*
TODO!!
*/

/* test for Normalize Function */
bool TestNormalize()
{
    XPRINT(0, stdout, "[TEST NORMALIZE] normalized the data with normal distribution \n");
    bool returnFlag = true, caseFlag = true;

    /* case 1 test */
    caseFlag = TestNormalize1();

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
