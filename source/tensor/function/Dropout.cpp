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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-09-12
 */

#include "../XName.h"
#include <time.h>
#include <math.h>
#include "Dropout.h"
#include "Dropout.cuh"
#include "../core/arithmetic/Multiply.h"
#include "../core/arithmetic/MultiplyDim.h"
#include "../core/math/ScaleAndShift.h"
#include "../core/getandset/SetData.h"
#include "DropoutWithIndex.h"

namespace nts{ // namespace nts(NiuTrans.Tensor

/*
dropout function
It randomly zeroes some of the elements of the input tensor
with probability p via a Bernoulli distribution.

See "Improving neural networks by preventing co-adaptation of feature detectors"
for more details.

Here, the output is scaled by a factor of \frac{1}{1-p} so that we do not need
to mark the tensor with probability p in the inference phase. Instead we perform
the same inference procedure as that on the test data withno nb use of dropout.
 
>> x - input tensor
>> y - output tensor
>> seed - random seed
>> dropProb - probability to set an element to zero
>> leadingDim - the dimension which we generate the random numbers and perform broadcasting
*/
void _Dropout(const XTensor * x, XTensor * y, unsigned int seed, DTYPE dropProb, int leadingDim)
{
    CheckNTErrors(dropProb >= 0.0 && dropProb <= 1.0, "The probability must be 0-1!");

    int n = leadingDim < 0 ? x->order - 1 : leadingDim;

    CheckNTErrors(n >= 0 && n < x->order, "Wrong leadingDim!");

    DTYPE scaleFactor = (DTYPE)1.0 / ((DTYPE)1.0 - dropProb);
    
    /* generate a mask tensor again with special probability */
    int unitNum = x->dimSize[n];
    DTYPE * maskArray = new DTYPE[unitNum];

    srand(seed);
    for (int i = 0; i < unitNum; i++)
        maskArray[i] = RandomBernoulli(dropProb, scaleFactor);

    XTensor * mask = NewTensor1DV2(unitNum, x->dataType, x->devID, x->mem);
    mask->SetData(maskArray, unitNum);

    /* call Multiply function for mask */
    _MultiplyDim(x, mask, y, n, 0);
    
    delete mask;
    delete[] maskArray;
}

/* 
backward computation of the dropout function

dE/dx = dE/dy * dy/dx

>> y - output of the dropout function
>> x - input of the dropout function
>> dedy - dE/dy
>> dedx - dE/dx
>> seed - random seed
>> dropProb - probability to set an element to zero
>> leadingDim - the dimension which we generate the random numbers and perform broadcasting
*/
void _DropoutBackward(const XTensor * y, const XTensor * x, 
                      const XTensor * dedy, XTensor * dedx, 
                      unsigned int seed, DTYPE dropProb, int leadingDim)
{
    CheckNTErrors(dropProb >= 0.0 && dropProb <= 1.0, "The probability must be 0-1!");

    int n = leadingDim < 0 ? x->order - 1 : leadingDim;

    CheckNTErrors(n >= 0 && n < x->order, "Wrong leadingDim!");

    if(x->dataType == DEFAULT_DTYPE && y->dataType == DEFAULT_DTYPE)
    {
        DTYPE scaleFactor = (DTYPE)1.0F / ((DTYPE)1.0F - dropProb);

        /* generate a mask tensor again with special probability */
        int unitNum = x->dimSize[n];
        DTYPE * maskArray = new DTYPE[unitNum];
        
        srand(seed);
        for (int i = 0; i < unitNum; i++)
            maskArray[i] = RandomBernoulli(dropProb, scaleFactor);

        XTensor * mask = NewTensor1DV2(unitNum, x->dataType, x->devID, x->mem);
        mask->SetData(maskArray, unitNum);

        /* call MultiplyDim function for mask */
        _MultiplyDim(dedy, mask, dedx, n, 0);

        delete mask;
        delete[] maskArray;
    }
    else
        ShowNTErrors("TODO!");
}

/* 
dropout function (we make tensor connections here)
It randomly zeroes some of the elements of the input tensor
with probability p via a Bernoulli distribution.
 
See "Improving neural networks by preventing co-adaptation of feature detectors"
for more details.
 
Here, the output is scaled by a factor of \frac{1}{1-p} so that we do not need
to mark the tensor with probability p in the inference phase. Instead we perform
the same inference procedure as that with no use of dropout on the test data.

>> x - input tensor
>> dropProb - probability to set an element to zero
>> leadingDim - the dimension which we generate the random numbers and perform broadcasting
>> leadingDim2 - another dimension which we generate the random numbers and perform broadcasting
<< return - tensor after dropout
*/
XTensor Dropout(const XTensor &x, DTYPE dropProb, int leadingDim, int leadingDim2)
{
    CheckNTErrors(dropProb >= 0.0 && dropProb <= 1.0, "The probability must be 0-1!");

    XTensor mask;
    DTYPE * maskArray = NULL;
    DTYPE scaleFactor = (DTYPE)1.0 / ((DTYPE)1.0 - dropProb);

    if(leadingDim < 0 && leadingDim2 < 0){
        XTensor mask;
        InitTensorV2(&mask, &x);

        _SetDataRandP(&mask, 0, 1.0F, dropProb, scaleFactor);

        return Multiply(x, mask);

        /* dropout with index */
        /*int unitNum = floor(x.unitNum*dropProb);
        maskArrayInt = new int[unitNum];

        for (int i = 0; i < unitNum; i++)
            maskArrayInt[i] = rand() % x.unitNum;

        XTensor maskindex;
        InitTensor1DV2(&maskindex, unitNum, X_INT, x.devID, x.mem);

        maskindex.SetData(maskArrayInt, unitNum);

        delete[] maskArrayInt;

        return DropoutWithIndex(x, maskindex, scaleFactor);*/

    }
    else if(leadingDim2 < 0){
        int n = leadingDim;

        CheckNTErrors(n >= 0 && n < x.order, "Wrong leadingDim!");

        /* generate a mask tensor with probability p */
        int unitNum = x.dimSize[n];
        maskArray = new DTYPE[unitNum];

        //srand((unsigned int)time(NULL));
        for (int i = 0; i < unitNum; i++)
            maskArray[i] = RandomBernoulli(dropProb, scaleFactor);
    
        XTensor mask;
        InitTensor1DV2(&mask, unitNum, x.dataType, x.devID, x.mem);
        mask.SetData(maskArray, unitNum);

        delete[] maskArray;
    
        return MultiplyDim(x, mask, n);
    }
    else{
        int n = leadingDim;
        int m = leadingDim2;

        CheckNTErrors(n >= 0 && n < x.order, "Wrong leadingDim!");
        CheckNTErrors(m >= 0 && m < x.order, "Wrong leadingDim!");
    
        /* generate a mask tensor with probability p */
        int unitNum = x.dimSize[n] * x.dimSize[m];
        maskArray = new DTYPE[unitNum];

        //srand((unsigned int)time(NULL));
        for (int i = 0; i < unitNum; i++)
            maskArray[i] = RandomBernoulli(dropProb, scaleFactor);

        int dims[MAX_TENSOR_DIM_NUM];

        for(int i = 0; i < x.order; i++)
            dims[i] = 1;
        dims[n] = x.GetDim(n);
        dims[m] = x.GetDim(m);
    
        InitTensorV2(&mask, x.order, dims, x.dataType, x.denseRatio,x.devID, x.mem);
        mask.SetData(maskArray, unitNum);

        delete[] maskArray;
    
        return MultiplyBroadcast(x, mask);
    }

}

/* 
dropout function without broadcast 

>> x - input tensor
>> dropProb - probability to set an element to zero
*/
XTensor DropoutWithoutBroadcast(const XTensor &x, DTYPE dropProb)
{
    CheckNTErrors(dropProb >= 0.0 && dropProb <= 1.0, "The probability must be 0-1!");

    DTYPE scaleFactor = (DTYPE)1.0 / ((DTYPE)1.0 - dropProb);
    
    /* generate a mask tensor with probability p */
    int unitNum = x.unitNum;
    DTYPE * maskArray = new DTYPE[unitNum];

    for (int i = 0; i < unitNum; i++)
        maskArray[i] = RandomBernoulli(dropProb, scaleFactor);
    
    XTensor mask;
    InitTensorV2(&mask, x.order, x.dimSize, x.dataType, x.denseRatio, x.devID, x.mem);
    mask.SetData(maskArray, unitNum);

    delete[] maskArray;
    
    return Multiply(x, mask);
}

} // namespace nts(NiuTrans.Tensor)
