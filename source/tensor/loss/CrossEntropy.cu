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

#ifndef __CROSSENTROPY_CUH__
#define __CROSSENTROPY_CUH__

#include "../XTensor.h"
#include "../XDevice.h"
#include "CrossEntropy.cuh"
#include "CrossEntropy.h"
#include "../core/arithmetic/Div.h"
#include "../core/arithmetic/Multiply.h"
#include "../core/arithmetic/MultiplyDim.h"
#include "../core/math/Unary.h"
#include "../core/math/ScaleAndShift.h"
#include "../core/reduce/ReduceSum.h"
#include "../core/reduce/ReduceSumAll.h"
#include "../core/shape/Transpose.h"
#include "../core/shape/Unsqueeze.h"
#include "../core/shape/IsSameShaped.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/*
compute the cross entropy loss (cuda version) 
loss = sum_{i} (-gold_i * log(output_i))
where gold and output are distributions 
        
>> output - model prediction
>> gold - gold standard
>> loss - compute loss
>> weight - a rescaling weight given to each class
>> padding - specify a target value that is ignored and does not contribute to the loss computation
>> leadingDim - the leading dimension for the output
*/
void _CudaCrossEntropyFast(const XTensor * output, const XTensor * gold,
                           XTensor * loss, const XTensor * weight, 
                           const XTensor * padding, int leadingDim)
{
    int n = leadingDim < 0 ? output->order - 1 : leadingDim;
    
    XTensor * interBuf1 = NewTensorBufV2(output, output->devID, output->mem);
    XTensor * interBuf2 = NewTensorBufV2(output, output->devID, output->mem);
    
    _Log(output, interBuf1);
    _Multiply(gold, interBuf1, interBuf2);

    if(weight != NULL)
        _MultiplyDimMe(interBuf2, weight, n);
    _NegateMe(interBuf2);
    _ReduceSum(interBuf2, loss, n);
    
    if(padding != NULL)
        _MultiplyMe(loss, padding);

    DelTensorBuf(interBuf2);
    DelTensorBuf(interBuf1);
}

/*
compute the cross entropy loss (scalar version) 

loss = sum_{i} (-gold_i * log(output_i))
where gold and output are distributions 
        
>> output - model prediction
>> gold - gold standard
>> reduceWay - loss compute way, sum or mean
>> weight - a rescaling weight given to each class
>> padding - specify a target value that is ignored and does not contribute to the loss computation
>> leadingDim - the leading dimension for the output
<< return - the cross entropy loss that is a scalar
*/
DTYPE _CudaCrossEntropyFast(const XTensor * output, const XTensor * gold,
                            LOSS_COMPUTE_WAY reduceWay, const XTensor * weight,
                            const XTensor * padding, int leadingDim)
{
    DTYPE loss = 0;

    int order = output->order;
    int n = leadingDim < 0 ? output->order - 1 : leadingDim;
    int leadingDimSize = output->GetDim(n);

    CheckNTErrors(n >= 0 && n < output->order, 
                 "Wrong leadingDim!");
    CheckNTErrors(_IsSameShaped(output, gold), 
                 "The output tensor and gold tensor must be of the same size!");
    CheckNTErrors(weight == NULL || weight->unitNum == leadingDimSize, 
                 "Wrong weight tensor!");
    CheckNTErrors(padding == NULL || padding->order == output->order - 1, 
                 "Wrong padding tensor!");
    CheckNTErrors(gold->dataType == DEFAULT_DTYPE && output->dataType == DEFAULT_DTYPE, 
                 "TODO!");
    
    int * dimSize = new int[output->order - 1];
    for (int i = 0; i < order; i++) {
        if(i < n)
            dimSize[i] = output->dimSize[i];
        else if(i > n)
            dimSize[i - 1] = output->dimSize[i];
    }

    XTensor * lossBuf = NewTensorBufV2(output->order - 1, dimSize, output->dataType, output->denseRatio, 
                                     output->devID, output->mem);

    _CudaCrossEntropyFast(output, gold, lossBuf, weight, padding, leadingDim);

    loss = _ReduceSumAll(lossBuf);

    if(reduceWay == REDUCE_MEAN) {
        int nonZeroNum;
        if(padding == NULL) {
            nonZeroNum = lossBuf->unitNum;
        }
        else {
            XTensor * tmp = NewTensorBufV2(padding, padding->devID, padding->mem);
            _IsNonZero(padding, tmp);
            nonZeroNum = (int)_ReduceSumAll(tmp);
            DelTensorBuf(tmp);
        }

        loss = loss / (DTYPE)nonZeroNum;
    }
    else if(reduceWay == REDUCE_SUM) {
        /* don't need to do anything */
    }
    else {
        ShowNTErrors("TODO");
    }

    delete[] dimSize;
    DelTensorBuf(lossBuf);

    return loss;
}

/* 
backward computation of cross entropy function 

loss = sum_{i} (-t_i * log(y_i))
dE/dy_i = -t_i / y_i
where E is the error(loss) function that measure the errors in y
with respect to gold standard, and y this the model output

>> dedy - dE/dy (for return)
>> output - model prediction
>> gold - gold standard
>> weight - a rescaling weight given to each class
>> padding - specify a target value that is ignored and does not contribute to the loss computation
>> leadingDim - the leading dimension for the output
*/
void _CudaCrossEntropyBackward(XTensor * dedy, const XTensor * output, 
                               const XTensor * gold, const XTensor * weight,
                               XTensor * padding, int leadingDim)
{
    int n = leadingDim < 0 ? output->order - 1 : leadingDim;
    
    _Div(gold, output, dedy);
    _NegateMe(dedy);
    if(weight != NULL)
        _MultiplyDimMe(dedy, weight, n);
    if(padding != NULL) {
        int paddingOrder = padding->order;
        int * paddingDims = new int[paddingOrder];
        memcpy(paddingDims, padding->dimSize, padding->order * sizeof(int));
        padding->Reshape(padding->unitNum);

        int order = dedy->order;
        int * dims = new int[order];
        memcpy(dims, dedy->dimSize, dedy->order * sizeof(int));
        dedy->Reshape(dedy->unitNum/dedy->GetDim(n), dedy->GetDim(n));
        _MultiplyDimMe(dedy, padding, 0);

        padding->Reshape(paddingOrder, paddingDims);
        dedy->Reshape(order, dims);

        delete[] paddingDims;
        delete[] dims;
    }

    if(padding != NULL) {
        XTensor * tmp = NewTensor(padding);
        _IsNonZero(padding, tmp);
        int nonZeroNum = (int)_ReduceSumAll(tmp);
        _ScaleAndShiftMe(dedy, (DTYPE)1.0/(DTYPE)nonZeroNum);
        delete tmp;
    }
    else {
        int num = dedy->unitNum / dedy->GetDim(n);
        _ScaleAndShiftMe(dedy, (DTYPE)1.0/(DTYPE)num);
    }

}

} // namespace nts(NiuTrans.Tensor)

#endif // __CROSSENTROPY_CUH__