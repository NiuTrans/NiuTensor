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
#include "CrossEntropy.h"
#include "CrossEntropy.cuh"
#include "../XTensor.h"
#include "../XName.h"
#include "../core/arithmetic/MultiplyDim.h"
#include "../core/arithmetic/Multiply.h"
#include "../core/math/Unary.h"
#include "../core/math/ScaleAndShift.h"
#include "../core/reduce/ReduceSum.h"
#include "../core/reduce/ReduceSumAll.h"
#include "../core/shape/IsSameShaped.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/*
compute the cross entropy loss

loss = sum_{i} (-gold_i * log(output_i))
where gold and output are distributions 
        
>> output - model prediction
>> gold - gold standard
>> loss - compute loss
>> weight - a rescaling weight given to each class
>> padding - specify a target value that is ignored and does not contribute to the loss computation
>> leadingDim - the leading dimension for the output
*/
void _CrossEntropy(const XTensor * output, const XTensor * gold,
                   XTensor * loss, const XTensor * weight, 
                   const XTensor * padding, int leadingDim)
{
    int n = leadingDim < 0 ? output->order - 1 : leadingDim;
    int unitNum = output->dimSize[n];

    CheckNTErrors(n >= 0 && n < output->order, "Wrong leadingDim!");
    CheckNTErrors(_IsSameShaped(output, gold), 
                 "The output tensor and gold tensor must be of the same size!");
    CheckNTErrors(weight == NULL || weight->unitNum == unitNum, "Wrong weight tensor!");
    CheckNTErrors(padding == NULL || _IsSameShaped(padding, loss), 
                 "The loss tensor and padding tensor must be same shape!");
    CheckNTErrors(loss->order == output->order - 1, "Wrong loss dimension!");
    CheckNTErrors(gold->dataType == DEFAULT_DTYPE && output->dataType == DEFAULT_DTYPE, "TODO!");

    XTensor * inter = NewTensor(output);
    
    _Log(output, inter);
    _MultiplyMe(inter, gold);

    if(weight != NULL)
        _MultiplyDimMe(inter, weight, n);
    _NegateMe(inter);
    _ReduceSum(inter, loss, n);

    if(padding != NULL)
        _MultiplyMe(loss, padding);

    DelTensor(inter);
}

/*
compute the cross entropy loss (faster implementation with optimized code)

loss = sum_{i} (-gold_i * log(output_i))
where gold and output are distributions 
        
>> output - model prediction
>> gold - gold standard
>> loss - compute loss
>> weight - a rescaling weight given to each class
>> padding - specify a target value that is ignored and does not contribute to the loss computation
>> leadingDim - the leading dimension for the output
*/
void _CrossEntropyFast(const XTensor * output, const XTensor * gold,
                       XTensor * loss, const XTensor * weight,
                       const XTensor * padding, int leadingDim)
{
    int order = output->order;
    int n = leadingDim < 0 ? output->order - 1 : leadingDim;
    int leadingDimSize = output->GetDim(n);

    CheckNTErrors(n >= 0 && n < output->order, 
                 "Wrong leading dimension!");
    CheckNTErrors(_IsSameShaped(output, gold), 
                 "The output tensor and gold tensor must be of the same size!");
    CheckNTErrors(weight == NULL || weight->unitNum == leadingDimSize, 
                 "Wrong weight tensor!");
    CheckNTErrors(padding == NULL || _IsSameShaped(padding, loss), 
                 "The loss tensor and padding tensor must be same shape!");
    CheckNTErrors(loss->order == output->order - 1, 
                 "Wrong loss dimension!");
    CheckNTErrors(gold->dataType == DEFAULT_DTYPE && output->dataType == DEFAULT_DTYPE, 
                 "TODO!");
    
    for(int i = 0; i < order; i++){
        if(i < n){
            CheckNTErrors((output->GetDim(i) == loss->GetDim(i)), "Unmatched tensors!");
        }
        else if(i > n){
            CheckNTErrors((output->GetDim(i) == loss->GetDim(i - 1)), "Unmatched tensors!");
        }
    }

#ifdef USE_CUDA
    if(output->devID >= 0) {
        _CudaCrossEntropyFast(output, gold, loss, weight, padding, leadingDim);
        return;
    }
#endif

    int blockNum = 1;
    int blockSize = 1;
    int stride = 1;

    for(int i = n + 1; i < order; i++)
        stride *= output->GetDim(i);
    
    blockSize = stride * leadingDimSize;
    blockNum = output->unitNum / blockSize;

    DTYPE * outputData = (DTYPE*)output->data;
    DTYPE * goldData = (DTYPE*)gold->data;
    DTYPE * lossData = (DTYPE*)loss->data;

    DTYPE tmpLoss;
    int lossPos;
    int goldPos;

    if(weight == NULL) {
        if(padding == NULL) {
            for(int i = 0; i < blockNum; i++) {
                for(int j = 0; j < stride; j++) {
                    tmpLoss = 0;
                    lossPos = i * stride + j;
                    for(int k = 0; k < leadingDimSize; k++) {
                        goldPos = i * blockSize + j + k * stride;
                        tmpLoss += -(*(goldData + goldPos)) * 
                                    (DTYPE)log(*(outputData + goldPos));
                    }
                    *(lossData + lossPos) = tmpLoss;
                }
            }
        }
        else {
            DTYPE * paddingData = (DTYPE*)padding->data;
            for(int i = 0; i < blockNum; i++) {
                for(int j = 0; j < stride; j++) {
                    lossPos = i * stride + j;
                    if(*(paddingData + lossPos) == 0)
                        *(lossData + lossPos) = 0;
                    else {
                        tmpLoss = 0;
                        for(int k = 0; k < leadingDimSize; k++) {
                            goldPos = i * blockSize + j + k * stride;
                            tmpLoss += -(*(goldData + goldPos)) * 
                                        (DTYPE)log(*(outputData + goldPos));
                        }
                        *(lossData + lossPos) = tmpLoss;
                    }
                }
            }            
        }
    }
    else {
        DTYPE * weightData = (DTYPE*)weight->data;
        if(padding == NULL) {
            for(int i = 0; i < blockNum; i++) {
                for(int j = 0; j < stride; j++) {
                    tmpLoss = 0;
                    lossPos = i * stride + j;
                    for(int k = 0; k < leadingDimSize; k++) {
                        goldPos = i * blockSize + j + k * stride;
                        tmpLoss += -(*(goldData + goldPos)) * 
                                    (DTYPE)log(*(outputData + goldPos)) *
                                    (*(weightData + k));
                    }
                    *(lossData + lossPos) = tmpLoss;                    
                }
            }
        }
        else {
            DTYPE * paddingData = (DTYPE*)padding->data;
            for(int i = 0; i < blockNum; i++) {
                for(int j = 0; j < stride; j++) {
                    lossPos = i * stride + j;
                    if(*(paddingData + lossPos) == 0)
                        *(lossData + lossPos) = 0;
                    else {
                        tmpLoss = 0;
                        for(int k = 0; k < leadingDimSize; k++) {
                            goldPos = i * blockSize + j + k * stride;
                            tmpLoss += -(*(goldData + goldPos)) * 
                                        (DTYPE)log(*(outputData + goldPos)) *
                                        (*(weightData + k));
                        }
                        *(lossData + lossPos) = tmpLoss;
                    }
                }
            }              
        }
    }
}

/*

*/
XTensor GetReduceTensor(const XTensor & input, int dim)
{
    CheckNTErrors(dim >= 0 && dim < input.order, "Illegal dimension to reduce!");

    int order = input.order - 1;
    int * dimSize = new int[order];
    for(int i = 0; i < order; i++){
        if(i < dim)
            dimSize[i] = input.dimSize[i];
        else if(i >= dim)
            dimSize[i] = input.dimSize[i + 1];
    }

    float dr = (!input.isSparse) ? 1.0F : input.denseRatio;
    XTensor output(order, dimSize, input.dataType, dr, input.devID, input.mem);
    output.SetTMPFlag();

    return output;
}

/*
compute the cross entropy loss (return an XTensor structure) 
make a new tensor to keep the result and return it

loss = sum_{i} (-gold_i * log(output_i))
where gold and output are distributions 

>> output - model prediction
>> gold - gold standard
>> loss - compute loss
>> weight - a rescaling weight given to each class
>> padding - specify a target value that is ignored and does not contribute to the loss computation
>> leadingDim - the leading dimension for the output
*/
XTensor CrossEntropy(const XTensor & output, const XTensor & gold,
                     int leadingDim)
{
    int dim = leadingDim < 0 ? output.order - 1 : leadingDim;
    XTensor loss;
    loss = GetReduceTensor(output, dim);

    XTensor * weight = NULL;
    XTensor * padding = NULL;

    /* call _CrossEntropy function */
    _CrossEntropy(&output, &gold, &loss, weight, padding, dim);

    /* tensor connection */
    TensorList tails(4);
    tails.Add((XTensor*)&output);
    tails.Add((XTensor*)&gold);
    tails.Add(weight);
    tails.Add(padding);

    if (output.enableGrad) {
        XLink::MakeLink(&tails, &loss, LOSS_CROSSENTROPY);
        XLink::AddParamToHeadInt(&loss, dim);
    }

    return loss;
}

XTensor CrossEntropy(const XTensor & output, const XTensor & gold,
                     const XTensor & padding,
                     int leadingDim)
{
    int dim = leadingDim < 0 ? output.order - 1 : leadingDim;
    XTensor loss;
    loss = GetReduceTensor(output, dim);

    XTensor * weight = NULL;

    /* call _CrossEntropy function */
    _CrossEntropy(&output, &gold, &loss, weight, &padding, dim);

    /* tensor connection */
    TensorList tails(4);
    tails.Add((XTensor*)&output);
    tails.Add((XTensor*)&gold);
    tails.Add(weight);
    tails.Add((XTensor*)&padding);

    if (output.enableGrad) {
        XLink::MakeLink(&tails, &loss, LOSS_CROSSENTROPY);
        XLink::AddParamToHeadInt(&loss, dim);
    }

    return loss;
}

/*
compute the cross entropy loss
loss = sum_{i} (-gold_i * log(output_i))
where gold and output are distributions 
        
>> output - model prediction
>> gold - gold standard
>> reduce - loss compute way, sum or mean
>> weight - a rescaling weight given to each class
>> padding - specify a target value that is ignored and does not contribute to the loss computation
>> leadingDim - the leading dimension for the output
*/
DTYPE _CrossEntropy(const XTensor * output, const XTensor * gold,
                    LOSS_COMPUTE_WAY reduceWay, const XTensor * weight, 
                    const XTensor * padding, int leadingDim)
{
    DTYPE loss = 0;
    
    int order = output->order;
    int n = leadingDim < 0 ? output->order - 1 : leadingDim;
    int unitNum = output->dimSize[n];
    
    CheckNTErrors(n >= 0 && n < output->order, "Wrong leadingDim!");
    CheckNTErrors(_IsSameShaped(output, gold), 
                 "The output tensor and gold tensor must be of the same size!");
    CheckNTErrors(weight == NULL || weight->unitNum == unitNum, "Wrong weight tensor!");
    CheckNTErrors(padding == NULL || padding->order == output->order - 1, 
                 "The loss tensor and padding tensor must be same shape!");
    CheckNTErrors(gold->dataType == DEFAULT_DTYPE && output->dataType == DEFAULT_DTYPE, "TODO!");

    int * dimSize = new int[order - 1];
    for (int i = 0; i < order; i++) {
        if(i < n)
            dimSize[i] = output->dimSize[i];
        else if(i > n)
            dimSize[i - 1] = output->dimSize[i];
    }

    XTensor * lossBuf = NewTensorBufV2(output->order - 1, dimSize, output->dataType, output->denseRatio, 
                                     output->devID, output->mem);

    _CrossEntropy(output, gold, lossBuf, weight, padding, leadingDim);

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
compute the cross entropy loss (faster implementation with optimized code)

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
DTYPE _CrossEntropyFast(const XTensor * output, const XTensor * gold,
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
    
    if(padding != NULL) {
        for(int i = 0; i < order; i++){
            if(i < n){
                CheckNTErrors((output->GetDim(i) == padding->GetDim(i)), "Unmatched tensors!");
            }
            else if(i > n){
                CheckNTErrors((output->GetDim(i) == padding->dimSize[i - 1]), "Unmatched tensors!");
            }
        }
    }

#ifdef USE_CUDA
    if(output->devID >= 0) {
        return _CudaCrossEntropyFast(output, gold, reduceWay, weight, padding, leadingDim);
    }
#endif

    int blockNum = 1;
    int blockSize = 1;
    int stride = 1;

    for(int i = n + 1; i < order; i++)
        stride *= output->GetDim(i);
    
    blockSize = stride * leadingDimSize;
    blockNum = output->unitNum / blockSize;

    DTYPE * outputData = (DTYPE*)output->data;
    DTYPE * goldData = (DTYPE*)gold->data;

    int paddingPos;
    int goldPos;
    int nonZeroNum = 0;

    if(weight == NULL) {
        if(padding == NULL) {
            nonZeroNum = blockNum * stride;

            for(int i = 0; i < blockNum; i++) {
                for(int j = 0; j < stride; j++) {
                    paddingPos = i * stride + j;
                    for(int k = 0; k < leadingDimSize; k++) {
                        goldPos = i * blockSize + j + k * stride;
                        loss += -(*(goldData + goldPos)) * 
                                 (DTYPE)log(*(outputData + goldPos));
                    }
                }
            }
        }
        else {
            DTYPE * paddingData = (DTYPE*)padding->data;
            for(int i = 0; i < blockNum; i++) {
                for(int j = 0; j < stride; j++) {
                    paddingPos = i * stride + j;
                    if(*(paddingData + paddingPos) == 0)
                        continue;
                    else {
                        nonZeroNum += 1;
                        for(int k = 0; k < leadingDimSize; k++) {
                            goldPos = i * blockSize + j + k * stride;
                            loss += -(*(goldData + goldPos)) * 
                                     (DTYPE)log(*(outputData + goldPos));
                        }    
                    }
                }
            }
        }
    }
    else {
        DTYPE * weightData = (DTYPE*)weight->data;
        if(padding == NULL) {
            nonZeroNum = blockNum * stride;
            for(int i = 0; i < blockNum; i++) {
                for(int j = 0; j < stride; j++) {
                    paddingPos = i * stride + j;
                    for(int k = 0; k < leadingDimSize; k++) {
                        goldPos = i * blockSize + j + k * stride;
                        loss += -(*(goldData + goldPos)) * 
                                 (DTYPE)log(*(outputData + goldPos)) *
                                 (*(weightData + k));
                    }
                }
            }
        }
        else {
            DTYPE * paddingData = (DTYPE*)padding->data;
            for(int i = 0; i < blockNum; i++) {
                for(int j = 0; j < stride; j++) {
                    paddingPos = i * stride + j;
                    if(*(paddingData + paddingPos) == 0)
                        continue;
                    else {
                        nonZeroNum += 1;
                        for(int k = 0; k < leadingDimSize; k++) {
                            goldPos = i * blockSize + j + k * stride;
                            loss += -(*(goldData + goldPos)) * 
                                     (DTYPE)log(*(outputData + goldPos)) *
                                     (*(weightData + j));
                        }    
                    }
                }
            }
        }
    }

    if(reduceWay == REDUCE_MEAN) {
        loss = loss / (DTYPE)nonZeroNum;
    }
    else if(reduceWay == REDUCE_SUM) {
        /* don't need to do anything */
    }
    else {
        ShowNTErrors("TODO");
    }

    return loss;
}

/* 
backward compuation for cross entropy function

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
void _CrossEntropyBackward(XTensor * dedy, const XTensor * output, 
                           const XTensor * gold, const XTensor * weight,
                           XTensor * padding, int leadingDim)
{
    int order = output->order;
    int n = leadingDim < 0 ? output->order - 1 : leadingDim;
    int leadingDimSize = output->GetDim(n);

    CheckNTErrors(n >= 0 && n < output->order, 
                 "Wrong leading dimension!");
    CheckNTErrors(_IsSameShaped(dedy, output, gold), 
                 "The output tensor and gold tensor must be of the same size!");
    CheckNTErrors(weight == NULL || weight->unitNum == leadingDimSize, 
                 "Wrong weight tensor!");
    CheckNTErrors(padding == NULL || padding->order == output->order - 1, 
                 "Wrong padding tensor!");
    CheckNTErrors(gold->dataType == DEFAULT_DTYPE && output->dataType == DEFAULT_DTYPE, 
                 "TODO!");

    if(padding != NULL) {
        for(int i = 0; i < order; i++){
            if(i < n){
                CheckNTErrors((output->GetDim(i) == padding->GetDim(i)), "Unmatched tensors!");
            }
            else if(i > n){
                CheckNTErrors((output->GetDim(i) == padding->dimSize[i - 1]), "Unmatched tensors!");
            }
        }    
    }


#ifdef USE_CUDA
    if(output->devID >= 0) {
        _CudaCrossEntropyBackward(dedy, output, gold, weight, padding, leadingDim);
        return;
    }
#endif

    int blockNum = 1;
    int blockSize = 1;
    int stride = 1;

    for(int i = n + 1; i < order; i++)
        stride *= output->GetDim(i);
    
    blockSize = stride * leadingDimSize;
    blockNum = output->unitNum / blockSize;

    DTYPE * dedyData = (DTYPE*)dedy->data;
    DTYPE * outputData = (DTYPE*)output->data;
    DTYPE * goldData = (DTYPE*)gold->data;

    int paddingPos;
    int goldPos;

    if(weight == NULL) {
        if(padding == NULL) {
            for(int i = 0; i < blockNum; i++) {
                for(int j = 0; j < stride; j++) {
                    for(int k = 0; k < leadingDimSize; k++) {
                        goldPos = i * blockSize + j + k * stride;
                        *(dedyData + goldPos) = -(*(goldData + goldPos)) / 
                                                 (*(outputData + goldPos));
                    }
                }
            }
        }
        else {
            DTYPE * paddingData = (DTYPE*)padding->data;
            for(int i = 0; i < blockNum; i++) {
                for(int j = 0; j < stride; j++) {
                    paddingPos = i * stride + j;
                    for(int k = 0; k < leadingDimSize; k++) {
                        goldPos = i * blockSize + j + k * stride;
                        if(*(paddingData + paddingPos) == 0)
                            *(dedyData + goldPos) = 0;
                        else
                            *(dedyData + goldPos) = -(*(goldData + goldPos)) / 
                                                     (*(outputData + goldPos));
                    }
                }
            }
        }
    }
    else {
        DTYPE * weightData = (DTYPE*)weight->data;
        if(padding == NULL) {
            for(int i = 0; i < blockNum; i++) {
                for(int j = 0; j < stride; j++) {
                    for(int k = 0; k < leadingDimSize; k++) {
                        goldPos = i * blockSize + j + k * stride;
                        *(dedyData + goldPos) = -(*(weightData + k)) * 
                                                 (*(goldData + goldPos)) / 
                                                 (*(outputData + goldPos));
                    }
                }
            }
        }
        else {
            DTYPE * paddingData = (DTYPE*)padding->data;
            for(int i = 0; i < blockNum; i++) {
                for(int j = 0; j < stride; j++) {
                    paddingPos = i * stride + j;
                    for(int k = 0; k < leadingDimSize; k++) {
                        goldPos = i * blockSize + j + k * stride;
                        if(*(paddingData + paddingPos) == 0)
                            *(dedyData + goldPos) = 0;
                        else
                            *(dedyData + goldPos) = -(*(weightData + k)) * 
                                                     (*(goldData + goldPos)) / 
                                                     (*(outputData + goldPos));
                    }
                }
            }
        }
    }

    if(padding != NULL) {
        XTensor * tmp = NewTensor(padding);
        _IsNonZero(padding, tmp);
        int nonZeroNum = (int)_ReduceSumAll(tmp);
        _ScaleAndShiftMe(dedy, (DTYPE)1.0/(DTYPE)nonZeroNum);
        delete tmp;
    }
    else {
        _ScaleAndShiftMe(dedy, (DTYPE)1.0/(DTYPE)blockNum);
    }
}

} // namespace nts(NiuTrans.Tensor)
