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
 * $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-04-24
 */

#include <math.h>
#include "Loss.h"
#include "Loss.cuh"
#include "../core/getandset/SetData.h"
#include "../core/shape/IsSameShaped.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/*
loss function to measure the "number" of errors
*/

/* 
compute the loss 
>> gold - gold standard
>> output - model prediction
>> LFName - name of loss function
>> isLogOutput - is the output in log scale?
>> leadDim - the leading dimension for the output
>> gBeg - where to start in the gold standard (along the leading dimension)
>> gLen - segment length from gBeg (along the leading dimension)
>> oBeg - where to start in the model output (along the leading dimension)
<< return - error in model prediction with respect to gold standard
*/
DTYPE _LossCompute(XTensor * gold, XTensor * output, LOSS_FUNCTION_NAME LFName,
                  bool isLogOutput, int leadDim, int gBeg, int gLen, int oBeg)
{
    DTYPE error = 0.0F;
    if (output->devID < 0) {
        CheckNTErrors((gLen >= 0 && gLen <= output->unitNum), "Illegal input length!");
        CheckNTErrors((_IsSameShaped(gold, output)), "The input tensors must be of the same size!");
        CheckNTErrors((gold->dimSize[gold->order - 1] == 1 && output->dimSize[output->order - 1] == 1), "TODO!");
        CheckNTErrors((gold->order > leadDim && leadDim >= 0), "Illegal leading dimension!");
        CheckNTErrors((gold->dataType == DEFAULT_DTYPE && output->dataType == DEFAULT_DTYPE), "TODO!");

        int dimensionSize = output->dimSize[leadDim];
        int stride = 1;
        int blockSize = 1;
        int blockNum = 1;

        for(int i = leadDim + 1; i < output->order; i++)
            stride *= output->dimSize[i];
        blockSize = stride * dimensionSize;
        blockNum = output->unitNum / blockSize;

        if(isLogOutput)
            return _LossComputeForLogScale(gold, output, LFName, leadDim, gBeg, gLen, oBeg);

        DTYPE * gp = (DTYPE*)gold->data;
        DTYPE * op = (DTYPE*)output->data;

        /* 
        squared error 
        loss = sum_{i} 0.5*(gold_i - output_i)^2
        where gold_i is the gold standard and output_i is the model prediction
        */
        if(LFName == SQUAREDERROR){
            if(gold->isSparse){
                CheckNTErrors((gBeg == 0 && gLen == dimensionSize), "TODO!");
                for(int i = 0; i < blockSize; i++){
                    DTYPE diff = 0 - *(op + oBeg + i);
                    error += (DTYPE)0.5 * diff * diff;
                }
                int num = gold->GetNonzeroSize();
                for(int i = 0; i < num; i++){
                    int key = gold->GetKeyInSparse(i);
                    DTYPE value = gold->GetInSparse(i);
                    int offset = key - gBeg;
                    DTYPE diff = value - *(op + oBeg + offset);
                    error += (DTYPE)0.5 * diff * diff;
                    DTYPE diff2 = 0 - *(op + oBeg + offset);
                    error -= (DTYPE)0.5 * diff2 * diff2;
                }
            }
            else{
                for(int k = 0; k < blockNum; k++){
                    int bg = k * blockSize + gBeg * stride;
                    int og = k * blockSize + oBeg * stride;
                    int size = stride * gLen;
                    for(int i = 0; i < size; i++){
                        DTYPE diff = *(gp + bg + i) - *(op + og + i);
                        error += (DTYPE)0.5 * diff * diff;
                    }
                }
            }
        }

        /* 
        cross entropy
        loss = sum_{i} (-gold_i * log(output_i))
        where gold and output are distributions 
        */
        if(LFName == CROSSENTROPY){
            if(gold->isSparse){
                CheckNTErrors((gBeg == 0 && gLen == dimensionSize), "TODO!");
                int num = gold->GetNonzeroSize();
                for(int i = 0; i < num; i++){
                    int key = gold->GetKeyInSparse(i);
                    DTYPE value = gold->GetInSparse(i);
                    int offset = key - gBeg;
                    error += -value * (DTYPE)log((*(op + oBeg + offset)));
                }
            }
            else{
                for(int k = 0; k < blockNum; k++){
                    int bg = k * blockSize + gBeg * stride;
                    int og = k * blockSize + oBeg * stride;
                    int size = stride * gLen;
                    for(int i = 0; i < size; i++){
                        error += -(*(gp + bg + i)) * (DTYPE)log(*(op + og + i));
                    }
                }
            }
        }
        
        /*
        one hot error
        loss = sum_{i} e_i 
        where e_i = 0.5*(t_i - y_i)^2 if t_i = 1, 
              e_i = 0 otherwise
        */
        if(LFName == ONEHOTERROR){
            if(gold->isSparse){
                CheckNTErrors((gBeg == 0 && gLen == dimensionSize), "TODO!");
                for(int i = 0; i < blockSize; i++){
                    DTYPE diff = 0 - *(op + oBeg + i);
                    error += (DTYPE)0.5 * diff * diff;
                }
                int num = gold->GetNonzeroSize();
                for(int i = 0; i < num; i++){
                    int key = gold->GetKeyInSparse(i);
                    DTYPE value = gold->GetInSparse(i);
                    int offset = key - gBeg;

                    if(value >= 1.0F)
                        continue;

                    DTYPE diff0 = 0 - *(op + oBeg + offset);
                    error += (DTYPE)0.5 * diff0 * diff0;
                    DTYPE diff = value - *(op + oBeg + offset);
                    error += (DTYPE)0.5 * diff * diff;
                    DTYPE diff2 = 0 - *(op + oBeg + offset);
                    error -= (DTYPE)0.5 * diff2 * diff2;
                }
            }
            else{
                for(int k = 0; k < blockNum; k++){
                    int size = stride * gLen;
                    for(int i = 0; i < size; i++){
                        if(*(gp + gBeg + i) < 1.0F)
                            continue;
                        DTYPE diff = *(gp + gBeg + i) - *(op + oBeg + i);
                        error += (DTYPE)0.5 * diff * diff;
                    }
                }
            }
        }
    }
    else {
#ifdef USE_CUDA
        error = _CudaLossCompute(gold, output, LFName, isLogOutput, leadDim, gBeg, gLen, oBeg);
#else
        ShowNTErrors("Please specify USE_CUDA and recompile the code!");
#endif
    }

    return error;
}

/* 
the log version of loss computation

>> gold - gold standard
>> output - model prediction
>> LFName - name of loss function
>> leadDim - the leading dimension for the output
>> gBeg - where to start in the gold standard (along the leading dimension)
>> gLen - segment length from gBeg (along the leading dimension)
>> oBeg - where to start in the model output (along the leading dimension)
<< return - error in model prediction with respect to gold standard
*/
DTYPE _LossComputeForLogScale(XTensor * gold, XTensor * output, 
                             LOSS_FUNCTION_NAME LFName,
                             int leadDim, int gBeg, int gLen, int oBeg)
{
    CheckNTErrors(gLen >= 0 && gLen <= output->unitNum, "Illegal input length!");
    CheckNTErrors(_IsSameShaped(gold, output), "The input tensors must be of the same size!");
    CheckNTErrors(gold->dimSize[gold->order - 1] == 1 && output->dimSize[output->order - 1] == 1, "TODO!");
    CheckNTErrors(gold->order > leadDim && leadDim >= 0, "Illegal leading dimension!");
    CheckNTErrors(gold->dataType == DEFAULT_DTYPE && output->dataType == DEFAULT_DTYPE, "TODO!");

    int dimensionSize = output->dimSize[leadDim];
    int stride = 1;
    int blockSize = 1;
    int blockNum = 1;

    for(int i = leadDim + 1; i < output->order; i++)
        stride *= output->dimSize[i];
    blockSize = stride * dimensionSize;
    blockNum = output->unitNum / blockSize;

    DTYPE * gp = (DTYPE*)gold->data;
    DTYPE * op = (DTYPE*)output->data;
    DTYPE error = 0.0F;

    /* 
    squared error 
    loss = sum_{i} 0.5*(gold_i - exp(output_i))^2
    where gold_i is the gold standard and output_i is the model prediction
    */
    if(LFName == SQUAREDERROR){
        if(gold->isSparse){
            CheckNTErrors((gBeg == 0 && gLen == dimensionSize), "TODO!");
            for(int i = 0; i < gLen; i++){
                DTYPE diff = 0 - (DTYPE)exp(*(op + oBeg + i));
                error += (DTYPE)0.5 * diff * diff;
            }
            int num = gold->GetNonzeroSize();
            for(int i = 0; i < num; i++){
                int key = gold->GetKeyInSparse(i);
                DTYPE value = gold->GetInSparse(i);
                int offset = key - gBeg;
                DTYPE diff = value - (DTYPE)exp(*(op + oBeg + offset));
                error += (DTYPE)0.5 * diff * diff;
                DTYPE diff2 = 0 - (DTYPE)exp(*(op + oBeg + offset));
                error -= (DTYPE)0.5 * diff2 * diff2;
            }
        }
        else{
            for(int k = 0; k < blockNum; k++){
                int bg = k * blockSize + gBeg * stride;
                int og = k * blockSize + oBeg * stride;
                int size = stride * gLen;
                for(int i = 0; i < size; i++){
                    DTYPE diff = *(gp + bg + i) - (DTYPE)exp(*(op + og + i));
                    error += (DTYPE)0.5 * diff * diff;
                }
            }
        }
    }

    /* 
    cross entropy
    loss = sum_{i} (-t_i * y_i), where t and y are distributions 
    */
    if(LFName == CROSSENTROPY){
        if(gold->isSparse){
            CheckNTErrors((gBeg == 0 && gLen == dimensionSize), "TODO!");
            int num = gold->GetNonzeroSize();
            for(int i = 0; i < num; i++){
                int key = gold->GetKeyInSparse(i);
                DTYPE value = gold->GetInSparse(i);
                int offset = key - gBeg;
                error += -value * (*(op + oBeg + offset));
            }
        }
        else{
            for(int k = 0; k < blockNum; k++){
                int bg = k * blockSize + gBeg * stride;
                int og = k * blockSize + oBeg * stride;
                int size = stride * gLen;
                for(int i = 0; i < size; i++){
                    error += -(*(gp + bg + i)) * (*(op + og + i));
                }
            }
        }
    }

    /*
    one hot error
    loss = sum_{i} e_i 
    where e_i = 0.5*(t_i - exp(y_i))^2 if t_i = 1, 
          e_i = 0 otherwise
    */
    if(LFName == ONEHOTERROR){
        if(gold->isSparse){
            CheckNTErrors((gBeg == 0 && gLen == dimensionSize), "TODO!");
            int num = gold->GetNonzeroSize();
            for(int i = 0; i < num; i++){
                int key = gold->GetKeyInSparse(i);
                DTYPE value = gold->GetInSparse(i);
                int offset = key - gBeg;
                if(value >= 1.0F)
                    continue;

                DTYPE diff0 = 0 - (DTYPE)exp(*(op + oBeg + offset));
                error += (DTYPE)0.5 * diff0 * diff0;
                DTYPE diff = value - (DTYPE)exp(*(op + oBeg + offset));
                error += (DTYPE)0.5 * diff * diff;
                DTYPE diff2 = 0 - (DTYPE)exp(*(op + oBeg + offset));
                error -= (DTYPE)0.5 * diff2 * diff2;
            }
        }
        else{
            for(int k = 0; k < blockNum; k++){
                int bg = k * blockSize + gBeg * stride;
                int og = k * blockSize + oBeg * stride;
                int size = stride * gLen;
                for(int i = 0; i < size; i++){
                    if(*(gp + gBeg + i) >= 1.0F)
                        continue;

                    DTYPE diff = *(gp + bg + i) - (DTYPE)exp(*(op + og + i));
                    error += (DTYPE)0.5 * diff * diff;
                }
            }
        }
    }

    return error;
}

/* 
backward compuation for a single element 
dE/dy
where E is the error(loss) function that measure the errors in y
with respect to gold standard, and y this the model output
>> t - gold standard
>> y - model output
>> LFName - name of loss function
<< return dE/dy
*/
DTYPE _LossBackwardPoint(DTYPE t, DTYPE y, LOSS_FUNCTION_NAME LFName)
{
    /* 
    squared error 
    loss = sum_{i} 0.5*(t_i - y_i)^2, where t_i is the gold standard and y_i is the model output
    dloss/dy_i = y_i - t_i
    */
    if(LFName == SQUAREDERROR){
        return y - t;
    }

    /* 
    cross entropy
    loss = sum_{i} (-t_i * log(y_i)), where t and y are distributions 
    dloss/dy_i = -t_i / y_i
    */
    if(LFName == CROSSENTROPY){
        return -t/y;
    }

    return 1;
}

/* 
backward compuation for (dense) vectors 
dE/dy
where E is the error(loss) function that measure the errors in y
with respect to gold standard, and y this the model output
>> dedy - dE/dy (for return)
>> t - gold standard (in vector/matrix)
>> y - model output (in vector/matrix)
>> LFName - name of loss function
>> leadDim - the leading dimension for the output
>> tBeg - where to start in the gold standard (along the leading dimension)
>> tLen - segment length from tBeg (along the leading dimension)
>> yBeg - where to start in the model output (along the leading dimension)
*/
void _LossBackward(XTensor * dedy, XTensor * t, XTensor * y, 
                  LOSS_FUNCTION_NAME LFName, 
                  int leadDim, int tBeg, int tLen, int yBeg)
{
    if(t == NULL){
        if(dedy->dataType == X_FLOAT)
            _SetDataFixedFloat(dedy, 1.0F);
        else if(dedy->dataType == X_DOUBLE)
            _SetDataFixedDouble(dedy, 1.0);
        else if(dedy->dataType == X_INT)
            _SetDataFixedInt(dedy, 1);
        else{
            ShowNTErrors("TODO");
        }
        return;
    }

    if(t->order < 0)
        return;
    
    if (y->devID < 0) {
        CheckNTErrors(tLen <= y->unitNum, "Illegal input length!");
        CheckNTErrors(_IsSameShaped(t, y)&& _IsSameShaped(dedy, y),
                     "The input tensors must be of the same size!");
        CheckNTErrors((dedy->devID == t->devID) && (dedy->devID == y->devID),
                     "Tensor must be on the same device!");
        CheckNTErrors(t->order > leadDim, "Illegal leading dimension!");
        CheckNTErrors(t->dataType == DEFAULT_DTYPE && y->dataType == DEFAULT_DTYPE, "TODO!");

        if (leadDim < 0) {
            leadDim = 0;
            tBeg = 0;
            yBeg = 0;
            tLen = y->dimSize[leadDim];
        }

        int dimensionSize = y->dimSize[leadDim];
        int stride = 1;
        int blockSize = 1;
        int blockNum = 1;

        for(int i = leadDim + 1; i < y->order; i++)
            stride *= y->dimSize[i];
        blockSize = stride * dimensionSize;
        blockNum = y->unitNum / blockSize;

        DTYPE * tp = (DTYPE*)t->data;
        DTYPE * yp = (DTYPE*)y->data;
        DTYPE * dedyp = (DTYPE*)dedy->data;

        CheckNTErrors((t->dataType == DEFAULT_DTYPE && 
                       y->dataType == DEFAULT_DTYPE && 
                       dedy->dataType == DEFAULT_DTYPE),
                       "Input vectors are not in default type!");

        /* 
        squared error 
        loss = sum_{i} 0.5*(t_i - y_i)^2, where t_i is the gold standard and y_i is the model output
        dloss/dy_i = y_i - t_i
        */
        if(LFName == SQUAREDERROR){
            if(t->isSparse){
                CheckNTErrors((tBeg == 0 && tLen == dimensionSize), "TODO!");
                int num = t->GetNonzeroSize();
                for(int i = 0; i < num; i++){
                    int key = t->GetKeyInSparse(i);
                    DTYPE value = t->GetInSparse(i);
                    if(key >= tBeg && key < tBeg + tLen)
                        *(dedyp + yBeg + key - tBeg) = -value;
                }
                for(int i = 0; i < tLen; i++){
                    *(dedyp + yBeg + i) += *(yp + yBeg + i);
                }
            }
            else{
                for(int k = 0; k < blockNum; k++){
                    int bg = k * blockSize + tBeg * stride;
                    int yg = k * blockSize + yBeg * stride;
                    int size = stride * tLen;
                    for(int i = 0; i < size; i++){
                        *(dedyp + bg + i) = *(yp + yBeg + i) - *(tp + yg + i);
                    }
                }
            }
        }

        /* 
        cross entropy
        loss = sum_{i} (-t_i * log(y_i)), where t and y are distributions 
        dloss/dy_i = -t_i / y_i
        */
        if(LFName == CROSSENTROPY){
            if(t->isSparse){
                memset(dedyp + yBeg, 0, sizeof(DTYPE) * tLen);
                int num = t->GetNonzeroSize();
                for(int i = 0; i < num; i++){
                    int key = t->GetKeyInSparse(i);
                    DTYPE value = t->GetInSparse(i);
                    if(key >= tBeg && key < tBeg + tLen)
                        *(dedyp + yBeg + key - tBeg) = -value/(DTYPE)*(yp + yBeg + key - tBeg);
                }
            }
            else{
                for (int i = 0; i < blockNum; i++) {
                    for (int j = 0; j < stride; j++) {
                        for (int k = 0; k < tLen; k++) {
                            *(dedyp + i * stride * dimensionSize + j + stride * (yBeg + k)) = 
                            -(DTYPE)*(tp + i * stride * dimensionSize + j + stride * (tBeg + k)) / 
                             (DTYPE)*(yp +  i * stride * dimensionSize + j + stride * (yBeg + k));
                        }
                    }
                }
            }
        }
    }
    else {
#ifdef USE_CUDA
        _CudaLossBackward(dedy, t, y, LFName, leadDim, tBeg, tLen, yBeg);
#else
        ShowNTErrors("Please specify USE_CUDA and recompile the code!");
#endif
    }
}

} // namespace nts(NiuTrans.Tensor)
