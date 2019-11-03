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

#include "Loss.h"
#include "Loss.cuh"
#include "../XDevice.h"
#include "../core/math/ScaleAndShift.h"
#include "../core/math/Unary.h"
#include "../core/math/Binary.h"
#include "../core/arithmetic/Sum.h"
#include "../core/arithmetic/Multiply.h"
#include "../core/reduce/ReduceSum.h"
#include "../core/movement/CopyValues.h"
#include "../core/shape/IsSameShaped.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/*
loss function to measure the "number" of errors
*/

/* 
compute the loss 
>> gold - gold standard
>> y - model prediction
>> LFName - name of loss function
>> isLogOutput - is the output in log scale?
>> leadDim - the leading dimension for the output
>> gBeg - where to start in the gold standard (along the leading dimension)
>> gLen - segment length from oBeg (along the leading dimension)
>> yBeg - where to start in the model output (along the leading dimension)
<< return - error in model prediction with respect to gold standard
*/
DTYPE _CudaLossCompute(XTensor * gold, XTensor * y, LOSS_FUNCTION_NAME LFName,
                      bool isLogOutput, int leadDim, int gBeg, int gLen, int yBeg)
{
    CheckNTErrors((gLen >= 0 && gLen <= y->unitNum), "Illegal input length!");
    CheckNTErrors((_IsSameShaped(gold, y)), "The input tensors must be of the same size!");
    CheckNTErrors((gold->dimSize[gold->order - 1] == 1 && y->dimSize[y->order - 1] == 1), "TODO!");
    CheckNTErrors((gold->order > leadDim && leadDim >= 0), "Illegal leading dimension!");
    CheckNTErrors((gold->dataType == DEFAULT_DTYPE && y->dataType == DEFAULT_DTYPE), "TODO!");
    CheckNTErrors((gold->devID == y->devID), "Tensors must be on the same device!");
    CheckNTErrors((gold->devID >= 0), "Tensors must be on GPU device!");
    CheckNTErrors((gLen == gold->dimSize[leadDim] && gBeg == 0 && yBeg == 0), "TODO!");

    if(isLogOutput)
        return _LossComputeForLogScale(gold, y, LFName, leadDim, gBeg, gLen, yBeg);

    DTYPE error = 0.0F;

    /* 
    squared error 
    loss = sum_{i} 0.5*(gold_i - output_i)^2
    where gold_i is the gold standard and output_i is the model prediction
    */
    if(LFName == SQUAREDERROR){
        XTensor * diff = NewTensorV2(gold->order, gold->dimSize, gold->dataType, gold->denseRatio, gold->devID, gold->mem);
        _Sum(gold, y, diff, -1.0F);
        _PowerMe(diff, 2.0F);
        _ScaleAndShiftMe(diff, 0.5F, 0.0F);

        int reduceTimes = diff->order;
        for (int i = 0; i < reduceTimes; i++) {
            int diffOrder = diff->order - 1;
            int * diffDimSize = new int[diffOrder];
            memcpy(diffDimSize, diff->dimSize + 1, diffOrder * sizeof(int));
            XTensor * diffNew = NewTensorV2(diffOrder, diffDimSize, X_FLOAT, 1.0F, diff->devID, diff->mem);
            int reducePlace = diff->dimSize[0] == 1 ? 1 : 0;
            _ReduceSum(diff, diffNew, reducePlace);
            if (diffNew->order == 1) {
                diffNew->order = 2;
                diffNew->dimSize[1] = diffNew->dimSize[0];
                diffNew->dimSize[0] = 1;
                diffNew->dimSize[diffNew->order - 2] = 1;
            }
            delete diff;
            diff = diffNew;
            delete diffDimSize;
        }
        error = diff->Get2D(0, 0);
        delete diff;
    }

    /* 
    cross entropy
    loss = sum_{i} (-gold_i * log(output_i))
    where gold and output are distributions 
    */
    if(LFName == CROSSENTROPY){
        XTensor * diff = NewTensorV2(y->order, y->dimSize, y->dataType, y->denseRatio, y->devID, y->mem);
        _CopyValues(y, diff);
        _LogMe(diff);
        _Multiply(gold, diff, diff);
        _NegateMe(diff);

        int reduceTimes = diff->order;
        for (int i = 0; i < reduceTimes; i++) {
            int diffOrder = diff->order - 1;
            int * diffDimSize = new int[diffOrder];
            memcpy(diffDimSize, diff->dimSize + 1, diffOrder * sizeof(int));
            XTensor * diffNew = NewTensorV2(diffOrder, diffDimSize, X_FLOAT, 1.0F, diff->devID, diff->mem);
            int reducePlace = diff->dimSize[0] == 1 ? 1 : 0;
            _ReduceSum(diff, diffNew, reducePlace);
            if (diffNew->order == 1) {
                diffNew->order = 2;
                diffNew->dimSize[1] = diffNew->dimSize[0];
                diffNew->dimSize[0] = 1;
                diffNew->dimSize[diffNew->order - 2] = 1;
            }
            delete diff;
            diff = diffNew;
            delete diffDimSize;
        }
        error = diff->Get2D(0, 0);
        delete diff;
    }
    
    /*
    one hot error
    loss = sum_{i} e_i 
    where e_i = 0.5*(t_i - y_i)^2 if t_i = 1, 
          e_i = 0 otherwise
    */
    if(LFName == ONEHOTERROR){
        XTensor * diff = NewTensorV2(gold->order, gold->dimSize, gold->dataType, gold->denseRatio, gold->devID, gold->mem);
        XTensor * yOnehot = NewTensorV2(y->order, y->dimSize, y->dataType, y->denseRatio, y->devID, y->mem);
        _CopyValues(y, yOnehot);
        _Multiply(gold, y, yOnehot);
        _Sum(gold, yOnehot, diff, -1.0F);
        _PowerMe(diff, 2.0F);
        _ScaleAndShiftMe(diff, 0.5F, 0.0F);

        int reduceTimes = diff->order;
        for (int i = 0; i < reduceTimes; i++) {
            int diffOrder = diff->order - 1;
            int * diffDimSize = new int[diffOrder];
            memcpy(diffDimSize, diff->dimSize + 1, diffOrder * sizeof(int));
            XTensor * diffNew = NewTensorV2(diffOrder, diffDimSize, X_FLOAT, 1.0F, diff->devID, diff->mem);
            int reducePlace = diff->dimSize[0] == 1 ? 1 : 0;
            _ReduceSum(diff, diffNew, reducePlace);
            if (diffNew->order == 1) {
                diffNew->order = 2;
                diffNew->dimSize[1] = diffNew->dimSize[0];
                diffNew->dimSize[0] = 1;
                diffNew->dimSize[diffNew->order - 2] = 1;
            }
            delete diff;
            diff = diffNew;
            delete diffDimSize;
        }
        error = diff->Get2D(0, 0);
        delete diff;
        delete yOnehot;
    }
    return error;

    // TODO: call cuda kernels for computing the errors
}

/* 
the log version of loss computation

>> gold - gold standard
>> y - model prediction
>> LFName - name of loss function
>> leadDim - the leading dimension for the output
>> gBeg - where to start in the gold standard (along the leading dimension)
>> gLen - segment length from oBeg (along the leading dimension)
>> yBeg - where to start in the model output (along the leading dimension)
<< return - error in model prediction with respect to gold standard
*/
DTYPE _CudaLossComputeForLogScale(XTensor * gold, XTensor * y, 
                                 LOSS_FUNCTION_NAME LFName,
                                 int leadDim, int gBeg, int gLen, int yBeg)
{
    return 0;

    // TODO: call cuda kernels for computing the errors
}

/* 
backward compuation for a single element (Cuda version)
dE/dy
where E is the error(loss) function that measure the errors in y
with respect to gold standard, and y this the model output
>> t - gold standard
>> y - model output
>> LFName - name of loss function
<< return dE/dy
*/
DTYPE _CudaLossBackward(DTYPE t, DTYPE y, LOSS_FUNCTION_NAME LFName)
{
    return _LossBackwardPoint(t, y, LFName);
   
    // TODO: call cuda kernels for computing the errors
}

/* 
backward compuation for squared error (Cuda kernel)
>> dedy - dE/dy (for return)
>> t - gold standard (in vector)
>> y - model output (in vector)
>> size - size of the vector (dedy)
*/
__global__ 
void KernelLossBackwardSquaredError(DTYPE * dedy, DTYPE * t, DTYPE * y, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size){
        dedy[i] = y[i] - t[i];
    }
}

/* 
backward compuation of blocks for squared error (Cuda kernel)
>> dedy - dE/dy (for return)
>> t - gold standard (in vector)
>> y - model output (in vector)
>> blockSize - size of a block
>> begInBlock - the begining position in a block for computation 
>> lenInBlock - number of items in a block for computation 
>> size - size of the vector (dedy)
*/
__global__ 
void KernelLossBackwardSquaredErrorBlock(DTYPE * dedy, DTYPE * t, DTYPE * y, 
                                         int blockSize, int begInBlock, int lenInBlock, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    int offset = i % blockSize;

    if(offset < begInBlock || offset >= begInBlock + lenInBlock)
        return;

    if (i < size){
        dedy[i] = y[i] - t[i];
    }
}

/* 
backward compuation for cross entropy (Cuda kernel)
>> dedy - dE/dy (for return)
>> t - gold standard (in vector)
>> y - model output (in vector)
>> size - size of the vector (dedy)
*/
__global__ 
void KernelLossBackwardCrossEntropy(DTYPE * dedy, DTYPE * t, DTYPE * y, int tBeg, int tLen, int yBeg, int blockNum, int stride, int dimensionSize)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i > stride * dimensionSize * blockNum) 
        return;

    int blockNumIndex = i / (stride * dimensionSize);
    int blockNumTail = i % (stride * dimensionSize);
    int dimensionSizeIndex = blockNumTail / stride;
    int strideIndex = blockNumTail % stride;

    if (dimensionSizeIndex >= tLen)
        return;

    dedy[blockNumIndex * stride * dimensionSize + strideIndex + stride * (yBeg + dimensionSizeIndex)] = -t[blockNumIndex * stride * dimensionSize + 
        strideIndex + stride * (tBeg + dimensionSizeIndex)] / y[blockNumIndex * stride * dimensionSize + strideIndex + stride * (yBeg + dimensionSizeIndex)];
    /*if (i < size){
        dedy[i] =  -t[i]/y[i];
    }*/
}

/* 
backward compuation for cross entropy (Cuda kernel)
>> dedy - dE/dy (for return)
>> t - gold standard (in vector)
>> y - model output (in vector)
>> blockSize - size of a block
>> begInBlock - the begining position in a block for computation 
>> lenInBlock - number of items in a block for computation 
>> size - size of the vector (dedy)
*/
__global__ 
void KernelLossBackwardCrossEntropyBlock(DTYPE * dedy, DTYPE * t, DTYPE * y, 
                                         int blockSize, int begInBlock, int lenInBlock, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    int offset = i % blockSize;

    if(offset < begInBlock || offset >= begInBlock + lenInBlock)
        return;

    if (i < size){
        dedy[i] =  -t[i]/y[i];
    }
}

/* 
backward compuation for (dense) vectors (Cuda version)
dE/dy
where E is the error(loss) function that measure the errors in y
with respect to gold standard, and y this the model output
>> dedy - dE/dy (for return)
>> t - gold standard (in vector)
>> y - model output (in vector)
>> LFName - name of loss function
>> leadDim - the leading dimension for the output
>> tBeg - where to start in the gold standard (along the leading dimension)
>> tLen - segment length from oBeg (along the leading dimension)
>> yBeg - where to start in the model output (along the leading dimension)
*/
void _CudaLossBackward(XTensor * dedy, XTensor * t, XTensor * y, 
                      LOSS_FUNCTION_NAME LFName, 
                      int leadDim, int tBeg, int tLen, int yBeg)
{
    CheckNTErrors((tLen <= y->unitNum), "Illegal input length!");
    CheckNTErrors((_IsSameShaped(t, y)&& _IsSameShaped(dedy, y)), 
                  "The input tensors must be of the same size!");
    CheckNTErrors(((dedy->devID == t->devID) && (dedy->devID == y->devID)), 
                  "Tensor must be on the same device!");
    CheckNTErrors((t->order > leadDim), "Illegal leading dimension!");
    CheckNTErrors((t->dataType == DEFAULT_DTYPE && 
                   y->dataType == DEFAULT_DTYPE && 
                   dedy->dataType == DEFAULT_DTYPE),
                  "Input vectors are not in default type.");

    CheckNTErrors((dedy->devID >= 0 && t->devID >= 0 && y->devID >= 0),
                  "The backward compuation must be performed on GPUs.");

    CheckNTErrors((dedy->devID == t->devID && dedy->devID == y->devID),
                  "The vectors must be on the same GPU.");
    CheckNTErrors((tBeg == yBeg), "TODO!");

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
    int size = 1;

    for(int i = leadDim + 1; i < y->order; i++)
        stride *= y->dimSize[i];
    size = tLen * stride;
    blockSize = stride * dimensionSize;
    blockNum = y->unitNum / blockSize;

    int cudaGridSize[3], cudaBlockSize[3];

    GDevs.GetCudaThread(dedy->devID, y->unitNum, cudaGridSize, cudaBlockSize);

    dim3 blocks(cudaGridSize[0]);
    dim3 threads(cudaBlockSize[0]);

    DTYPE * tp = (DTYPE*)t->data;
    DTYPE * yp = (DTYPE*)y->data;
    DTYPE * dedyp = (DTYPE*)dedy->data;

    int devIDBackup;
    ProtectCudaDev(y->devID, devIDBackup);

    /* 
    squared error 
    loss = sum_{i} 0.5*(t_i - y_i)^2, where t_i is the gold standard and y_i is the model output
    dloss/dy_i = y_i - t_i
    */
    if(LFName == SQUAREDERROR){
        if(t->isSparse){
            ShowNTErrors("TODO!");
        }
        else if(size == y->unitNum){
            KernelLossBackwardSquaredError<<<blocks, threads>>>(dedyp, tp, yp, y->unitNum);
        }
        else{
            KernelLossBackwardSquaredErrorBlock<<<blocks, threads>>>(dedyp, tp, yp, blockSize, tBeg * stride, tLen * stride, y->unitNum);
        }
    }

    /* 
    cross entropy
    loss = sum_{i} (-t_i * log(y_i)), where t and y are distributions 
    dloss/dy_i = -t_i / y_i
    */
    else if(LFName == CROSSENTROPY){
        if(t->isSparse){
            ShowNTErrors("TODO!");
        }
        else if(size == y->unitNum){
            KernelLossBackwardCrossEntropy<<<blocks, threads>>>(dedyp, tp, yp, tBeg, tLen, yBeg, blockNum, stride, dimensionSize);
        }
        else{
            KernelLossBackwardCrossEntropyBlock<<<blocks, threads>>>(dedyp, tp, yp, blockSize, tBeg * stride, tLen * stride, y->unitNum);
        }
    }
    else{
        ShowNTErrors("TODO");
    }

    BacktoCudaDev(y->devID, devIDBackup);
}

#endif

} // namespace nts(NiuTrans.Tensor)