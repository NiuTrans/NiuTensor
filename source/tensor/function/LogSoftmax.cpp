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
* $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-04-26
*/

#include <math.h>
#include "LogSoftmax.h"
#include "LogSoftmax.cuh"
#include "../XName.h"
#include "../XUtility.h"
#include "../core/reduce/ReduceSum.h"
#include "../core/reduce/ReduceMax.h"
#include "../core/movement/CopyValues.h"
#include "../core/shape/IsSameShaped.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
log scale softmax y = log(e^x / \sum_{i} e^{x_i})
>> x - input vector
>> y - result
>> leadDim - leading dimension (along which we perform reduction)
*/
void _LogSoftmax(const XTensor * x, XTensor * y, int leadDim)
{
    CheckNTErrors(!x->isSparse && !y->isSparse, "TODO!");
    CheckNTErrors(x && y, "Empty input tensors!");

    if(leadDim < 0)
        leadDim = x->order - 1;

    if(y->dimSize[leadDim] == 1){
        y->SetZeroAll();
        return;
    }

    if (!x->isSparse && !y->isSparse &&
        x->dataType == DEFAULT_DTYPE && y->dataType == DEFAULT_DTYPE)
    {
        int * dimSize = new int[x->order - 1];
        for (int i = 0; i < x->order; i++) {
            if (i < leadDim)
                dimSize[i] = -x->dimSize[i];
            else if (i > leadDim)
                dimSize[i - 1] = -x->dimSize[i];
        }

        XMem * mem = x->mem;
        XTensor * max = NULL;
        XTensor * sum = NULL;
        XTensor * blockx = NULL;
        XTensor * blocky = NULL;
        XTensor * blockMax = NULL;
        XTensor * blockSum = NULL;

        int dimensionSize = y->dimSize[leadDim];
        int stride = 1;
        int blockSize = 1;
        int blockNum = 1;

        for (int i = leadDim + 1; i < x->order; i++)
            stride *= y->dimSize[i];
        blockSize = stride * dimensionSize;
        blockNum = y->unitNum / blockSize;

        max = NewTensorBufV2(x->order - 1, dimSize, x->dataType, x->denseRatio, x->devID, mem);
        sum = NewTensorBufV2(x->order - 1, dimSize, x->dataType, x->denseRatio, x->devID, mem);

        _ReduceMax(x, max, leadDim);
        _ReduceSum(x, sum, leadDim, max, 1.0F, true);

        if (x->devID >= 0) {
            if(leadDim == x->order - 1){
                blockSize = y->unitNum;
                blockNum  = 1;
                blockx = NewTensor2DV2(blockSize/dimensionSize, -dimensionSize, x->dataType, x->devID, mem);
                blocky = NewTensor2DV2(blockSize/dimensionSize, -dimensionSize, x->dataType, x->devID, mem);
                blockMax = NewTensor2DV2(blockSize/dimensionSize, -1, x->dataType, x->devID, mem);
                blockSum = NewTensor2DV2(blockSize/dimensionSize, -1, x->dataType, x->devID, mem);
            }
            else{
                blockx = NewTensor2DV2(-stride, dimensionSize, x->dataType, x->devID, mem);
                blocky = NewTensor2DV2(-stride, dimensionSize, x->dataType, x->devID, mem);
                blockMax = NewTensor2DV2(-stride, 1, x->dataType, x->devID, mem);
                blockSum = NewTensor2DV2(-stride, 1, x->dataType, x->devID, mem);
            }
        }

        for (int k = 0; k < blockNum; k++) {
            int m = stride;
            int n = dimensionSize;

            DTYPE * ip = (DTYPE*)x->data + k * blockSize;
            DTYPE * op = (DTYPE*)y->data + k * blockSize;
            DTYPE * mp = (DTYPE*)max->data + k * blockSize / dimensionSize;
            DTYPE * sp = (DTYPE*)sum->data + k * blockSize / dimensionSize;

            if (x->devID < 0) {
                for (int j = 0; j < m; j++) {
                    DTYPE sumValue = sp[j];
                    if (sumValue == 0) {
                        for (int i = 0; i < n; i++)
                            op[i * m + j] = 0;
                    }
                    else {
                        for (int i = 0; i < n; i++) {
                            DTYPE r = (DTYPE)log(exp(ip[i * m + j] - mp[j]) / sp[j]);
                            if (IsNAN(r))
                                r = LOGPROB_MIN;
                            if (IsINF(r))
                                r = LOGPROB_MIN;

                            op[i * m + j] = MAX(r, LOGPROB_MIN);
                        }
                    }
                }
            }
            else {
                blockx->data = ip;
                blocky->data = op;
                blockMax->data = mp;
                blockSum->data = sp;
#ifdef USE_CUDA
                if(leadDim == x->order - 1)
                    _CudaLogSoftmaxSumMax(blockx, blocky, 1, blockSum, blockMax);
                else
                    _CudaLogSoftmaxSumMax(blockx, blocky, leadDim, blockSum, blockMax);
#else
                ShowNTErrors("Please specify USE_CUDA and recompile the code!");
#endif
                blockx->data = NULL;
                blocky->data = NULL;
                blockMax->data = NULL;
                blockSum->data = NULL;
            }
        }

        DelTensorBuf(max);
        DelTensorBuf(sum);

        if (x->devID >= 0) {
            delete blockx;
            delete blocky;
            delete blockMax;
            delete blockSum;
        }

        delete[] dimSize;
    }
    else
        ShowNTErrors("TODO!");
}

/*
log scale softmax y = log(e^x / \sum_{i} e^{x_i}) (return an XTensor structure) 
make a new tensor to keep the result and return it

>> x - input vector
>> leadDim - leading dimension (along which we perform reduction)
<< return - y
*/
XTensor LogSoftmax(const XTensor &x, int leadDim)
{
    int ld = leadDim;
    if (ld < 0)
        ld = x.order - 1;

    XTensor y(&x);
    y.SetTMPFlag();

    /* call _LogSoftmax function */
    _LogSoftmax(&x, &y, ld);

    /* tensor connection */
    if (x.enableGrad) {
        XLink::MakeLink(&x, NULL, &y, FUNC_LOGSOFTMAX);
        XLink::AddParamToHeadInt(&y, ld);
    }

    return y;
}

/*
log scale softmax y = log(e^x / \sum_{i} e^{x_i})
make a new tensor to keep the result and return it

>> x - input vector
>> y - output vector
>> leadDim - leading dimension (along which we perform reduction)
*/
void LogSoftmax(const XTensor &x, XTensor &y, int leadDim)
{
    int ld = leadDim;
    if (ld < 0)
        ld = x.order - 1;

    if (!y.isInit || !IsSameShaped(y, x)) {
        InitTensorV2(&y, &x);
    }

    /* call _LogSoftmax function */
    _LogSoftmax(&x, &y, ld);

    if (x.enableGrad) {
        /* tensor connection */
        XLink::MakeLink(&x, NULL, &y, FUNC_LOGSOFTMAX);
        XLink::AddParamToHeadInt(&y, ld);
    }
}

/*
backward computation for dense matrices with default data type

dE/dx = dE/dy * dy/dx

log softmax: y_i = log(e^{x_i} / \sum_{k} e^{x_k})

  dy_i/dx_j 
= d{log(e^{x_i} / \sum_{k} e^{x_k})}/dx_j
= d{log(e^{x_i})}/dx_j - d{log(\sum_{k} e^{x_k})}/dx_j
= \delta(i,j) - e^{x_j}/\sum_{k} e^{x_k})
= \delta(i,j) - exp(y_j)

where \delta(i,j) = 1 if i = j, and \delta(i,j) = 0 otherwise

if loss E is defined as cross entropy, i.e., E = -\sum_{k} (gold_k * y_k), we have

dE/dy_i = -gold_i

(where {gold_k} is the gold standard distribution)

then

dE/dx_j 
= \sum_{i} {dE/dy_i * dy_i/dx_j}
= \sum_{i} {-gold_i * (\delta(i,j) - exp(y_j))}
= \sum_{i} {-gold_i * \delta{i,j)} + \sum_{i} {gold_i * exp(y_j)}
= -gold_i * \delta(i,j) + \sum_{i} {gold_i * exp(y_j)}
= -gold_j + exp(y_j)

Note: gold_i is a distribution, i.e., \sum_{i} gold_i = 1
if gold is with a one-hot representation (gold_i = 1 for only one dimension),
we can reformulize it as dE/dx_j = -\delta(i,j) + exp(y_j)

There are two ways to implement this process.
Method 1. we compute dE/dy and dy/dx resepectively, and then reach dE/dx by dE/dx = dE/dy * dy/dx
(or more precisely dE/dx_j = \sum_{i} {dE/dy_i * dy_i/dx_j})
Method 2. we compute dE/dx (or dE/dx_j) in a single step, rather than resorting to the
sub-models of dE/dy and dy/dx. We can do this by using dE/dx_j = -gold_j + exp(y_j)

Here we choose Method 2, i.e., we straightforwardly compute dE/dx_j by

dE/dx_j = -gold_j + exp(y_j)

(or dE/dx_j = -\delta(i,j) + exp(y_j) for a Maximum A Posteriori Estimation (MAP))

Method 1 is also fine but is more time consuming due to the summation over dimensions.
Note that this method is not good for the standard version softmax when we work with
the cross entropy loss because it is numerical unstable. When we use a usual method to
define softmax, we have softmax: y_i = log(e^{x_i} / \sum_{k} e^{x_k}). It is trivial to
know that dy_i/dx_j = y_i * \delta(i,j) - y_i * y_j. As y_i and y_j could be small numbers,
y_i * y_i would result in a much smaller value with a risk of lossing precision. This is even
worse we multiply dy_i/dx_j with dE/dy_i. So it is in general to use log softmax for
better numerical stability.

>> gold - gold standard to measure error (or loss)
>> y - output of the function
>> x - input of the function
>> dedy - dE/dy
>> dedx - dE/dx
>> lossName - type of loss function, e.g., cross entropy
>> leadDim - leading dimension (along which we perform reduction)
*/
void _LogSoftmaxBackward(XTensor * gold, XTensor * y, XTensor * x,
                         XTensor * dedy, XTensor * dedx, 
                         XTensor * padding, int leadDim, 
                         LOSS_FUNCTION_NAME lossName)
{
    CheckNTErrors((!dedx->isSparse), "The gradient matrix must be dense!");
    CheckNTErrors((gold != NULL), "The gold standard cannot be empty!");

    if(leadDim < 0)
        leadDim = y->order - 1;

#ifdef USE_CUDA
    if (gold->devID >= 0) {
        _CudaLogSoftmaxBackward(gold, y, x, dedy, dedx, padding, leadDim, lossName);
        return;
    }
#endif

    int dimensionSize = y->dimSize[leadDim];
    int stride = 1;
    int blockSize = 1;
    int blockNum = 1;
    for (int i = leadDim + 1; i < y->order; i++)
        stride *= y->dimSize[i];
    blockSize = stride * dimensionSize;
    blockNum = y->unitNum / blockSize;

    if (x->dataType == DEFAULT_DTYPE && y->dataType == DEFAULT_DTYPE)
    {
        DTYPE * gp = (DTYPE*)gold->data;
        DTYPE * op = (DTYPE*)y->data;
        DTYPE * sp = (DTYPE*)dedx->data;

        if (lossName == CROSSENTROPY) {
            if (gold->isSparse) {
                CheckNTErrors((gold->order == 2), "TODO!");
                int gm = gold->dimSize[1];
                int size = dimensionSize * stride;

                /* dE/dx_j = exp(y_j) */
                for (int j = 0; j < size; j++) {
                    *(sp + j) = (DTYPE)exp(*(op + j));
                }

                /* for j \in gold (sparse), dE/dx_j += -gold_j */
                int num = gold->GetNonzeroSize();
                for (int i = 0; i < num; i++) {
                    int key = gold->GetKeyInSparse(i);
                    DTYPE value = gold->GetInSparse(i);
                    int offset = key;
                    if (dedx->dimSize[dedx->order - 1] != gm) {
                        int mi = key % gm;
                        int ni = key / gm;
                        int key2 = ni * dedx->dimSize[dedx->order - 1] + mi;
                        offset = key2;
                    }
                    if (key >= 0 && key < size)
                        *(sp + offset) += -value;
                    else {
                        ShowNTErrors("Something is wrong with the matrix!");
                    }
                }
            }
            else {
                CheckNTErrors((_IsSameShaped(gold, y)), "The tensors must be of the same size!");
                for (int k = 0; k < blockNum; k++) {
                    gp = (DTYPE*)gold->data + k * blockSize;
                    op = (DTYPE*)y->data + k * blockSize;
                    sp = (DTYPE*)dedx->data + k * blockSize;
                    int size = stride * dimensionSize;

                    /* dE/ds_j = -gold_j + exp(y_j) */
                    for (int j = 0; j < size; j++) {
                        *(sp + j) = -(*(gp + j)) + (DTYPE)exp(*(op + j));
                    }
                }
            }
        }
        else if (lossName == SQUAREDERROR) {
            /*
            dE/dx_j = \sum_{i} {dE/dy_i * dy_i/dx_j}
            = \sum_{i} {(exp(y_i) - gold_i) * (\delta(i,j) - exp(y_j))}
            = \sum_{i} {(exp(y_i) - gold_i) * \delta(i,j)}
            - \sum_{i} {(exp(y_i) - gold_i) * exp(y_j)}
            = exp(y_j) - gold_j - exp(y_j) * (\sum_i{exp(y_i)} - \sum_i{gold_i})
            = exp(y_j) - gold_j - exp(y_j) * (1 - 1)
            = exp(y_j) - gold_j
            = gold_j - exp(y_j)
            i.e., minimizing squared error is actually the same as minimizing cross entropy
            when working with (log) softmax!
            */
            if (gold->isSparse) {
                CheckNTErrors((gold->order == 2), "TODO!");
                int gm = gold->dimSize[1];
                int size = dimensionSize * stride;

                /* dE/ds_j = exp(y_j) */
                for (int j = 0; j < size; j++) {
                    *(sp + j) = (DTYPE)exp(*(op + j));
                }

                /* for j \in gold (sparse), dE/ds_j += -gold_j */
                int num = gold->GetNonzeroSize();
                for (int i = 0; i < num; i++) {
                    int key = gold->GetKeyInSparse(i);
                    DTYPE value = gold->GetInSparse(i);
                    int offset = key;
                    if (dedx->dimSize[dedx->order - 1] != gm) {
                        int mi = key % gm;
                        int ni = key / gm;
                        int key2 = ni * dedx->dimSize[dedx->order - 1] + mi;
                        offset = key2;
                    }
                    if (key >= 0 && key < size)
                        *(sp + offset) += -value;
                }
            }
            else {
                CheckNTErrors((_IsSameShaped(gold, y)), "The tensors must be of the same size!");
                for (int k = 0; k < blockNum; k++) {
                    gp = (DTYPE*)gold->data + k * blockSize;
                    op = (DTYPE*)y->data + k * blockSize;
                    sp = (DTYPE*)dedx->data + k * blockSize;
                    int size = stride * dimensionSize;

                    /* dE/dx_j = -gold_j + exp(y_j) */
                    for (int j = 0; j < size; j++) {
                        *(sp + j) = -(*(gp + j)) + (DTYPE)exp(*(op + j));
                    }
                }
            }
        }
        else if (lossName == NOLOSS) {
            ShowNTErrors("TODO!");
        }
        else {
            ShowNTErrors("No loss function is found for (log) softmax!");
        }

        /* for columns with no xs we set dE/ds = 0 */
        if (gold != NULL && gold->isSparse) {
            CheckNTErrors((gold->order == 2), "The gold standard tensor must be of order 2!");
            if ((gold->dimSize[1] > 1 && !gold->isAllValued[0]) || gold->dimSize[1] != dedx->dimSize[dedx->order - 1]) {
                int gn = gold->dimSize[0];
                int gm = gold->dimSize[1];
                int sm = dedx->dimSize[dedx->order - 1];
                int sn = dedx->dimSize[dedx->order - 2];

                int * flags = new int[sm];
                memset(flags, 0, sizeof(int)*sm);
                int num = gold->GetNonzeroSize();
                for (int i = 0; i < num; i++) {
                    int key = gold->GetKeyInSparse(i);
                    int mi = key % gm;
                    flags[mi] = 1;
                }
                for (int mi = 0; mi < sm; mi++) {
                    if (flags[mi] == 0) {
                        if (mi >= gm) {
                            for (int i = 0; i < sn; i++) {
                                int key = i * sm + mi;
                                int offset = key;
                                *(sp + offset) = 0;
                            }
                        }
                        else {
                            for (int i = 0; i < gn; i++) {
                                int key = i * gm + mi;
                                if (key >= 0 && key < dimensionSize) {
                                    int offset = key;
                                    *(sp + offset) = 0;
                                }
                                else {
                                    ShowNTErrors("Illegal key in the index of softmax");
                                }
                            }
                        }
                    }
                }
                delete[] flags;
            }
        }
    }
    else {
        XPRINT(0, stderr, "TODO!");
        exit(1);
    }
}

} // namespace nts(NiuTrans.Tensor)
