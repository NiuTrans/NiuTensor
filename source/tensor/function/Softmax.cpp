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
 * $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-04-27
 */

#include <math.h>
#include "Softmax.h"
#include "Softmax.cuh"
#include "../XName.h"
#include "../XUtility.h"
#include "../core/reduce/ReduceSum.h"
#include "../core/reduce/ReduceMax.h"
#include "../core/shape/IsSameShaped.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
softmax y = e^x / \sum_{i} e^{x_i}
>> x - input vector
>> y - result
>> leadDim - leading dimension (along which we perform reduction)
*/
void _Softmax(const XTensor * x, XTensor * y, int leadDim)
{
    if(leadDim < 0)
        leadDim = x->order - 1;

    if(!x->isSparse && !y->isSparse && x->dataType == y->dataType){
        int * dimSize = new int[x->order - 1];
        for(int i = 0; i < x->order; i++){
            if(i < leadDim)
                dimSize[i] = x->dimSize[i];
            else if(i > leadDim)
                dimSize[i - 1] = x->dimSize[i];
        }

        XMem * mem = x->mem;
        XTensor * max = NULL;
        XTensor * sum = NULL;

        max = NewTensorBufV2(x->order - 1, dimSize, x->dataType, x->denseRatio, x->devID, mem);
        sum = NewTensorBufV2(x->order - 1, dimSize, x->dataType, x->denseRatio, x->devID, mem);

        _ReduceMax(x, max, leadDim);
        _ReduceSum(x, sum, leadDim, max, 1.0F, true);

        if(x->devID >= 0){
#ifdef USE_CUDA
            _CudaSoftmaxSumMax(x, y, leadDim, sum, max);
#else
            ShowNTErrors("Please specify USE_CUDA and recompile the code!");
#endif
        }
        else{
            CheckNTErrors((x->dataType == DEFAULT_DTYPE), "TODO!");

            int dimensionSize = y->dimSize[leadDim];
            int stride = 1;
            int blockSize = 1;
            int blockNum = 1;

            for(int i = leadDim + 1; i < y->order; i++)
                stride *= y->dimSize[i];
            blockSize = stride * dimensionSize;
            blockNum = y->unitNum / blockSize;

            for(int k = 0; k < blockNum; k++){
                int m = stride;
                int n = dimensionSize;
                int blockOffset = k * blockSize;
                int blockOffsetMax = k * blockSize / dimensionSize;

                DTYPE * ip = (DTYPE*)x->data + blockOffset;
                DTYPE * op = (DTYPE*)y->data + blockOffset;
                DTYPE * mp = (DTYPE*)max->data + blockOffsetMax;
                DTYPE * sp = (DTYPE*)sum->data + blockOffsetMax;

                for(int j = 0; j < m; j++){
                    DTYPE sumValue = sp[j];
                    if(sumValue == 0){
                        for(int i = 0; i < n; i++)
                            op[i * m + j] = 0;
                    }
                    else{
                        for(int i = 0; i < n; i++){
                            DTYPE r = (DTYPE)exp(ip[i * m + j] - mp[j])/sp[j];
                            if (r > (DTYPE)1.0F)
                                r = (DTYPE)1.0F;
                            else if (r < 0)
                                r = 0;
                            op[i * m + j] = r;
                        }
                    }
                }
            }
        }

        DelTensorBuf(sum);
        DelTensorBuf(max);

        delete[] dimSize;
    }
    else
        ShowNTErrors("TODO!");
    
}

/*
softmax y = e^x / \sum_{i} e^{x_i} (return an XTensor structure) 
make a new tensor to keep the result and return it

>> x - input vector
>> leadDim - leading dimension (along which we perform reduction)
<< return - y
*/
XTensor Softmax(const XTensor &x, int leadDim)
{
    int ld = leadDim;
    if (ld < 0)
        ld = x.order - 1;

    XTensor y(&x);
    y.SetTMPFlag();

    /* call _Softmax function */
    _Softmax(&x, &y, ld);

    /* tensor connection */
    if (x.enableGrad) {
        XLink::MakeLink(&x, NULL, &y, FUNC_SOFTMAX);
        XLink::AddParamToHeadInt(&y, ld);
    }

    return y;
}

void Softmax(const XTensor &x, XTensor &y, int leadDim)
{
    int ld = leadDim;
    if (ld < 0)
        ld = x.order - 1;

    if (!y.isInit || !IsSameShaped(y, x)) {
        InitTensorV2(&y, &x);
    }

    /* call _Softmax function */
    _Softmax(&x, &y, ld);

    if (x.enableGrad) {
        /* tensor connection */
        XLink::MakeLink(&x, NULL, &y, FUNC_SOFTMAX);
        XLink::AddParamToHeadInt(&y, ld);
    }
}

/*
backward computation for dense tensors

dE/dx = dE/dy * dy/dx

    softmax: y_i = e^{x_i} / \sum_{k} e^{x_k}

       dy_i/dx_j = y_i * (\delta(i,j) - y_j)

for cross-entropy error function,

         dE/dy_i = -gold_i / y_i
then
         dE/dx_j = -gold_j + y_j

See more details in LogSoftmaxBackward(...)

>> gold - gold standard to measure error (or loss)
>> y - output of the function
>> x - input of the function
>> dedy - dE/dy
>> dedx - dE/dx
>> lossName - type of loss function, e.g., cross entropy
>> leadDim - leading dimension (along which we perform reduction)
*/
void _SoftmaxBackward(XTensor * gold, XTensor * y, XTensor * x, 
                      XTensor * dedy, XTensor * dedx, 
                      XTensor * padding, int leadDim,
                      LOSS_FUNCTION_NAME lossName)
{
    CheckNTErrors(dedx->isSparse == false, "The gradient tensor must be dense!");
    CheckNTErrors(gold != NULL || lossName == NOLOSS, "Gold standard is required for computing loss!");

    if(leadDim < 0)
        leadDim = y->order - 1;

#ifdef USE_CUDA
    if(y->devID >= 0){
        _CudaSoftmaxBackward(gold, y, x, dedy, dedx, padding, leadDim, lossName);
        return;
    }
#endif

    int dimensionSize = y->dimSize[leadDim];
    int stride = 1;
    int blockSize = 1;
    int blockNum = 1;
    for(int i = leadDim + 1; i < y->order; i++)
        stride *= y->dimSize[i];
    blockSize = stride * dimensionSize;
    blockNum = y->unitNum / blockSize;

    if(x->dataType == DEFAULT_DTYPE && y->dataType == DEFAULT_DTYPE)
    {
        DTYPE * gp = gold != NULL ? (DTYPE*)gold->data : NULL;
        DTYPE * op = (DTYPE*)y->data;
        DTYPE * sp = (DTYPE*)dedx->data;
        DTYPE * yp = NULL;

        if(lossName == CROSSENTROPY){
            if(gold->isSparse){
                CheckNTErrors((gold->order == 2), "TODO!");
                int size = dimensionSize * stride;

                /* dE/dx_j = y_j */
                for(int j = 0; j < size; j++){
                    *(sp+j) = *(op+j);
                }

                /* for j \in gold (sparse), dE/dx_j += -gold_j */
                int num = gold->GetNonzeroSize();
                for(int i = 0; i < num; i++){
                    int key = gold->GetKeyInSparse(i);
                    DTYPE value = gold->GetInSparse(i);
                    int offset = key;
                    if(key >= 0 && key < size)
                        *(sp+offset) += -value;
                }
            }
            else{
                CheckNTErrors((_IsSameShaped(gold, y)), "The tensors must be of the same size!");
                for(int k = 0; k < blockNum; k++){
                    gp = (DTYPE*)gold->data + k * blockSize;
                    op = (DTYPE*)y->data + k * blockSize;
                    sp = (DTYPE*)dedx->data + k * blockSize;
                    int size = stride * dimensionSize;

                    /* dE/dx_j = -gold_j + y_j */
                    for(int j = 0; j < size; j++){
                        *(sp+j) = -(*(gp+j)) + *(op+j);
                    }
                }
            }
        }
        else if(lossName == SQUAREDERROR){
            /* 
            dE/dx_j = -gold_j - y_j
            it is actually the same as that in cross entropy.
            */
            if(gold->isSparse){
                CheckNTErrors((gold->order == 2), "TODO!");
                int size = dimensionSize * stride;

                /* dE/dx_j = y_j */
                for(int j = 0; j < size; j++){
                    *(sp+j) = *(op+j);
                }

                /* for j \in gold (sparse), dE/dx_j += -gold_j */
                int num = gold->GetNonzeroSize();
                for(int i = 0; i < num; i++){
                    int key = gold->GetKeyInSparse(i);
                    DTYPE value = gold->GetInSparse(i);
                    int offset = key;
                    if(key >= 0 && key < size)
                        *(sp+offset) += -value;
                }
            }
            else{
                CheckNTErrors((_IsSameShaped(gold, y)), "The tensors must be of the same size!");
                for(int k = 0; k < blockNum; k++){
                    gp = (DTYPE*)gold->data + k * blockSize;
                    op = (DTYPE*)y->data + k * blockSize;
                    sp = (DTYPE*)dedx->data + k * blockSize;
                    int size = stride * dimensionSize;

                    /* dE/dx_j = -gold_j + y_j */
                    for(int j = 0; j < size; j++){
                        *(sp+j) = -(*(gp+j)) + *(op+j);
                    }
                }
            }
        }
        else if(lossName == NOLOSS){
            /* 
            for softmax: 
            y_i = e^{x_i} / \sum_{k} e^{x_k}
            we have
            dy_i/dx_j = y_i * (\delta(i,j) - y_j)
            Then
            dE/dx_j = \sum_i dE/dy_i * y_i * (\delta(i,j) - y_j) 
                    = dE/dy_j * y_j - y_j * \beta
                    = y_j * (dE/dy_j - \beta)
            where
            \beta = \sum_i (dE/dy_i * y_i) 
            */

            for(int m = 0; m < blockNum; m++){
                yp = (DTYPE*)dedy->data + m * blockSize;
                op = (DTYPE*)y->data + m * blockSize;
                sp = (DTYPE*)dedx->data + m * blockSize;
                
                int nCols = stride;
                for(int k = 0; k < stride; k++){
                    /* \beta = \sum_i (dE/dy_i * y_i) */
                    DTYPE beta = 0;
                    for(int i = 0; i < dimensionSize; i++)
                        beta += yp[i * nCols + k] * op[i * nCols + k];

                    /* dE/ds_j = y_j * (dE/dy_j - \beta) */
                    for(int j = 0; j < dimensionSize; j++)
                        sp[j * nCols + k] = op[j * nCols + k] * (yp[j * nCols + k] - beta);
                }
            }
        }
        else
            ShowNTErrors("TODO!");
    }
    else
        ShowNTErrors("TODO!");
}

} // namespace nts(NiuTrans.Tensor)
