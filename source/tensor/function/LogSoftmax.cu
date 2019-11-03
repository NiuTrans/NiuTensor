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

#include "LogSoftmax.h"
#include "LogSoftmax.cuh"
#include "Loss.cuh"
#include "../core/arithmetic/MultiplyDim.h"
#include "../core/reduce/ReduceSum.cuh"
#include "../core/reduce/ReduceMax.cuh"
#include "../core/shape/IsSameShaped.h"
#include "../XDevice.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/*
log scale softmax y = log(e^x / \sum_{i} e^{x_i}) (Cuda version)
>> x - input vector
>> y - result
>> leadDim - leading dimension (along which we perform reduction)
*/
void _CudaLogSoftmax(const XTensor * x, XTensor * y, int leadDim)
{
    ShowNTErrors("You should call LogSoftmax instead!");
}

/* 
log softmax forward computation (Cuda kernel)

for each column j, let y_{i,j} and x_{i,j} are the output
and state value for the i-th element of column j. We have

y_{i,j} = log(e^x_{i,j} / \sum_{i} e^{x_{i,j})

>> x - input tensor (in matrix)
>> max - the max value for each column j
>> sum - \sum_{i} e^{x_{i,j}) for each column j
>> y - output tensor (in matrix)
>> rowNum - row number of the matrix
>> colNum - column number of the matrix
*/
__global__
void KernelLogSoftmaxComputeByRow(DTYPE * x, DTYPE * max, DTYPE * sum, DTYPE * y, int rowNum, int colNum)
{
    __shared__ DTYPE inputSum[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ DTYPE inputMax[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    /* we keep the sum and max number in the shared memory for each column */
    if (threadIdx.y == 0) {
        inputSum[threadIdx.x] = sum[j];
        inputMax[threadIdx.x] = max[j];
    }

    /* synchronize to make sure the values of max and sum are loaded */
    __syncthreads();

    /* y_{i,j} = log(e^(s_{i,j} - max_{j}) / \sum_{k} e^{s_{k,j} - max_{j}}) */
    if (i < rowNum && j < colNum) {
        int key = i * colNum + j;
        DTYPE r = log(exp(x[key] - inputMax[threadIdx.x]) / inputSum[threadIdx.x]);

        if (isnan(r))
            r = LOGPROB_MIN;
        if (isinf(r))
            r = LOGPROB_MIN;

        y[key] = MAX(r, LOGPROB_MIN);
    }
}

/* 
log softmax forward computation (Cuda kernel)

for each row i, let y_{i,j} and x_{i,j} are the output
and state value for the j-th element of row i. We have

y_{i,j} = log(e^x_{i,j} / \sum_{j} e^{x_{i,j})

>> x - input tensor (in matrix)
>> max - the max value for each row i
>> sum - \sum_{j} e^{x_{i,j}) for each row i
>> y - output tensor (in matrix)
>> rowNum - row number of the matrix
>> colNum - column number of the matrix
*/
__global__
void KernelLogSoftmaxComputeByCol(DTYPE * x, DTYPE * max, DTYPE * sum, DTYPE * y, int rowNum, int colNum)
{
    __shared__ DTYPE inputSum[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ DTYPE inputMax[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    /* we keep the sum and max number in the shared memory for each row */
    if (threadIdx.x == 0) {
        inputSum[threadIdx.y] = sum[i];
        inputMax[threadIdx.y] = max[i];
    }

    /* synchronize to make sure the values of max and sum are loaded */
    __syncthreads();

    /* y_{i,j} = log(e^(s_{i,j} - max_{i}) / \sum_{k} e^{s_{i,k} - max_{i}}) */
    if (i < rowNum && j < colNum) {
        int key = i * colNum + j;
        DTYPE r = log(exp(x[key] - inputMax[threadIdx.y]) / inputSum[threadIdx.y]);

        /*if (r < LOGPROB_MIN)
        {
            printf("min %e %e, %e %e, %e %e\n", r, x[key] - inputMax[threadIdx.y], x[key], inputMax[threadIdx.y], exp(x[key] - inputMax[threadIdx.y]), inputSum[threadIdx.y]);
        }*/

        if (isnan(r))
            r = LOGPROB_MIN;
        if (isinf(r))
            r = LOGPROB_MIN;
        
        y[key] = MAX(r, LOGPROB_MIN);
    }
}

/*
log scale softmax y = log(e^x / \sum_{i} e^{x_i}) (Cuda version)
>> x - input vector
>> y - result
>> leadDim - leading dimension (along which we perform reduction)
>> sum - \sum_{i} e^{x_i}
>> max - \max_{i} e^{x_i}
*/
void _CudaLogSoftmaxSumMax(XTensor * x, XTensor * y, int leadDim, XTensor * sum, XTensor * max)
{
    CheckNTErrors((x->devID >= 0), "Forward computation of log softmax must be run on GPUs.");
    CheckNTErrors((x->devID == y->devID), "Input tensors must be on the same GPU.");
    CheckNTErrors((x->order == y->order), "Input tensors must be of the same size.");
    CheckNTErrors((x->order == 2), "Input tensors must be of order 2.");

    int devIDBackup;
    ProtectCudaDev(x->devID, devIDBackup);

    if (x->dataType == DEFAULT_DTYPE && y->dataType == DEFAULT_DTYPE) {
        int gridSize[3], blockSize[3];

        int n = x->dimSize[0];
        int m = x->dimSize[1];

        /* allocate the buffer */
        DTYPE * maxData = (DTYPE*)max->data;
        DTYPE * sumData = (DTYPE*)sum->data;

        if (leadDim == 0) {
            GDevs.GetCudaThread2D(x->devID, n, m, MAX_INT, gridSize, blockSize);

            /* y_{i,j} = log(e^(s_{i,j} - max_{j}) / \sum_{k} e^{s_{k,j} - max_{j}}) */
            KernelLogSoftmaxComputeByRow << <dim3(gridSize[1], gridSize[0]), dim3(blockSize[1], blockSize[0]) >> >
                                            ((DTYPE*)x->data, maxData, sumData, (DTYPE*)y->data, n, m);
        }
        else {
            GDevs.GetCudaThread2D(x->devID, m, n, MAX_INT, gridSize, blockSize);

            /* y_{i,j} = log(e^(s_{i,j} - max_{i}) / \sum_{k} e^{s_{i,k} - max_{i}}) */
            KernelLogSoftmaxComputeByCol << <dim3(gridSize[0], gridSize[1]), dim3(blockSize[0], blockSize[1]) >> >
                                            ((DTYPE*)x->data, maxData, sumData, (DTYPE*)y->data, n, m);
        }
    }
    else {
        ShowNTErrors("TODO!");
    }

    BacktoCudaDev(x->devID, devIDBackup);
}

/*
set dE/dx = exp(y)

>> dedy - dE/dy
>> dedx - dE/dx
>> y - output of the function
>> size - size of output
>> lossName - name of the loss function
*/
__global__
void KernelExpLoss(DTYPE * dedy, DTYPE * dedx, DTYPE * y, int size, LOSS_FUNCTION_NAME lossName)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size) {
        /* dE/dx_j = exp(y_j) */
        if (lossName == CROSSENTROPY)
            dedx[i] = exp(y[i]);
        /* dE/dx_j = exp(y_j) */
        else if (lossName == SQUAREDERROR)
            dedx[i] = exp(y[i]);
        else if (lossName == ONEHOTERROR)
            dedx[i] = 0;
        else
            dedx[i] = 0;
    }
}

/*
backward computation for log softmax

dE/dx = dE/dy * dy/dx

>> dedy - dE/dy
>> dedx - dE/dx
>> gold - gold standard to measure error (or loss)
>> y - output of the function
>> x - input of the function
>> size - size of input/output
>> lossName - name of the loss function
*/
__global__
void KernelLogSoftmaxBackwardDEDS(DTYPE * dedy, DTYPE * dedx, DTYPE * gold, DTYPE * y, DTYPE * x, 
                                  int size, LOSS_FUNCTION_NAME lossName)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size) {
        DTYPE r = 0;
        /* dE/ds_j = exp(y_j) */
        if (lossName == CROSSENTROPY)
            r = -gold[i] + exp(y[i]);
        /* dE/ds_j = exp(y_j) */
        else if (lossName == SQUAREDERROR)
            r = -gold[i] + exp(y[i]);
        else if (lossName == ONEHOTERROR) {
            if (gold[i] == 1.0F)
                r = -gold[i] + exp(y[i]);
            else
                r = 0;
        }
        else {
            r = dedy[i];
        }

        if (isnan(r))
            r = 0;
        if (isinf(r))
            r = 0;

        dedx[i] = r;
    }
}

/*
backward computation for log softmax (sparse matrices) for each column

dE/dx_j += -gold_j

(for dE/dx = dE/dy * dy/dx)

>> dedy - dE/dy
>> dedx - dE/dx
>> gold - gold standard to measure error (or loss)
>> y - output of the function
>> x - input of the function
>> rowNum - row number of the matrix
>> colNum - column number of the matrix
>> gNonZeroNum - 
>> lossName - name of the loss function
*/
__global__
void KernelLogSoftmaxBackwardDEDSSparseByRow(DTYPE * dedy, DTYPE * dedx, void * gold, DTYPE * y, DTYPE * x,
                                             int rowNum, int colNum, int gNonZeroNum, LOSS_FUNCTION_NAME lossName)
{
    int tupleSize = sizeof(int) + sizeof(DTYPE);
    int k = blockDim.x * blockIdx.x + threadIdx.x;

    if (k < gNonZeroNum) {
        /* load the sub-block of the sparse matrix b */
        int key = *(int*)((char*)gold + tupleSize * k);
        int ni = key / colNum;
        int mi = key % colNum;
        int value = *(DTYPE*)((char*)gold + tupleSize * k + sizeof(int));

        if (lossName == CROSSENTROPY)
            dedx[colNum * ni + mi] += -value;
        else if (lossName == SQUAREDERROR)
            dedx[colNum * ni + mi] += -value;
        else if (lossName == ONEHOTERROR) {
            int offset = colNum * ni + mi;
            if (value == 1.0F)
                dedx[offset] += (-value + exp(y[offset]));
            //dedx[offset] += -value * 0.005;
        }
    }
}

/*
backward computation for dense matrics with default data type

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
sub-models dE/dy and dy/dx. We can do this by using dE/dx_j = -gold_j + exp(y_j)

Here we choose Method 2, i.e., we straightforwardly compute dE/dx_j by

dE/dx_j = -gold_j + exp(y_j)

(or dE/dx_j = -\delta(i,j) + exp(y_j) for a Maximum A Posteriori Estimation (MAP))

Method 1 is also fine but is more time consuming due to the summation over dimensions.
Note that this method is not good for the standard version softmax when working with
the cross entropy loss. Because it is numerical unstable. When we use a usual method to
define softmax, we have softmax: y_i = log(e^{x_i} / \sum_{k} e^{x_k}). It is trivial to
know that dy_i/dx_j = y_i * \delta(i,j) - y_i * y_j. As y_i and y_j could be a small number,
y_i * y_i would result in a much smaller one with a risk of lossing precision. This is even
worse we multiply dy_i/dx_j with dE/dy_i. So it is in general to use log softmax instead for
better numerical stability.

>> gold - gold standard to measure error (or loss)
>> y - output of the function
>> x - input of the function
>> dedy - dE/dy
>> deds - dE/dx
>> lossName - type of loss function, e.g., cross entropy
>> leadDim - leading dimension (along which we perform reduction)
*/
void _CudaLogSoftmaxBackward(XTensor * gold, XTensor * y, XTensor * x,
                             XTensor * dedy, XTensor * dedx, 
                             XTensor * padding, int leadDim, 
                             LOSS_FUNCTION_NAME lossName)
{
    leadDim = leadDim < 0 ? y->order - 1 : leadDim;

    CheckNTErrors((x->devID >= 0), "Backward computation of log softmax must be run on GPUs.");
    CheckNTErrors((x->devID == y->devID && gold->devID == y->devID),
                  "Tensors used in log softmax are not on the same GPU.");
    CheckNTErrors((gold != NULL), "No x gold standard is found!");

    int dimensionSize = y->dimSize[leadDim];
    int stride = 1;
    int blockSize = 1;
    int blockNum = 1;
    for (int i = leadDim + 1; i < y->order; i++)
        stride *= y->dimSize[i];
    blockSize = stride * dimensionSize;
    blockNum = y->unitNum / blockSize;

    int devIDBackup;
    ProtectCudaDev(x->devID, devIDBackup);

    if (x->dataType == DEFAULT_DTYPE && y->dataType == DEFAULT_DTYPE) {

        CheckNTErrors((lossName == CROSSENTROPY || lossName == SQUAREDERROR || lossName == NOLOSS),
                      "Unknown loss function.");

        int cudaGridSize[3], cudaBlockSize[3];

        if (lossName == CROSSENTROPY || lossName == SQUAREDERROR) {
            if (gold->isSparse) {
                CheckNTErrors((gold->order == 2), "TODO!")
                CheckNTErrors((leadDim == 0), "TODO!");
                GDevs.GetCudaThread(x->devID, x->unitNum, cudaGridSize, cudaBlockSize);

                /* dE/ds_j = exp(y_j) */
                KernelExpLoss <<<dim3(cudaGridSize[0]), dim3(cudaBlockSize[0]) >>>
                                 (NULL,
                                 (DTYPE*)dedx->data,
                                 (DTYPE*)y->data,
                                 dimensionSize * stride,
                                 lossName);

                GDevs.GetCudaThread(x->devID, gold->unitNumNonZero, cudaGridSize, cudaBlockSize);

                /* dE/ds_j += -gold_j */
                KernelLogSoftmaxBackwardDEDSSparseByRow <<<dim3(cudaGridSize[0]), dim3(cudaBlockSize[0]) >>>
                                                           (NULL,
                                                           (DTYPE*)dedx->data,
                                                           (char*)gold->data + sizeof(int),
                                                           (DTYPE*)y->data,
                                                           (DTYPE*)x->data,
                                                           dedx->dimSize[0], dedx->dimSize[1], gold->unitNumNonZero, lossName);
            }
            else {
                CheckNTErrors((_IsSameShaped(gold, y)), "The tensors must be of the same size!");

                for (int k = 0; k < blockNum; k++) {
                    GDevs.GetCudaThread(x->devID, blockSize, cudaGridSize, cudaBlockSize);

                    /* dE/ds_j = -gold_j + exp(y_j) */
                    KernelLogSoftmaxBackwardDEDS <<<dim3(cudaGridSize[0]), dim3(cudaBlockSize[0]) >>>
                                                    (NULL,
                                                    (DTYPE*)dedx->data + k * blockSize,
                                                    (DTYPE*)gold->data + k * blockSize,
                                                    (DTYPE*)y->data + k * blockSize,
                                                    (DTYPE*)x->data + k * blockSize,
                                                    dimensionSize * stride, lossName);
                }
            }
            if(padding != NULL) {
                int n = leadDim;

                int paddingOrder = padding->order;
                int * paddingDims = new int[paddingOrder];
                memcpy(paddingDims, padding->dimSize, padding->order * sizeof(int));
                padding->Reshape(padding->unitNum);

                int order = dedx->order;
                int * dims = new int[order];
                memcpy(dims, dedx->dimSize, dedx->order * sizeof(int));
                dedx->Reshape(dedx->unitNum/dedx->GetDim(n), dedx->GetDim(n));
                _MultiplyDimMe(dedx, padding, 0);

                padding->Reshape(paddingOrder, paddingDims);
                dedx->Reshape(order, dims);

                delete[] paddingDims;
                delete[] dims;
            }
        }
        else {
            ShowNTErrors("TODO!");
        }
    }
    else{
        ShowNTErrors("TODO!");
    }

    BacktoCudaDev(x->devID, devIDBackup);
}

#endif

} // namespace nts(NiuTrans.Tensor)