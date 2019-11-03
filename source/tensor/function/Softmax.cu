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

#include "Softmax.h"
#include "Softmax.cuh"
#include "Loss.cuh"
#include "../core/reduce/ReduceSum.h"
#include "../core/arithmetic/Multiply.h"
#include "../core/arithmetic/MultiplyDim.h"
#include "../core/shape/Unsqueeze.h"
#include "../core/shape/IsSameShaped.h"
#include "../core/arithmetic/Sum.h"
#include "../XDevice.h"
#include "../XUtility.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/*
softmax y = e^x / \sum_{i} e^{x_i} (Cuda version)
>> x - x vector
>> y - result
>> leadDim - leading dimension (along which we perform reduction)
*/
void _CudaSoftmax(const XTensor * x, XTensor * y, int leadDim)
{
    ShowNTErrors("You should call Softmax instead!");
}

/* 
softmax forward computation (Cuda kernel)

given a data block, 
for each column j, let y_{i,j} and x_{i,j} are the y
and state value for the i-th element of column j. We have

 y_{i,j} = e^{x_{i,j}-max_j} / \sum_{i} e^{x_{i,j}-max_j}

>> x - x tensor
>> max - the max value for each column j
>> sum - \sum_{i} e^{s_{i,j}) for each column j
>> y - y tensor
>> stride - number of items we go over when move to the next step alone the leading dimension
>> strideNum - size of the leading dimension in a block
>> blockSize - size of a block (i.e., stride * strideNum)
>> blockNum - number of blocks
>> strideSizeTotal - stride * blockNum
*/
__global__ 
void KernelSoftmaxComputeTensor(DTYPE * x, DTYPE * max, DTYPE * sum, DTYPE * y, int stride, int strideNum, int blockSize, int blockNum, int strideSizeTotal)
{
    __shared__ DTYPE xSum[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ DTYPE xMax[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ int i2[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    /* we keep the sum and max number in the shared memory for each column */
    if(threadIdx.y == 0){
        xSum[threadIdx.x] = sum[i];
        xMax[threadIdx.x] = max[i];
        i2[threadIdx.x] = i % stride;
    }

    /* synchronize to make sure the values of max and sum are loaded */
    __syncthreads();

    if(i < strideSizeTotal && j < strideNum){
        int offset = int(i / stride) * blockSize + j * stride + i2[threadIdx.x];
        DTYPE r = exp(x[offset] - xMax[threadIdx.x])/xSum[threadIdx.x];
        if (r >(DTYPE)1.0F)
            r = (DTYPE)1.0F;
        else if (r < 0)
            r = 0;
        y[offset] = r;

    }
}

/* 
softmax forward computation (Cuda kernel)
This is for float16 computation

given a data block, 
for each column j, let y_{i,j} and x_{i,j} are the y
and state value for the i-th element of column j. We have

 y_{i,j} = e^{x_{i,j}-max_j} / \sum_{i} e^{x_{i,j}-max_j}

>> x - x tensor
>> max - the max value for each column j
>> sum - \sum_{i} e^{s_{i,j}) for each column j
>> y - y tensor
>> stride - number of items we go over when move to the next step alone the leading dimension
>> strideNum - size of the leading dimension in a block
>> blockSize - size of a block (i.e., stride * strideNum)
>> blockNum - number of blocks
>> strideSizeTotal - stride * blockNum
*/
__global__ 
void KernelSoftmaxComputeTensor(__half * x, __half * max, __half * sum, __half * y, int stride, int strideNum, int blockNum)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ int i2[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ int blockSize;

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    __shared__ __half xSum[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ __half xMax[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    /* we keep the sum and max number in the shared memory for each column */
    if(threadIdx.y == 0){
        xSum[threadIdx.x] = sum[i];
        xMax[threadIdx.x] = max[i];
        i2[threadIdx.x] = i % stride;
        blockSize = stride * strideNum;
    }
#else
    __shared__ DTYPE xSum[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ DTYPE xMax[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    /* we keep the sum and max number in the shared memory for each column */
    if(threadIdx.y == 0){
        xSum[threadIdx.x] = __half2float(sum[i]);
        xMax[threadIdx.x] = __half2float(max[i]);
        i2[threadIdx.x] = i % stride;
        blockSize = stride * strideNum;
    }
#endif

    /* synchronize to make sure the values of max and sum are loaded */
    __syncthreads();

    if(i < stride * blockNum && j < strideNum){
        int offset = int(i / stride) * blockSize + j * stride + i2[threadIdx.x];
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
        y[offset] = __hdiv(hexp(x[offset] - xMax[threadIdx.x]), xSum[threadIdx.x]);
#else
        y[offset] = __float2half(exp(__half2float(x[offset]) - xMax[threadIdx.x])/xSum[threadIdx.x]);
#endif
    }
}

/*
use PTX code to broadcast float data
*/
__device__ __forceinline__ 
float broadcast(float input)
{
    float output;
    asm(
        "{"
        "shfl.sync.idx.b32 %0,%1,0x0,0x1f,0xffffffff;"
        "}"
        :"=f"(output) : "f"(input)
    );
    return output;
}

/*
use warp broadcast to optimize softmax computing
*/
__global__
void KernelSoftmaxComputeTensorUseBroadcast(DTYPE * input, DTYPE * max, DTYPE * sum, DTYPE * output, 
                                            int stride, int strideNum, int blockNum)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    int i2 = j % stride;
    int blockSize = stride * strideNum;

    if (j < stride * blockNum) {
        DTYPE sumData, maxData;
        if (i % 32 == 0) {
            sumData = sum[j];
            maxData = max[j];
        }
        sumData = broadcast(sumData);
        maxData = broadcast(maxData);
        if (i < strideNum){
            int offset = int(j / stride) * blockSize + i * stride + i2;
            DTYPE r = exp(input[offset] - maxData) / sumData;
            if (r > (DTYPE)1.0F)
                r = (DTYPE)1.0F;
            else if (r < 0)
                r = 0;
            output[offset] = r;
        }
    }
}

/*
softmax y = e^x / \sum_{i} e^{x_i} (Cuda version)
>> x - x vector
>> y - result
>> leadDim - leading dimension (along which we perform reduction)
>> sum - \sum_{i} e^{x_i}
>> max - \max_{i} e^{x_i}
*/
void _CudaSoftmaxSumMax(const XTensor * x, XTensor * y, int leadDim, XTensor * sum, XTensor * max)
{
    CheckNTErrors((x->devID >= 0), "Forward computation of softmax must be run on GPUs.");
    CheckNTErrors((x->devID == y->devID), "Tensors used in softmax are not on the same GPU.");
    CheckNTErrors((_IsSameShaped(x, y)), "Input tensors must be of the same size!");

    int dimensionSize = y->dimSize[leadDim];
    int stride = 1;
    int blockSize = 1;
    int blockNum = 1;

    for(int i = leadDim + 1; i < y->order; i++)
        stride *= y->dimSize[i];
    blockSize = stride * dimensionSize;
    blockNum = y->unitNum / blockSize;

    int cudaGridSize[3];
    int cudaBlockSize[3];

    if (leadDim != 0 || dimensionSize <= 10){
        /* allocate thread num for old function */
        GDevs.GetCudaThread2D(x->devID, stride * blockNum, dimensionSize, MAX_INT, cudaGridSize, cudaBlockSize);
    }
    else {
        /* allocate thread num for new function */
        GDevs.GetCudaThread2D(x->devID, dimensionSize, stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);
        if (cudaBlockSize[0] < 32) {
            /* use at least a warp */
            cudaBlockSize[0] = 32;

            if (cudaBlockSize[1] > 32) {
                cudaGridSize[1] = int(ceil(float(stride * blockNum) / 32));
                cudaBlockSize[1] = 32;
            }
        }
    }
    int devIDBackup;
    ProtectCudaDev(x->devID, devIDBackup);

    if(x->dataType == DEFAULT_DTYPE && y->dataType == DEFAULT_DTYPE){
        if (leadDim != 0 || dimensionSize <= 10) {
            KernelSoftmaxComputeTensor <<< dim3(cudaGridSize[0], cudaGridSize[1]), dim3(cudaBlockSize[0], cudaBlockSize[1]) >>>
                                         ((DTYPE*)x->data, (DTYPE*)max->data, (DTYPE*)sum->data, (DTYPE*)y->data,
                                           stride, dimensionSize, stride * dimensionSize, blockNum, stride * blockNum);
        }
        else {
            KernelSoftmaxComputeTensorUseBroadcast <<< dim3(cudaGridSize[0], cudaGridSize[1]), dim3(cudaBlockSize[0], cudaBlockSize[1]) >>>
                                                     ((DTYPE*)x->data, (DTYPE*)max->data, (DTYPE*)sum->data, (DTYPE*)y->data,
                                                       stride, dimensionSize, blockNum);
        }
    }
    else if(x->dataType == X_FLOAT16 && y->dataType == X_FLOAT16){
        KernelSoftmaxComputeTensor <<< dim3(cudaGridSize[0], cudaGridSize[1]), dim3(cudaBlockSize[0], cudaBlockSize[1]) >>>
                                     ((__half*)x->data, (__half*)max->data, (__half*)sum->data, (__half*)y->data, 
                                       stride, dimensionSize, blockNum);
    }
    else{
        ShowNTErrors("TODO!");
    }

    BacktoCudaDev(x->devID, devIDBackup);
}

/*
backward computation for dense matrics with default data type

dE/ds = dE/dy * dy/dx

    softmax: y_i = e^{x_i} / \sum_{k} e^{x_k}

       dy_i/dx_j = y_i * (\delta(i,j) - y_j)

for cross-entropy error function,

         dE/dy_i = -gold_i / y_i
then
         dE/dx_j = -gold_j + y_j

See more details in SoftmaxBackward

>> gold - gold standard to measure error (or loss)
>> y - y of the function
>> x - x of the function
>> dedy - dE/dy
>> dedx - dE/dx
>> lossName - type of loss function, e.g., cross entropy
>> leadDim - leading dimension (along which we perform reduction)
*/
void _CudaSoftmaxBackward(XTensor * gold, XTensor * y, XTensor * x, 
                          XTensor * dedy, XTensor * dedx,
                          XTensor * padding, int leadDim,
                          LOSS_FUNCTION_NAME lossName)
{
    int n = leadDim < 0 ? y->order - 1 : leadDim;

    CheckNTErrors((x->devID >= 0), "Backward computation of log softmax must be run on GPUs.");
    CheckNTErrors((x->devID == y->devID), "Matrices used in log softmax are not on the same GPU.");
    CheckNTErrors((y->order >= 1), "Empty tensor!");

    int devIDBackup;
    ProtectCudaDev(x->devID, devIDBackup);

    if(x->dataType == DEFAULT_DTYPE && y->dataType == DEFAULT_DTYPE){
        
        CheckNTErrors((lossName == CROSSENTROPY || 
                       lossName == SQUAREDERROR || 
                       lossName == ONEHOTERROR || 
                       lossName == NOLOSS),
                       "Unknown loss function.");

        if(lossName == CROSSENTROPY || lossName == SQUAREDERROR){
            _Sum(y, gold, dedx, -1.0F);
            if(padding != NULL) {
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
        else if(lossName == ONEHOTERROR){
            ShowNTErrors("TODO!");
        }
        else if(lossName == NOLOSS){
            /*
            for softmax: 
            y_i = e^{x_i} / \sum_{k} e^{x_k}
            we have
            dy_i/ds_j = y_i * (\delta(i,j) - y_j)
            Then
            dE/dx_j = \sum_i dE/dy_i * dy_i/dx_j
                    = \sum_i dE/dy_i * y_i * (\delta(i,j) - y_j) 
                    = dE/dy_j * y_j - y_j * \beta
                    = y_j * (dE/dy_j - \beta)
            where
            \beta = \sum_i (dE/dy_i * y_i) 
            */

            int * dimSize = new int[y->order];
            for(int i = 0; i < y->order; i++){
                if(i < leadDim)
                    dimSize[i] = y->dimSize[i];
                else if(i > leadDim)
                    dimSize[i - 1] = y->dimSize[i];
            }

            /* make a matrix of the same size as the y (i.e., y) */
            XTensor * ytmp = NewTensor(y);

            /* make a matrix to keep \beta */
            XTensor * beta = NewTensorV2(y->order - 1, dimSize, y->dataType, y->denseRatio, y->devID, y->mem);

            /* \beta = \sum_i (dE/dy_i * y_i) */
            _Multiply(dedy, y, ytmp, 0, 0);
            _ReduceSum(ytmp, beta, leadDim);

            /* ytmp = dE/dy_j - \beta */
            _Unsqueeze(beta, ytmp, leadDim, y->dimSize[leadDim]);
            _Sum(dedy, ytmp, ytmp, -1.0F);

            /* dE/ds_j = y_j * ytmp = y_j * (dE/dy_j - \beta) */
            _Multiply(y, ytmp, dedx, 0, 0);

            delete[] dimSize;
            delete ytmp;
            delete beta;
        }
        else{
            ShowNTErrors("TODO!");
        }
    }
    else
        ShowNTErrors("TODO!");

    BacktoCudaDev(x->devID, devIDBackup);
}

#endif

} // namespace nts(NiuTrans.Tensor)
