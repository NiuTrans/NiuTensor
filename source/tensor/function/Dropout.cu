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

#include "Dropout.h"
#include "Dropout.cuh"
#include "Loss.cuh"
#include "../XDevice.h"

#ifdef USE_CUDA

// the CUDA stuff
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>

#endif

namespace nts{ // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* 
dropout function (Cuda kernel) 
>> x - input data pointer
>> y - output data pointer
>> m - mask indicator to set zero 
>> s - the scale factor
>> size - size of input/output
*/
__global__ 
void KernelDropoutCompute(DTYPE * x, DTYPE * y, DTYPE * m, DTYPE s, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size){
        y[i] = x[i] * m[i] * s;
    }
}

/*
dropout function (Cuda version)
>> x - input tensor
>> y - output tensor
>> mask - mask tensor to set 0
>> scaleFactor - the scale factor
*/
void _CudaDropout(const XTensor * x, XTensor * y, const XTensor * mask, DTYPE scaleFactor)
{
    if(x->dataType == DEFAULT_DTYPE && y->dataType == DEFAULT_DTYPE){

        CheckNTErrors(!x->isSparse && !y->isSparse, "the activation function (rectify) does not support sparse matrices.");
        CheckNTErrors(x->unitNum && y->unitNum, "we require two vectors with the same length.");

        int gridSize[3], blockSize[3];

        GDevs.GetCudaThread(x->devID, x->unitNum, gridSize, blockSize);

        int devIDBackup;
        ProtectCudaDev(x->devID, devIDBackup);

        KernelDropoutCompute<<<dim3(gridSize[0]), dim3(blockSize[0])>>>((DTYPE*)x->data, (DTYPE*)y->data, (DTYPE*)mask->data, scaleFactor, x->unitNum);

        BacktoCudaDev(x->devID, devIDBackup);
    }
    else
        ShowNTErrors("TODO!");
}

/* 
backward computation of dropout function (Cuda kernel)

dE/dx = dE/dy * dy/dx

>> dedy - dE/dy
>> dedx - dE/dx
>> m - mask indicator to set zero 
>> s - the scale factor
>> size - size of input/output
*/
__global__
void KernelDropoutBackward(DTYPE * dedy, DTYPE * dedx, 
                           DTYPE * m, DTYPE s, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size){
        dedx[i] = dedy[i] * m[i] * s;
    }
}

/* 
backward computation of dropout function (Cuda version)

dE/dx = dE/dy * dy/dx

>> y - output of the dropout function
>> x - input of the dropout function
>> dedy - dE/dy
>> dedx - dE/dx
>> mask - mask tensor to set 0
>> scaleFactor - the scale factor
*/
void _CudaDropoutBackward(const XTensor * y, const XTensor * x,
                          const XTensor * dedy, XTensor * dedx,
                          const XTensor * mask, DTYPE scaleFactor)
{
    int gridSize[3], blockSize[3];

    if(x->dataType == DEFAULT_DTYPE && y->dataType == DEFAULT_DTYPE){
        GDevs.GetCudaThread(x->devID, x->unitNum, gridSize, blockSize);

        int devIDBackup;
        ProtectCudaDev(x->devID, devIDBackup);

        /* dE/ds = dE/dy * dy/ds */
        KernelDropoutBackward<<<dim3(gridSize[0]),dim3(blockSize[0])>>>
                              ((DTYPE*)dedy->data, (DTYPE*)dedx->data, 
                               (DTYPE*)mask->data, scaleFactor, x->unitNum);

        BacktoCudaDev(x->devID, devIDBackup);
    }
    else
        ShowNTErrors("TODO!");
}

#endif

} // namespace nts(NiuTrans.Tensor)