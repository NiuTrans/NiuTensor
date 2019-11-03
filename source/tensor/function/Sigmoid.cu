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
 * $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-04-25
 */

#include "Sigmoid.h"
#include "Sigmoid.cuh"
#include "Loss.cuh"
#include "../loss/CrossEntropy.cuh"
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
sigmoid function y = 1/(1+exp(-x))  (Cuda kernel) 
>> x - input data pointer
>> y - output data pointer
>> size - size of input/output
*/
__global__ 
void KernelSigmoidCompute(DTYPE * x, DTYPE * y, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size){
        y[i] = 1/(1+exp(-x[i]));
    }
}

/*
sigmoid function y = 1/(1+exp(-x)) (Cuda version)
>> x - input vector
>> y - result
*/
void _CudaSigmoid(const XTensor * x, XTensor * y)
{
    CheckNTErrors(!x->isSparse && !y->isSparse, "the activation function (rectify) does not support sparse matrices.");
    CheckNTErrors(x->unitNum && y->unitNum, "we require two vectors with the same length.");

    int gridSize[3], blockSize[3];

    GDevs.GetCudaThread(x->devID, x->unitNum, gridSize, blockSize);

    int devIDBackup;
    ProtectCudaDev(x->devID, devIDBackup);

    KernelSigmoidCompute<<<dim3(gridSize[0]), dim3(blockSize[0])>>>((DTYPE*)x->data, (DTYPE*)y->data, x->unitNum);

    BacktoCudaDev(x->devID, devIDBackup);
}

/* 
sigmoid backward computation of dE/dx (Cuda kernel)

dE/ds = dE/dy * dy/dx

sigmoid: y = 1/(1+exp(-x))

   and dy/ds = y * (1 -y)

>> dedy - dE/dy
>> dedx - dE/ds
>> y - output of the function
>> x - input of the function
>> size - size of output/input
*/
__global__ 
void KernelSigmoidBackward(DTYPE * dedy, DTYPE * dedx, DTYPE * y, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size){
        dedx[i] = dedy[i] * y[i] * ((DTYPE)1.0 - y[i]);
    }
}

/*
backward computation (Cuda version)

dE/ds = dE/dy * dy/dx

sigmoid: y = 1/(1+exp(-x))

   and dy/dx = y * (1 -y)

>> y - output of the function
>> x - input of the function
>> dedy - dE/dy
>> dedx - dE/dx
*/
void _CudaSigmoidBackward(XTensor * y, XTensor * x, 
                          XTensor * dedy, XTensor * dedx)
{
    int gridSize[3], blockSize[3];

    GDevs.GetCudaThread(y->devID, y->unitNum, gridSize, blockSize);

    int devIDBackup;
    ProtectCudaDev(y->devID, devIDBackup);

    /* dE/dx = dE/dy * dy/dx */
    KernelSigmoidBackward<<<dim3(gridSize[0]),dim3(blockSize[0])>>>
                            ((DTYPE*)dedy->data,
                            (DTYPE*)dedx->data,
                            (DTYPE*)y->data,
                            y->unitNum);

    BacktoCudaDev(x->devID, devIDBackup);
}

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)