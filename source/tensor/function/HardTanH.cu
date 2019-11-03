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

#include "HardTanH.h"
#include "HardTanH.cuh"
#include "../XDevice.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* 
hard tanh forward computation (Cuda kernel) 
y =  1    if x > 1
     x    if -1 <= x <= 1
    -1    if x < -1
>> x - input data array
>> y - output data array
>> size - size of input/output
*/
__global__ 
void KernelHardtanhCompute(DTYPE * x, DTYPE * y, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size){
        DTYPE p = x[i];
        if(p > (DTYPE)1.0)
            p = (DTYPE)1.0;
        else if(p < (DTYPE)-1.0)
            p = (DTYPE)-1.0;
        y[i] = p;
    }
}

/*
hard tanh forward computation (Cuda version) 
y =  1    if x > 1
     x    if -1 <= x <= 1
    -1    if x < -1
>> x - input tensor
>> y - output tensor
*/
void _CudaHardTanH(const XTensor * x, XTensor * y)
{
    CheckNTErrors(!x->isSparse && !y->isSparse, 
                  "The hard tanh activation function does not support sparse tensors.");

    int gridSize[3], blockSize[3];

    GDevs.GetCudaThread(x->devID, x->unitNum, gridSize, blockSize);

    int devIDBackup;
    ProtectCudaDev(x->devID, devIDBackup);

    KernelHardtanhCompute<<<dim3(gridSize[0]), dim3(blockSize[0])>>>((DTYPE*)x->data, (DTYPE*)y->data, x->unitNum);

    BacktoCudaDev(x->devID, devIDBackup);
}

/* 
hard tanh backward computation of dE/dx (Cuda kernel)

dy/dx = 1     if -1 <= x <= 1
        0     otherwise

>> dedy - dE/dy
>> dedx - dE/dx
>> y - y of the function
>> x - x of the function
>> size - size of y/x
*/
__global__ 
void KernelHardtanhBackward(DTYPE * dedy, DTYPE * dedx, DTYPE * x, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size){
        DTYPE s = x[i];
        if(s > (DTYPE)1.0 || s < (DTYPE)-1.0)
            dedx[i] = 0;
        else
            dedx[i] = dedy[i];
    }
}

/*
backward computation (Cuda version)

dE/dx = dE/dy * dy/dx

hard tanh: y =  1    if x > 1
                x    if -1 <= x <= 1
               -1    if x< -1

   and dy/dx =  1    if -1 <= x <= 1
                0    otherwise

>> y - output of the hardtanh function
>> x - input of the hardtanh function
>> dedy - dE/dy
>> dedx - dE/dx
*/
void _CudaHardTanHBackward(XTensor * y, XTensor * x, 
                           XTensor * dedy, XTensor * dedx)
{
    int gridSize[3], blockSize[3];

    GDevs.GetCudaThread(x->devID, x->unitNum, gridSize, blockSize);

    int devIDBackup;
    ProtectCudaDev(x->devID, devIDBackup);

    /* dE/dx = dE/dy * dy/dx */
    KernelHardtanhBackward<<<dim3(gridSize[0]),dim3(blockSize[0])>>>
                            ((DTYPE*)dedy->data, 
                            (DTYPE*)dedx->data,
                            (DTYPE*)x->data, 
                             x->unitNum);

    BacktoCudaDev(x->devID, devIDBackup);
}

#endif

} // namespace nts(NiuTrans.Tensor)