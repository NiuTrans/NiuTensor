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

#include "Rectify.cuh"
#include "../XDevice.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* 
hard rectify computation (Cuda kernel) 
rectify   : y =  x    if x >= 0
                 0    if x < 0
>> input - input tensor
>> output - output tensor
>> size - size of input/output
*/
__global__ 
void KernelRectify(DTYPE * x, DTYPE * y, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size){
        DTYPE p = x[i];
        if(p < 0)
            p = 0;
        y[i] = p;
    }
}

/*
rectify function y = max(0, x)
>> x - input tensor
>> y - result
*/
void _CudaRectify(const XTensor * x, XTensor * y)
{
    int gridSize[3], blockSize[3];

    GDevs.GetCudaThread(x->devID, x->unitNum, gridSize, blockSize);

    int devIDBackup;
    ProtectCudaDev(x->devID, devIDBackup);

    KernelRectify<<<dim3(gridSize[0]), dim3(blockSize[0])>>>
                  ((DTYPE*)x->data, (DTYPE*)y->data, x->unitNum);

    BacktoCudaDev(x->devID, devIDBackup);
}

/* 
rectify backward computation of dE/dx (Cuda kernel)

dy/dx =  1    if x >= 0
         0    otherwise

>> dedy - dE/dy
>> dedx - dE/dx
>> x - input of the function
>> size - size of output/input
*/
__global__ 
void KernelRectifyBackward(DTYPE * dedy, DTYPE * dedx, DTYPE * x, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size){
        DTYPE s = x[i];
        if(s >= 0)
            dedx[i] = dedy[i];
        else
            dedx[i] = 0;
    }
}

/*
backward computation (Cuda version)

dE/dx = dE/dy * dy/dx

rectify  : y =  s    if s >= 0
                0    if s < 0

   and dy/ds =  1    if s >= 0
                0    otherwise

>> y - output of the rectify function
>> x - input of the rectify function
>> dedy - dE/dy
>> dedx - dE/dx
*/
void _CudaRectifyBackward(XTensor * y, XTensor * x, 
                          XTensor * dedy, XTensor * dedx)
{
    int gridSize[3], blockSize[3];

    GDevs.GetCudaThread(x->devID, x->unitNum, gridSize, blockSize);

    int devIDBackup;
    ProtectCudaDev(x->devID, devIDBackup);

    /* dE/ds = dE/dy * dy/ds */
    KernelRectifyBackward<<<dim3(gridSize[0]),dim3(blockSize[0])>>>
                          ((DTYPE*)dedy->data, 
                           (DTYPE*)dedx->data,
                           (DTYPE*)x->data, 
                            x->unitNum);

    BacktoCudaDev(x->devID, devIDBackup);
}

#endif

} // namespace nts(NiuTrans.Tensor)