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
* $Created by: Lin Ye (email: linye2015@outlook.com) 2018-08-03
*/

#include "../../XDevice.h"
#include "../../XTensor.h"
#include "../shape/IsSameShaped.h"
#include "Clip.h"
#include "Clip.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA
/*
set each entry to its clip value (CUDA Kernel)
>> a - pointer to input data array
>> b - pointer to output data array
>> lower - the lower border
>> upper - the upper border
>> size - size of the data array
*/
__global__
void KernelClip(DTYPE * a, DTYPE * b, DTYPE lower, DTYPE upper, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size) {
        if (a[i] > upper)
            b[i] = upper;
        else if (a[i] < lower)
            b[i] = lower;
        else
            b[i] = a[i];
    }
}

/*
set each entry to its clip value with float16 data type value (CUDA Kernel)
This is for float16 computation
>> a - pointer to input data array
>> b - pointer to output data array
>> lower - the lower border
>> upper - the upper border
>> size - size of the data array
*/
__global__
void KernelClip(__half * a, __half * b, DTYPE lower, DTYPE upper, int size)
{
    return;
}

/*
set each entry to its clip value
>> a - input tensor we are processing
>> b - output tensor we are processing
>> lower - the lower border
>> upper - the upper border
*/
void _CudaClip(const XTensor * a, XTensor * b, DTYPE lower, DTYPE upper)
{
    CheckNTErrors((_IsSameShaped(a, b)), "Input tensors should have the same type!");
    CheckNTErrors((a->isSparse == false), "TODO!");

    int gridSize[3];
    int blockSize[3];

    GDevs.GetCudaThread(a->devID, a->unitNum, gridSize, blockSize);

    dim3 blocks(gridSize[0]);
    dim3 threads(blockSize[0]);

    int devIDBackup;
    ProtectCudaDev(a->devID, devIDBackup);

    if (a->dataType == DEFAULT_DTYPE) {
        KernelClip << <blocks, threads >> >((DTYPE*)a->data, (DTYPE*)b->data, lower, upper, a->unitNum);
    }
    else if (a->dataType == X_FLOAT16) {
        KernelClip << <blocks, threads >> >((__half*)a->data, (__half*)b->data, lower, upper, a->unitNum);
    }
    else {
        ShowNTErrors("TODO!");
    }

    BacktoCudaDev(a->devID, devIDBackup);
}

/*
clip backward computation of dE/dx (Cuda kernel)

dy/dx = 1     if lower <= x <= upper
0     otherwise

>> dedy - dE/dy
>> dedx - dE/dx
>> y - y of the function
>> x - x of the function
>> lower 
>> upper 
*/
__global__
void KernelClipBackward(DTYPE * dedy, DTYPE * dedx, DTYPE * y, DTYPE * x, DTYPE lower, DTYPE upper, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size) {
        DTYPE s = x[i];
        if (s > upper || s < lower)
            dedx[i] = 0;
        else
            dedx[i] = dedy[i];
    }
}

/*
backward computation (Cuda version)

dE/dx = dE/dy * dy/dx

hard tanh: y =  upper    if x > upper
x    if lower <= x <= upper
lower    if x< lower

and dy/dx =  1    if lower <= x <= upper
0    otherwise

>> gold - gold standard to measure error (or loss)
>> y - output of the function
>> x - input of the function
>> dedy - dE/dy
>> dedx - dE/dx
>> lossName - type of loss function, e.g., cross entropy
*/
void _CudaClipBackward(XTensor * y, XTensor * x, XTensor * dedy, XTensor * dedx, DTYPE lower, DTYPE upper)
{
    if (x->dataType == DEFAULT_DTYPE && y->dataType == DEFAULT_DTYPE) {

        int gridSize[3], blockSize[3];

        GDevs.GetCudaThread(x->devID, x->unitNum, gridSize, blockSize);

        int devIDBackup;
        ProtectCudaDev(x->devID, devIDBackup);

        /* dE/dx = dE/dy * dy/dx */
        KernelClipBackward <<<dim3(gridSize[0]), dim3(blockSize[0])>>>
                             ((DTYPE*)dedy->data,
                              (DTYPE*)dedx->data,
                              (DTYPE*)y->data, (DTYPE*)x->data,
                              lower, upper,
                              x->unitNum);

        BacktoCudaDev(x->devID, devIDBackup);
    }
    else
        ShowNTErrors("TODO!");
}


#endif // USE_CUDA
} // namespace nts(NiuTrans.Tensor)
