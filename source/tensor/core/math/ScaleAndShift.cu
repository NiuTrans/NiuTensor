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

#include "ScaleAndShift.cuh"
#include "../../XDevice.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* 
scale and shift all tensor entires b = a * scale + shift (CUDA Kernel) 
>> a - the input data array
>> b - the output data array
>> size - the size of d
>> scale - how much we want to scale it
>> shift - how much we want to shift it
*/
template<bool isUnitScale, bool isZeroShift>
__global__ 
void KernelScaleAndShift(DTYPE * a, DTYPE * b, int size, DTYPE scale, DTYPE shift)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size){
        if (isUnitScale && !isZeroShift){
            b[i] = a[i] + shift;
        }
        else if (isUnitScale && isZeroShift) {
            b[i] = a[i];
        }
        else if (!isUnitScale && isZeroShift) {
            b[i] = a[i] * scale;
        }
        else {
            b[i] = a[i] * scale + shift;
        }
    }
}

/* 
scale and shift all tensor entires p = p * scale + shift (CUDA Kernel) 
This is for float16 computation
>> a - the input data array
>> b - the output data array
>> size - the size of d
>> scale - how much we want to scale it
>> shift - how much we want to shift it
*/
__global__ 
void KernelScaleAndShift(__half * a, __half * b, int size, __half scale, __half shift)
{

    int i = blockDim.x * blockIdx.x + threadIdx.x;
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    if(i < size)
        b[i] = __hadd(__hmul(a[i], scale), shift);
#else
    if (i < size)
        b[i] = __float2half(__half2float(a[i]) * __half2float(scale) + __half2float(shift));
#endif
}

/* 
scale and shift all tensor entires
p = p * scale + shift
>> a - the input tensor
>> b - the output tensor
>> scale - the scaler factor
>> shift - the shift factor
*/
void _CudaScaleAndShift(const XTensor * a, XTensor * b, DTYPE scale, DTYPE shift)
{
    /* sparse tensor */
    if(a->isSparse){
        ShowNTErrors("TODO!");
    }
    /* dense tensor */
    else{
        int gridSize[3];
        int blockSize[3];

        GDevs.GetCudaThread(a->devID, a->unitNum, gridSize, blockSize);

        dim3 blocks(gridSize[0]);
        dim3 threads(blockSize[0]);

        int devIDBackup;
        ProtectCudaDev(a->devID, devIDBackup);

        if(a->dataType == DEFAULT_DTYPE){
            if(scale == 1.0F && shift == 0)
                KernelScaleAndShift<true, true> <<<blocks, threads>>>((DTYPE*)a->data, (DTYPE*)b->data, a->unitNum, scale, shift);
            else if (scale == 1.0F && shift != 0)
                KernelScaleAndShift<true, false> << <blocks, threads >> >((DTYPE*)a->data, (DTYPE*)b->data, a->unitNum, scale, shift);
            else if(scale != 1.0F && shift == 0)
                KernelScaleAndShift<false, true> << <blocks, threads >> >((DTYPE*)a->data, (DTYPE*)b->data, a->unitNum, scale, shift);
            else
                KernelScaleAndShift<false, false> << <blocks, threads >> >((DTYPE*)a->data, (DTYPE*)b->data, a->unitNum, scale, shift);
        }
        else if(a->dataType == X_FLOAT16){
            unsigned short scale2 = FloatToFloat16(scale);
            unsigned short shift2 = FloatToFloat16(shift);
            __half * scaleft16p = (__half*)&scale2;
            __half * shiftft16p = (__half*)&shift2;
            KernelScaleAndShift<<<blocks, threads>>>((__half*)a->data, (__half*)b->data, a->unitNum, *scaleft16p, *shiftft16p);
        }
        else{
            ShowNTErrors("TODO!");
        }

        BacktoCudaDev(a->devID, devIDBackup);
    }
}

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)