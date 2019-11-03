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
 * $Created by: LI Yinqiao (li.yin.qiao.2012@hotmail.com) 2018-7-11
 */

#include "../../XTensor.h"
#include "../../XDevice.h"
#include "ConvertDataType.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

__global__ 
void KernelFloatToFloat16(float * s, __half * t, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size){
        t[i] = __float2half(s[i]);
    }
}

__global__ 
void KernelFloat16ToFloat(__half * s, float * t, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size){
        t[i] = __half2float(s[i]);
    }
}

__global__ 
void KernelFloatToInt(float * inputData, int * outputData, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size){
        outputData[i] = (int)(inputData[i]);
    }
}

__global__ 
void KernelIntToFloat(int * inputData, float * outputData, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size){
        outputData[i] = (float)(inputData[i]);
    }
}

/*
convert data type (cuda code) 
>> input - input tensor
>> output - output tensor
*/
void _CudaConvertDataType(const XTensor * input, XTensor * output)
{
    if (input->dataType == output->dataType)
        return;

    int gridSize[3];
    int blockSize[3];

    GDevs.GetCudaThread(input->devID, input->unitNum, gridSize, blockSize);

    dim3 blocks(gridSize[0]);
    dim3 threads(blockSize[0]);

    int devIDBackup;
    ProtectCudaDev(input->devID, devIDBackup);

    if(input->dataType == X_FLOAT && output->dataType == X_INT)
        KernelFloatToInt<<<blocks, threads>>>
                         ((float*)input->data, (int*)output->data, input->unitNum);
    else if(input->dataType == X_INT && output->dataType == X_FLOAT)
        KernelIntToFloat<<<blocks, threads>>>
                         ((int*)input->data, (float*)output->data, input->unitNum);
    else if(input->dataType == X_FLOAT && output->dataType == X_FLOAT16)
        KernelFloatToFloat16<<<blocks, threads>>>
                             ((float*)input->data, (__half*)output->data, input->unitNum);
    else if(input->dataType == X_FLOAT16 && output->dataType == X_FLOAT)
        KernelFloat16ToFloat<<<blocks, threads>>>
                             ((__half*)input->data, (float*)output->data, input->unitNum);
    else{
        ShowNTErrors("Unsupported data types for conversion!");
    }

    ProtectCudaDev(input->devID, devIDBackup);
}

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)