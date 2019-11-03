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
* $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2019-04-24
* I'll attend several conferences and workshops in the following weeks -
* busy days :(
*/

#include "../../XDevice.h"
#include "../../XUtility.h"
#include "Sub.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/*
mask entries of a given tensor (CUDA Kernel)
c = a - b * \beta
>> a - A matrix
>> mask - mask matrix
>> c - where we put masked a
>> size - the size of a/b/c
>> alpha - value
*/
__global__
    void KernelMASK(DTYPE * a, int * mask, DTYPE * c, int size, DTYPE alpha)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size) {
        if (mask[i] == 0) {
            c[i] = alpha;
        }
        else {
            c[i] = a[i];
        }
    }
}

/*
mask entries of a given tensor (cuda version)
>> a - a tensor
>> mask - mask tensor
>> c - where we put masked a
>> alpha - value 
*/
void _CudaMask(const XTensor * a, const XTensor * mask, XTensor * c, DTYPE alpha)
{
    CheckNTErrors(a && mask && c, "Empty tensor input!");
    CheckNTErrors((a->unitNum == mask->unitNum && a->unitNum == c->unitNum),
        "Unmatched tensors in addition!");
    CheckNTErrors(mask->dataType == X_INT, "The mask tensor must be in X_INT!")
    //CheckNTErrors((a->dataType == mask->dataType && a->dataType == c->dataType),
    //    "Unmatched tensors in addition!");
    CheckNTErrors((a->devID == mask->devID && a->devID == c->devID),
        "The tensors must be on the same!");

    int devIDBackup = XDevice::GetGPUDevice();
    XDevice::SetGPUDevice(a->devID);

    if (!a->isSparse && !mask->isSparse) {
        CheckNTErrors(!c->isSparse, "Illegal use of sparse matrix in addition!");

        if (a->dataType == DEFAULT_DTYPE &&
            mask->dataType == X_INT &&
            c->dataType == DEFAULT_DTYPE)
        {
            int gridSize[3], blockSize[3];

            GDevs.GetCudaThread(a->devID, a->unitNum, gridSize, blockSize);
            dim3 blocks(gridSize[0]);
            dim3 threads(blockSize[0]);
            KernelMASK << <blocks, threads >> >((DTYPE*)a->data, (int *)mask->data, (DTYPE*)c->data, a->unitNum, alpha);
        }
        else {
            // TODO!!
            ShowNTErrors("TODO!");
        }
    }
    else {
        // TODO!!
        ShowNTErrors("TODO!");
    }

    XDevice::SetGPUDevice(devIDBackup);
}

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)