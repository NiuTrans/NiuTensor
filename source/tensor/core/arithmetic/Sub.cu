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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-08-01
 */

#include "../../XDevice.h"
#include "../../XUtility.h"
#include "Sub.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/*
subtraction of data arrays (CUDA Kernel)
c = a - b * \beta
>> a - A matrix
>> b - another matrix
>> c - where we put a-b
>> size - the size of a/b/c
>> beta - the coefficient
*/
__global__
void KernelSUB(DTYPE * a, DTYPE * b, DTYPE * c, int size, DTYPE beta)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size)
        c[i] = a[i] - b[i] * beta;
}

/*
tensor subtraction c = a - b * \beta (cuda version)
>> a - a tensor
>> b - another tensor
>> c - where we put a-b*\beta.
>> beta - the scaling factor
*/
void _CudaSub(const XTensor * a, const XTensor * b, XTensor * c, DTYPE beta)
{
    CheckNTErrors(a && b && c, "Empty tensor input!");
    CheckNTErrors((a->unitNum == b->unitNum && a->unitNum == c->unitNum),
                  "Unmatched tensors in addition!");
    CheckNTErrors((a->dataType == b->dataType && a->dataType == c->dataType),
                  "Unmatched tensors in addition!");
    CheckNTErrors((a->devID == b->devID && a->devID == c->devID),
                  "The tensors must be on the same!");

    int devIDBackup = XDevice::GetGPUDevice();
    XDevice::SetGPUDevice(a->devID);

    if (!a->isSparse && !b->isSparse) {
        CheckNTErrors(!c->isSparse, "Illegal use of sparse matrix in addition!");

        if (a->dataType == DEFAULT_DTYPE &&
            b->dataType == DEFAULT_DTYPE &&
            c->dataType == DEFAULT_DTYPE)
        {
            int gridSize[3], blockSize[3];

            GDevs.GetCudaThread(a->devID, a->unitNum, gridSize, blockSize);
            dim3 blocks(gridSize[0]);
            dim3 threads(blockSize[0]);
            KernelSUB << <blocks, threads >> >((DTYPE*)a->data, (DTYPE*)b->data, (DTYPE*)c->data, a->unitNum, beta);
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

/* subtraction over arrays
tensor subtraction c = a - b * \beta (cuda version) with an input handle
>> devID - device ID (MUST >= 0)
>> handle - cuda handle
>> a - an array
>> b - another array
>> c - where we put a-b
>> size - size of the array
>> beta - the coefficient
*/
void _CudaSubWithHandle(int devID, cublasHandle_t * handle, DTYPE * a, DTYPE * b, DTYPE * c, int size, DTYPE beta)
{
    if (size == 0)
        return;

    if (c == NULL)
        c = a;

    CheckNTErrors((a && b && c), "Empty arrays in addition!");

    int devIDBackup;
    ProtectCudaDev(devID, devIDBackup);

    if (c == a) {
#ifdef DOUBELPRICSION
        cublasDaxpy(*handle, size, &beta, b, 1, a, 1);
#else
        cublasSaxpy(*handle, size, &beta, b, 1, a, 1);
#endif
    }
    else {
        int gridSize[3], blockSize[3];

        GDevs.GetCudaThread(devID, size, gridSize, blockSize);

        dim3 blocks(gridSize[0]);
        dim3 threads(blockSize[0]);

        KernelSUB<<<blocks, threads>>>((DTYPE*)a, (DTYPE*)b, (DTYPE*)c, size, beta);
    }

    BacktoCudaDev(devID, devIDBackup);
}

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)
