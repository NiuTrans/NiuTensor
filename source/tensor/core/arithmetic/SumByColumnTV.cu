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

#include "../../XDevice.h"
#include "../../XTensor.h"
#include "SumByColumnTV.h"
#include "SumByColumnTV.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/*
summation of a tensor and a vector (column vector)
c_col = a_col  + b * \beta
>> a - a tensor
>> b - a vector with the same column size with a
>> c - where we put a+b. we save it in a
>> colNum - column number (of a block)
>> blockSize - size of a block
>> size - size of the entire data array
>> beta - the scaling factor
*/
__global__
void KernelADDByColumnTV(DTYPE * a, DTYPE * b, DTYPE * c, int colNum, int blockSize, int size, DTYPE beta)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= size)
        return;

    int offset = i % blockSize;
    int row = offset / colNum;

    c[i] = a[i] + b[row] * beta;
}

/*
summation of a tensor and a vector (column vector)
for each column a_col (in a block), we have
c_col = a_col + b * \beta
where b is a vector.

>> a - a tensor
>> b - a vector with the same column size with a
>> c - where we put a+b. we save it in a if c is NULL
>> beta - the scaling factor
*/
void _CudaSumByColumnTV(const XTensor * a, const XTensor * b, XTensor * c, DTYPE beta)
{
    CheckNTErrors((a && b && c), "Empty input tensors!");
    CheckNTErrors((XTensor::IsSameShaped(a, c)), "Unmatched tensors in addition!");
    CheckNTErrors((b->order == 2 && b->dimSizeRDI[0] == 1 && b->dimSizeRDI[1] == a->dimSizeRDI[1]),
                  "Illegal input vector size!");
    CheckNTErrors((a->dataType == DEFAULT_DTYPE && b->dataType == DEFAULT_DTYPE &&
                  c->dataType == DEFAULT_DTYPE), "TODO");

    int rowNum = a->dimSize[0];
    int colNum = a->dimSize[1];
    int blockNum = 1;
    for (int i = 2; i < a->order; i++)
        blockNum *= a->dimSizeRDI[i];

    int cudaGridSize[3];
    int cudaBlockSize[3];

    GDevs.GetCudaThread(c->devID, a->unitNum, cudaGridSize, cudaBlockSize);

    int devIDBackup;
    ProtectCudaDev(a->devID, devIDBackup);

    KernelADDByColumnTV << <dim3(cudaGridSize[0]), dim3(cudaBlockSize[0]) >> >
                          ((DTYPE*)a->data, (DTYPE*)b->data, (DTYPE*)c->data, colNum, rowNum * colNum, a->unitNum, beta);

    BacktoCudaDev(a->devID, devIDBackup);
}

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)