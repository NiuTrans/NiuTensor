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
#include "SumByColumnVT.h"
#include "SumByColumnVT.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/*
summation of a vector (column vector) and a tensor
c = a + \sum{col} b_col * \beta
>> a - a vector with the same column size with b
>> b - a tensor
>> c - where we put a+b. we save it in a
>> colNum - column number (of a block)
>> blockSize - size of a block
>> size - size of the entire data array
>> beta - the scaling factor
*/
__global__
void KernelADDByColumnVT(DTYPE * a, DTYPE * b, DTYPE * c, int colNum, int rowNum, int blockNum, DTYPE beta)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row >= rowNum)
        return;

    DTYPE sum = 0;
    for (int k = 0; k < blockNum; k++) {
        DTYPE * bp = b + (rowNum * k + row) * colNum;
        if (colNum % 4 == 0) {
            for (int i = 0; i < colNum; i += 4)
                sum += bp[i] + bp[i + 1] + bp[i + 2] + bp[i + 3];
        }
        else if (colNum % 2 == 0) {
            for (int i = 0; i < colNum; i += 2)
                sum += bp[i] + bp[i + 1];
        }
        else {
            for (int i = 0; i < colNum; i++)
                sum += bp[i];
        }
        __syncthreads();
    }

    c[row] = a[row] + beta * sum;
}

/*
summation of a vector (column vector) and a tensor

for each column b_col, we have
c = a + \sum{col} b_col * \beta
where c and a are vectors, and b_col is a column in b.

>> a - a vector with the same column size with b
>> b - a tensor
>> c - where we put a+b. we save it in a if c is NULL
>> beta - the scaling factor
*/
void _CudaSumByColumnVT(const XTensor * a, const XTensor * b, XTensor * c, DTYPE beta)
{
    CheckNTErrors((a && b && c), "Empty input tensors!");
    CheckNTErrors((XTensor::IsSameShaped(a, c)), "Unmatched tensors in addition!");
    CheckNTErrors((a->order == 2 && a->dimSizeRDI[0] == 1 && b->dimSizeRDI[1] == a->dimSizeRDI[1]),
                  "Illegal input vector size!");
    CheckNTErrors((a->dataType == DEFAULT_DTYPE && b->dataType == DEFAULT_DTYPE &&
                  c->dataType == DEFAULT_DTYPE), "TODO");

    int rowNum = b->dimSize[0];
    int colNum = b->dimSize[1];
    int blockNum = 1;
    for (int i = 2; i < b->order; i++)
        blockNum *= b->dimSizeRDI[i];

    int cudaGridSize[3];
    int cudaBlockSize[3];

    GDevs.GetCudaThread(c->devID, a->dimSizeRDI[1], cudaGridSize, cudaBlockSize);

    int devIDBackup = 0;
    ProtectCudaDev(a->devID, devIDBackup);

    KernelADDByColumnVT << <dim3(cudaGridSize[0]), dim3(cudaBlockSize[0]) >> >
                         ((DTYPE*)a->data, (DTYPE*)b->data, (DTYPE*)c->data, colNum, rowNum, blockNum, beta);

    BacktoCudaDev(a->devID, devIDBackup);
}
#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)