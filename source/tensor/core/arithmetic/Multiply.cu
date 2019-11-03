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
#include "Multiply.h"
#include "Multiply.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA
/*
multiplication of data arrays in a element-wise manner c(i) = a(i)*b(i)
>> a - data array a
>> b - data array b
>> c - result data array
>> size - size of c
*/
__global__
void KernelMulElementWise(DTYPE * a, DTYPE * b, DTYPE * c, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size)
        c[i] = a[i] * b[i];
}

/*
multiplication of data arrays in a element-wise manner c(i) = a(i)*b(i) + \alpha*c(i)
>> a - data array a
>> b - data array b
>> c - result data array
>> size - size of c
>> alpha - the coefficient
*/
__global__
void KernelMulElementWiseV2(DTYPE * a, DTYPE * b, DTYPE * c, int size, DTYPE alpha)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size)
        c[i] = a[i] * b[i] + alpha * c[i];
}

/*
multiplication of two tensors in a element-wise manner c(i) = a(i)*b(i).
Note that a and b can be of different sizes here, i.e.,
|a_lead| <= |c_lead| and |b_lead| <= |c_lead|
where |a_lead| means the size of the leading dimension of a
>> a - tensor a
>> b - tensor b
>> c - result tensor
>> alpha - the coefficient
>> stride - the number of items we go over when move next along the leading dimension in a block
>> ldSizeA - size of the leading dimension of a
>> ldSizeB - size of the leading dimension of b
>> ldSizeC - size of the leading dimension of c
>> blockNum - number of blocks
*/
template<int nonZeroAlpha> __global__
void KernelMulElementWiseTensorDynamic(DTYPE * a, DTYPE * b, DTYPE * c, DTYPE alpha,
                                       int stride, int ldSizeA, int ldSizeB, int ldSizeC, int blockNum)
{
    __shared__ DTYPE* ap[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ DTYPE* bp[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ DTYPE* cp[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= blockNum * stride || j >= ldSizeC)
        return;

    if (threadIdx.y == 0) {
        int block = i / stride;
        int size = block * stride;
        ap[threadIdx.x] = a + size * ldSizeA;
        bp[threadIdx.x] = b + size * ldSizeB;
        cp[threadIdx.x] = c + size * ldSizeC;
    }

    __syncthreads();

    int aj = j >= ldSizeA ? j % ldSizeA : j;
    int bj = j >= ldSizeB ? j % ldSizeB : j;
    int offseti = i % stride;

    if (nonZeroAlpha == 0)
        cp[threadIdx.x][j * ldSizeC + offseti] = ap[threadIdx.x][aj * ldSizeA + offseti] * bp[threadIdx.x][bj * ldSizeB + offseti];
    else
        cp[threadIdx.x][j * ldSizeC + offseti] = ap[threadIdx.x][aj * ldSizeA + offseti] * bp[threadIdx.x][bj * ldSizeB + offseti] +
        alpha * cp[threadIdx.x][j * ldSizeC + offseti];
}

/*
element-wise product of two tensors
c(i) = a(i)*b(i) + \alpha * c(i)
where i is the item index
>> a - tensor a
>> b - tensor b
>> c - result tensor
>> alpha - the coefficient
>> leadingDim - dimension along which we perform broadcasting
*/
void _CudaMultiply(const XTensor * a, const XTensor * b, XTensor * c, DTYPE alpha, int leadingDim)
{
    CheckNTErrors((a->unitNum <= c->unitNum && b->unitNum <= c->unitNum),
                  "Unmatched tensors in multiplication!");
    CheckNTErrors((a->order == b->order && a->order == c->order), "Unmatched tensors!");

    int stride = 1;
    int blockSizeA = 1;
    int blockNum = 1;
    int dimensionSizeA = a->dimSize[leadingDim];
    int dimensionSizeB = b->dimSize[leadingDim];
    int dimensionSizeC = c->dimSize[leadingDim];

    for (int i = 0; i < a->order; i++) {
        if (i != leadingDim) {
            CheckNTErrors((a->dimSize[i] == b->dimSize[i] &&
                           a->dimSize[i] == c->dimSize[i]),
                          "Unmatched tensors!");
        }
        if (i > leadingDim)
            stride *= a->dimSize[i];
    }

    blockSizeA = stride * dimensionSizeA;
    blockNum = a->unitNum / blockSizeA;

    int devIDBackup;
    ProtectCudaDev(a->devID, devIDBackup);

    if (!a->isSparse && !b->isSparse) {
        if (a->dataType == DEFAULT_DTYPE && b->dataType == DEFAULT_DTYPE) {
            int cudaGridSize[3];
            int cudaBlockSize[3];

            if (a->unitNum == c->unitNum && b->unitNum == c->unitNum) {
                GDevs.GetCudaThread(a->devID, c->unitNum, cudaGridSize, cudaBlockSize);
                dim3 blocks(cudaGridSize[0]), threads(cudaBlockSize[0]);

                if (alpha == 0)
                    KernelMulElementWise << <blocks, threads >> >((DTYPE*)a->data, (DTYPE*)b->data, (DTYPE*)c->data, c->unitNum);
                else
                    KernelMulElementWiseV2 << <blocks, threads >> >((DTYPE*)a->data, (DTYPE*)b->data, (DTYPE*)c->data, c->unitNum, alpha);
            }
            else {
                GDevs.GetCudaThread2D(c->devID, stride * blockNum, dimensionSizeC, MAX_INT, cudaGridSize, cudaBlockSize);
                dim3 blocks(cudaGridSize[0], cudaGridSize[1]), threads(cudaBlockSize[0], cudaBlockSize[1]);

                if (alpha == 0) {
                    KernelMulElementWiseTensorDynamic<0> << <blocks, threads >> >
                        ((DTYPE*)a->data, (DTYPE*)b->data, (DTYPE*)c->data, 0,
                          stride, dimensionSizeA, dimensionSizeB, dimensionSizeC, blockNum);
                }
                else {
                    KernelMulElementWiseTensorDynamic<1> << <blocks, threads >> >
                        ((DTYPE*)a->data, (DTYPE*)b->data, (DTYPE*)c->data, alpha,
                          stride, dimensionSizeA, dimensionSizeB, dimensionSizeC, blockNum);
                }
            }
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

    BacktoCudaDev(a->devID, devIDBackup);
}

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)