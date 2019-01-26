/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2018, Natural Language Processing Lab, Northestern University.
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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-08-15
 */

#include "DivDim.cuh"
#include "../../XDevice.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* 
tensor division of a tensor and a row vector
c = a / b + alpha * c
where a is a tensor and b is a row vector
>> a - pointer to the data array of a
>> b - pointer to the data array of b
>> c - pointer to the data array of c
>> rowNum - number of rows of a and c
>> colNum - number of columns of a and c (i.e., the size of b)
>> alpha - the scaling factor
*/
template <class T, bool alphaFired>
__global__
void KernelDivWithRow(T * a, T * b, T * c, int rowNum, int colNum, T alpha)
{
    __shared__ T bv[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if(col >= colNum || row >= rowNum)
        return;

    if(threadIdx.y == 0)
        bv[threadIdx.x] = b[col];

    __syncthreads();

    int offset = colNum * row + col;
    if(alphaFired)
        c[offset] = a[offset] / bv[threadIdx.x] + c[offset] * alpha;
    else
        c[offset] = a[offset] / bv[threadIdx.x];
}

/* 
tensor division of a tensor and a colum vector
c = a / b + alpha * c
where a is a tensor and b is a colum vector
>> a - pointer to the data array of a
>> b - pointer to the data array of b
>> c - pointer to the data array of c
>> rowNum - number of rows of a and c (i.e., the size of b)
>> colNum - number of columns of a and c 
>> blockNum - size of a block (matrix), i.e., rowNum * colNum
>> blockNum - number of matrics 
>> alpha - the scaling factor
*/
template <class T, bool alphaFired>
__global__
void KernelDivWithCol(T * a, T * b, T * c, int rowNum, int colNum, int blockSize, int blockNum, T alpha)
{
    __shared__ T bv[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    int colIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    int col = colIndex % colNum;
    int block = colIndex / colNum;

    if(row >= rowNum || block >= blockNum)
        return;

    if(threadIdx.x == 0)
        bv[threadIdx.y] = b[row];

    __syncthreads();

    int offset = block * blockSize + row * colNum + col;
    
    if(alphaFired)
        c[offset] = a[offset] / bv[threadIdx.y] + c[offset] * alpha;
    else
        c[offset] = a[offset] / bv[threadIdx.y];
}

/*
tensor division

c = a / b + \alpha * c
where the size of b is equal to the n-th dimension of a, 
i.e., a is divided with b by broadcasting 

>> a - a tensor
>> b - another tensor whose size is equal to that of dimension n of a
>> c - where we put a / b + \alpha * c. we save it in a if c is NULL
>> n - the dimension index
>> alpha - the scaling factor
*/
void _CudaDivDim(const XTensor * a, const XTensor * b, XTensor * c, int n, DTYPE alpha)
{
    CheckNTErrors(a && b && c, "Empty tensor input!");
    CheckNTErrors(a->unitNum == c->unitNum, "Unmatched tensors in division!");
    CheckNTErrors(a->dataType == b->dataType && a->dataType == c->dataType,
                 "Unmatched data types in division!");
    CheckNTErrors(a->order == c->order, "The input tensors do not have the same order in division!");
    CheckNTErrors(!a->isSparse && !b->isSparse && !c->isSparse, "Dense tensors are required!");
    CheckNTErrors(a->dimSize[n] == b->unitNum, "Wrong tensor size!");

    int stride = 1;
    int blockSize = a->dimSize[n];
    int blockNum = 1;

    for(int i = a->order - 1; i >= 0; i--){
        if(i > n)
            stride *= a->dimSize[i];
        else if(i < n)
            blockNum *= a->dimSize[i];
    }

    int cudaGrids[3];
    int cudaBlocks[3];

    int devIDBackup = 0;
    ProtectCudaDev(a->devID, devIDBackup);

    if (a->dataType == DEFAULT_DTYPE){
        if(stride > 1){
            GDevs.GetCudaThread2D(a->devID, stride * blockNum, blockSize, MAX_INT, cudaGrids, cudaBlocks);
            if(alpha == (DTYPE)0.0F)
                KernelDivWithCol<DTYPE, false> <<<dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1])>>>
                                                ((DTYPE*)a->data, (DTYPE*)b->data, (DTYPE*)c->data, 
                                                  blockSize, stride, blockSize * stride, blockNum, alpha);
            else
                KernelDivWithCol<DTYPE, true>  <<<dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1])>>>
                                                ((DTYPE*)a->data, (DTYPE*)b->data, (DTYPE*)c->data, 
                                                  blockSize, stride, blockSize * stride, blockNum, alpha);
        }
        else if(stride == 1){
            GDevs.GetCudaThread2D(a->devID, blockSize, blockNum, MAX_INT, cudaGrids, cudaBlocks);
            if(alpha == (DTYPE)0.0F)
                KernelDivWithRow<DTYPE, false> <<<dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1])>>>
                                                ((DTYPE*)a->data, (DTYPE*)b->data, (DTYPE*)c->data, 
                                                  blockNum, blockSize, alpha);
            else
                KernelDivWithRow<DTYPE, true>  <<<dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1])>>>
                                                ((DTYPE*)a->data, (DTYPE*)b->data, (DTYPE*)c->data, 
                                                  blockNum, blockSize, alpha);
        }
        else{
            ShowNTErrors("Something is wrong!");
        }
    }
    else {
        ShowNTErrors("TODO!");
    }

    BacktoCudaDev(a->devID, devIDBackup);
}

#endif

} // namespace nts(NiuTrans.Tensor)

