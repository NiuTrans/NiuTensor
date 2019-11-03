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
* $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-07-29
* &Updated by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-12-26
* Add summation by broadcasting.
*/

#include "SumDim.cuh"
#include "../../XDevice.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* 
tensor summation of a tensor and a row vector
c = a + b * \beta 
where a is a tensor and b is a row vector
>> a - pointer to the data array of a
>> b - pointer to the data array of b
>> c - pointer to the data array of c
>> rowNum - number of rows of a and c
>> colNum - number of columns of a and c (i.e., the size of b)
>> beta - the scaling factor
*/
template <class T, bool betaFired>
__global__
void KernelAddWithRow(T * a, T * b, T * c, int rowNum, int colNum, T beta)
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
    if(betaFired)
        c[offset] = a[offset] + bv[threadIdx.x] * beta;
    else
        c[offset] = a[offset] + bv[threadIdx.x];
}

/* 
tensor summation of a tensor and a colum vector
c = a + b * \beta 
where a is a tensor and b is a colum vector
>> a - pointer to the data array of a
>> b - pointer to the data array of b
>> c - pointer to the data array of c
>> rowNum - number of rows of a and c (i.e., the size of b)
>> colNum - number of columns of a and c 
>> blockNum - size of a block (matrix), i.e., rowNum * colNum
>> blockNum - number of matrics 
>> beta - the scaling factor
*/
template <class T, bool betaFired>
__global__
void KernelAddWithCol(T * a, T * b, T * c, int rowNum, int colNum, int blockSize, int blockNum, T beta)
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
    
    if(betaFired)
        c[offset] = a[offset] + bv[threadIdx.y] * beta;
    else
        c[offset] = a[offset] + bv[threadIdx.y];
}

/*
tensor summation (cuda version)

c = a + b * \beta 
where the size of b is equal to the n-th dimension of a, 
i.e., a is summed with b by broadcasting

>> a - a tensor
>> b - another tensor whose size is equal to that of dimension n of a
>> c - where we put a+b*\beta. we save it in a if c is NULL
>> n - the dimension index
>> beta - the scaling factor
*/
void _CudaSumDim(const XTensor * a, const XTensor * b, XTensor * c, int n, DTYPE beta)
{
    CheckNTErrors(a && b && c, "Empty tensor input!");
    CheckNTErrors(a->unitNum == c->unitNum, "Unmatched tensors in addition!");
    CheckNTErrors(a->dataType == b->dataType && a->dataType == c->dataType,
                  "Unmatched data types in addition!");
    CheckNTErrors(a->order == c->order, "The input tensors do not have the same order in addition!");
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
            if(beta == (DTYPE)1.0F)
                KernelAddWithCol<DTYPE, false> <<<dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1])>>>
                                                ((DTYPE*)a->data, (DTYPE*)b->data, (DTYPE*)c->data, 
                                                  blockSize, stride, blockSize * stride, blockNum, beta);
            else
                KernelAddWithCol<DTYPE, true>  <<<dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1])>>>
                                                ((DTYPE*)a->data, (DTYPE*)b->data, (DTYPE*)c->data, 
                                                  blockSize, stride, blockSize * stride, blockNum, beta);
        }
        else if(stride == 1){
            GDevs.GetCudaThread2D(a->devID, blockSize, blockNum, MAX_INT, cudaGrids, cudaBlocks);
            if(beta == (DTYPE)1.0F)
                KernelAddWithRow<DTYPE, false> <<<dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1])>>>
                                                ((DTYPE*)a->data, (DTYPE*)b->data, (DTYPE*)c->data, 
                                                  blockNum, blockSize, beta);
            else
                KernelAddWithRow<DTYPE, true>  <<<dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1])>>>
                                                ((DTYPE*)a->data, (DTYPE*)b->data, (DTYPE*)c->data, 
                                                  blockNum, blockSize, beta);
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

