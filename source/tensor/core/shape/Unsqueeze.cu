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
#include "Unsqueeze.h"
#include "Unsqueeze.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef  USE_CUDA

/*
insert a dimension by copying the blocks for n times (where n is the size of the inerted dimension)
>> s - pointer to the source data array
>> blockSize - size of a block

>> totalSize - total size of the blocks (i.e., blockSIze * n)
>> t - pointer to the target data array
>> n - number of blocks to copy data
*/
template<class T>
__global__
void KernelUnsqueezeFlat(void * s, int blockSize, int totalSize, void * t, int n)
{
    /* index of data items */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= blockSize)
        return;

    T value = ((T*)s)[i];
    T * tData = (T*)t;

    __syncthreads();

    for (int k = i; k < totalSize; k += blockSize)
        tData[k] = value;
}

/*
insert a dimension by copying the blocks for n times (where n is the size of the inerted dimension)
>> s - pointer to the source data array
>> blockSize - size of a block

>> totalSize - total size of the blocks (i.e., blockSIze * n)
>> t - pointer to the target data array
>> n - number of blocks to copy data
*/
template<class T>
__global__
void KernelUnsqueezeFlatBigram(void * s, int blockSize, int totalSize, void * t, int n)
{
    /* index of data items */
    int i = (blockDim.x * blockIdx.x + threadIdx.x) * 2;

    if (i >= blockSize)
        return;

    T value = ((T*)s)[i];
    T value2 = ((T*)s)[i + 1];
    T * tData = (T*)t;

    __syncthreads();

    for (int k = i; k < totalSize; k += blockSize){
        tData[k] = value;
        tData[k + 1] = value2;
    }
}

/*
insert a dimension by copying the blocks for n times (where n is the size of the inerted dimension)
>> s - pointer to the source data array
>> blockSize - size of a block

>> totalSize - total size of the blocks (i.e., blockSIze * n)
>> t - pointer to the target data array
>> n - number of blocks to copy data
*/
template<class T>
__global__
void KernelUnsqueezeFlat2D(void * s, int blockSize, int totalSize, void * t, int n)
{
    __shared__ T data[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ int offsets[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    /* index of data items */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* index of data items */
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= blockSize || j >= n)
        return;

    if(threadIdx.y == 0)
        data[threadIdx.x] = ((T*)s)[i];
    if(threadIdx.x == 0)
        offsets[threadIdx.y] = blockSize * j;

    __syncthreads();

    ((T*)t)[offsets[threadIdx.y] + i] = data[threadIdx.x];
}

/*
insert a dimension by copying the blocks for n times (where n is the size of the inerted dimension)
>> s - pointer to the source data array
>> blockSize - size of a block
>> blockNum - number of the blocks
>> totalSize - total size of the blocks (i.e., blockSize * n)
>> t - pointer to the target data array
>> n - number of blocks to copy data
*/
template<class T>
__global__
void KernelUnsqueeze(void * s, int blockSize, int blockNum, int totalSize, void * t, int n)
{
    /* index of data items */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* block index */
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= blockSize || j >= blockNum)
        return;

    MTYPE offset = blockSize * j;
    T value = ((T*)s)[offset + i];
    T * tData = (T*)t + offset * n;

    __syncthreads();

    for (int k = i; k < totalSize; k += blockSize)
        tData[k] = value;
}

/*
insert a dimension by copying the blocks for n times (where n is the size of the inerted dimension)
This is special case where we actually copy a v-dimentional column vector by n times to form a v * n matrix
>> s - pointer to the source data array
>> rowNum - number of rows (i.e., dimension size of s)
>> colNum - number of columns (i.e., number of copies)
>> t - pointer to the target data array
*/
template<class T>
__global__
void KernelUnsqueezeByCol(void * s, int rowNum, int colNum, void * t)
{
    __shared__ T values[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ T * ts[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    /* column index */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* row index */
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= colNum || j >= rowNum)
        return;

    if(threadIdx.x == 0){
        values[threadIdx.y] = ((T*)s)[j];
        ts[threadIdx.y] = (T*)t + colNum * j;
    }

    __syncthreads();

    ts[threadIdx.y][i] = values[threadIdx.y];
}

/*
insert a dimension by copying the blocks for n times (where n is the size of the inerted dimension)
This is special case where we actually copy a v-dimentional column vector by n times to form a v * n matrix
And a row is very big so that it occupies the cuda threads in a block
>> s - pointer to the source data array
>> rowNum - number of rows (i.e., dimension size of s)
>> colNum - number of columns (i.e., number of copies)
>> t - pointer to the target data array
*/
template<class T>
__global__
void KernelUnsqueezeByColBigRow(void * s, int rowNum, int colNum, void * t)
{
    __shared__ T value;
    __shared__ T * tData;

    /* column index */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* row index */
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= colNum || j >= rowNum)
        return;

    if (threadIdx.x == 0) {
        value = ((T*)s)[j];
        tData = (T*)t + colNum * j;
    }

    __syncthreads();

    tData[i] = value;
}

/*
insert a dimension by copying the blocks for x times (where x is the size of the inerted dimension)
>> a - input tensor
>> b - output tensor
>> dim - where to insert the dimension
>> dSize - size of the newly-inserted dimension
*/
void _CudaUnsqueeze(const XTensor * a, XTensor * b, int dim, int dSize)
{
    int blockSize = 1;
    int blockNumA = 1;
    int blockNumB = 1;
    for (int i = dim; i < a->order; i++)
        blockSize *= a->dimSize[i];

    blockNumA = a->unitNum / blockSize;
    blockNumB = b->unitNum / blockSize;

    CheckNTErrors((blockNumA * dSize == blockNumB), "Unmatched tensors!");;

    int cudaGrids[3];
    int cudaBlocks[3];

    int devIDBackup = 0;
    ProtectCudaDev(a->devID, devIDBackup);

    if (dim == b->order - 1) {
        GDevs.GetCudaThread2D(a->devID, dSize, blockNumA, MAX_INT, cudaGrids, cudaBlocks);

        if (a->dataType == X_FLOAT && b->dataType == X_FLOAT) {
            if (cudaBlocks[1] == 1)
                KernelUnsqueezeByColBigRow<float> << <dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >> >
                                                     (a->data, blockNumA, dSize, b->data);
            else
                KernelUnsqueezeByCol<float> << <dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >> >
                                               (a->data, blockNumA, dSize, b->data);
        }
        else if (a->dataType == X_INT && b->dataType == X_INT) {
            if (cudaBlocks[1] == 1)
                KernelUnsqueezeByColBigRow<int> << <dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >> >
                                                   (a->data, blockNumA, dSize, b->data);
            else
                KernelUnsqueezeByCol<int> << <dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >> >
                                             (a->data, blockNumA, dSize, b->data);
        }
        else {
            ShowNTErrors("TODO!");
        }

        
    }
    else if(blockNumA > 1){
        GDevs.GetCudaThread2D(a->devID, blockSize, blockNumA, MAX_INT, cudaGrids, cudaBlocks);

        if (a->dataType == X_FLOAT && b->dataType == X_FLOAT) {
            KernelUnsqueeze<float> << <dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >> >
                                      (a->data, blockSize, blockNumA, blockSize * dSize, b->data, dSize);
        }
        else if (a->dataType == X_INT && b->dataType == X_INT) {
            KernelUnsqueeze<int> << <dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >> >
                                    (a->data, blockSize, blockNumA, blockSize * dSize, b->data, dSize);
        }
        else {
            ShowNTErrors("TODO!");
        }
    }
    else if(blockNumA == 1 && blockSize < MAX_CUDA_THREAD_NUM_PER_BLOCK){
        GDevs.GetCudaThread2D(a->devID, blockSize, dSize, MAX_CUDA_THREAD_NUM_PER_BLOCK/4, cudaGrids, cudaBlocks);

        if (a->dataType == X_FLOAT && b->dataType == X_FLOAT) {
            KernelUnsqueezeFlat2D<float> << <dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >> >
                                          (a->data, blockSize, blockSize * dSize, b->data, dSize);
        }
        else if (a->dataType == X_INT && b->dataType == X_INT) {
            KernelUnsqueezeFlat2D<int> << <dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >> >
                                        (a->data, blockSize, blockSize * dSize, b->data, dSize);
        }
        else {
            ShowNTErrors("TODO!");
        }
    }
    else if(blockNumA == 1 && blockSize % 2 == 0){
        GDevs.GetCudaThread(a->devID, blockSize/2, cudaGrids, cudaBlocks);

        if (a->dataType == X_FLOAT && b->dataType == X_FLOAT) {
            KernelUnsqueezeFlatBigram<float> << <dim3(cudaGrids[0]), dim3(cudaBlocks[0]) >> >
                                                (a->data, blockSize, blockSize * dSize, b->data, dSize);
        }
        else if (a->dataType == X_INT && b->dataType == X_INT) {
            KernelUnsqueezeFlatBigram<int> << <dim3(cudaGrids[0]), dim3(cudaBlocks[0]) >> >
                                              (a->data, blockSize, blockSize * dSize, b->data, dSize);
        }
        else {
            ShowNTErrors("TODO!");
        }
    }
    else if(blockNumA == 1){
        GDevs.GetCudaThread(a->devID, blockSize, cudaGrids, cudaBlocks);

        if (a->dataType == X_FLOAT && b->dataType == X_FLOAT) {
            KernelUnsqueezeFlat<float> << <dim3(cudaGrids[0]), dim3(cudaBlocks[0]) >> >
                                          (a->data, blockSize, blockSize * dSize, b->data, dSize);
        }
        else if (a->dataType == X_INT && b->dataType == X_INT) {
            KernelUnsqueezeFlat<int> << <dim3(cudaGrids[0]), dim3(cudaBlocks[0]) >> >
                                        (a->data, blockSize, blockSize * dSize, b->data, dSize);
        }
        else {
            ShowNTErrors("TODO!");
        }
    }
    else{
        ShowNTErrors("Something is wrong!");
    }

    BacktoCudaDev(a->devID, devIDBackup);
}

#endif // USE_CUDA
} // namespace nts(NiuTrans.Tensor)