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

#include "CopyBlocksInGrid.h"
#include "CopyBlocksInGrid.cuh"
#include "../../XDevice.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/*
copy data by index (device code)
here we keep all the data of a grid within the shared memory, and then move it
the indexed positions to the target place
>> source - pointer to the source data array
>> blockSize - size of a data block
>> blockNum - number of the blocks (in a grid)
>> gridNum - number of the grids.
Note that a grid may have a number of blocks
>> target - pointer to the target data array
>> index - source block id for each target block
*/
template<class T>
__global__
void KernelCopyBlocksInGrid(T * source, int blockSize, int blockNum, int gridNum, T * target, int * index)
{
    __shared__ T   data[SHARED_MEMORY_SIZE / sizeof(T) - 4 * MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ int indexData[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ int indexOffset[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    /* item index */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* grid index */
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (j >= gridNum || i >= blockDim.x)
        return;

    if (i < blockNum) {
        indexData[i] = index[j * blockNum + i];
        indexOffset[i] = i * blockDim.x;
    }

    __syncthreads();

    int gridSize = blockSize * blockNum;
    T * s = source + j * gridSize;
    T * t = target + j * gridSize;

    for (int k = i, k2 = i; k < blockSize; k += blockDim.x) {

        /* load data into shared memroy */
        for (int offset = 0, offset2 = 0; offset < gridSize; offset += blockSize, offset2 += blockDim.x) {
            data[offset2 + k2] = s[offset + k];
        }

        __syncthreads();

        /* distribute data to the target grid */
        for (int p = 0, offset = 0; p < blockNum; p++, offset += blockSize) {
            int blockIndex = indexData[p];
            if (blockIndex >= 0 && blockIndex < blockNum) {
                t[offset + k] = data[indexOffset[blockIndex] + k2];
            }
        }

        __syncthreads();
    }
}

/*
copy data by index (device code)
here we keep all the data of a grid within the shared memory, and then move it
the indexed positions to the target place
>> source - pointer to the source data array
>> blockSize - size of a data block
>> blockNum - number of the blocks (in a grid)
>> gridNum - number of the grids.
Note that a grid may have a number of blocks
>> target - pointer to the target data array
>> index - source block id for each target block
*/
template<class T, int goodBlockNum, int stepScale>
__global__
void KernelCopyBlocksInGridFast(T * source, int blockSize, int blockNum, int gridNum, T * target, int * index)
{
    __shared__ T   data[SHARED_MEMORY_SIZE / sizeof(T) - 2 * MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ int indexData[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ int indexOffset[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    /* item index */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* grid index */
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (j >= gridNum || i >= blockDim.x)
        return;

    int step = stepScale == 1 ? blockDim.x : blockDim.x * stepScale;

    if (i < blockNum) {
        indexData[i] = index[j * blockNum + i];
        indexOffset[i] = i * step;
    }

    __syncthreads();

    int gridSize = blockSize * blockNum;
    T * s = source + j * gridSize;
    T * t = target + j * gridSize;

    for (int k = i, k2 = i; k < blockSize; k += step) {

        int bidx;
        int offset = k;
        int offset2 = k2;
        int stepInStep = 0;

        /* load data into shared memroy */
        for (int i = 0; i < stepScale && offset < blockSize; i++) {
            if (goodBlockNum >= 1) { data[offset2] = s[offset];  offset += blockSize; offset2 += step; }
            if (goodBlockNum >= 2) { data[offset2] = s[offset];  offset += blockSize; offset2 += step; }
            if (goodBlockNum >= 3) { data[offset2] = s[offset];  offset += blockSize; offset2 += step; }
            if (goodBlockNum >= 4) { data[offset2] = s[offset];  offset += blockSize; offset2 += step; }
            if (goodBlockNum >= 5) { data[offset2] = s[offset];  offset += blockSize; offset2 += step; }
            if (goodBlockNum >= 6) { data[offset2] = s[offset];  offset += blockSize; offset2 += step; }
            if (goodBlockNum >= 7) { data[offset2] = s[offset];  offset += blockSize; offset2 += step; }
            if (goodBlockNum >= 8) { data[offset2] = s[offset];  offset += blockSize; offset2 += step; }
            if (goodBlockNum >= 9) { data[offset2] = s[offset];  offset += blockSize; offset2 += step; }
            if (goodBlockNum >= 10) { data[offset2] = s[offset];  offset += blockSize; offset2 += step; }
            if (goodBlockNum >= 11) { data[offset2] = s[offset];  offset += blockSize; offset2 += step; }
            if (goodBlockNum >= 12) { data[offset2] = s[offset];  offset += blockSize; offset2 += step; }
            if (goodBlockNum >= 13) {
                for (; offset < gridSize; offset += blockSize, offset2 += step) {
                    data[offset2] = s[offset];
                }
            }

            if (stepScale > 1) {
                stepInStep += blockDim.x;
                offset = k + stepInStep;
                offset2 = k2 + stepInStep;
            }
        }

        __syncthreads();

        offset = k;
        offset2 = k2;
        stepInStep = 0;

        /* distribute data to the target grid */
        for (int i = 0; i < stepScale && offset < blockSize; i++) {
            if (goodBlockNum >= 1) { bidx = indexData[0];  if (bidx >= 0 && bidx < blockNum) { t[offset] = data[indexOffset[bidx] + offset2]; }  offset += blockSize; }
            if (goodBlockNum >= 2) { bidx = indexData[1];  if (bidx >= 0 && bidx < blockNum) { t[offset] = data[indexOffset[bidx] + offset2]; }  offset += blockSize; }
            if (goodBlockNum >= 3) { bidx = indexData[2];  if (bidx >= 0 && bidx < blockNum) { t[offset] = data[indexOffset[bidx] + offset2]; }  offset += blockSize; }
            if (goodBlockNum >= 4) { bidx = indexData[3];  if (bidx >= 0 && bidx < blockNum) { t[offset] = data[indexOffset[bidx] + offset2]; }  offset += blockSize; }
            if (goodBlockNum >= 5) { bidx = indexData[4];  if (bidx >= 0 && bidx < blockNum) { t[offset] = data[indexOffset[bidx] + offset2]; }  offset += blockSize; }
            if (goodBlockNum >= 6) { bidx = indexData[5];  if (bidx >= 0 && bidx < blockNum) { t[offset] = data[indexOffset[bidx] + offset2]; }  offset += blockSize; }
            if (goodBlockNum >= 7) { bidx = indexData[6];  if (bidx >= 0 && bidx < blockNum) { t[offset] = data[indexOffset[bidx] + offset2]; }  offset += blockSize; }
            if (goodBlockNum >= 8) { bidx = indexData[7];  if (bidx >= 0 && bidx < blockNum) { t[offset] = data[indexOffset[bidx] + offset2]; }  offset += blockSize; }
            if (goodBlockNum >= 9) { bidx = indexData[8];  if (bidx >= 0 && bidx < blockNum) { t[offset] = data[indexOffset[bidx] + offset2]; }  offset += blockSize; }
            if (goodBlockNum >= 10) { bidx = indexData[9];  if (bidx >= 0 && bidx < blockNum) { t[offset] = data[indexOffset[bidx] + offset2]; }  offset += blockSize; }
            if (goodBlockNum >= 11) { bidx = indexData[10]; if (bidx >= 0 && bidx < blockNum) { t[offset] = data[indexOffset[bidx] + offset2]; }  offset += blockSize; }
            if (goodBlockNum >= 12) { bidx = indexData[11]; if (bidx >= 0 && bidx < blockNum) { t[offset] = data[indexOffset[bidx] + offset2]; }  offset += blockSize; }
            if (goodBlockNum >= 13) {
                for (int p = 12; p < blockNum; p++, offset += blockSize) {
                    bidx = indexData[p];
                    if (bidx >= 0 && bidx < blockNum) {
                        t[offset] = data[indexOffset[bidx] + offset2];
                    }
                }
            }

            if (stepScale > 1) {
                stepInStep += blockDim.x;
                offset = k + stepInStep;
                offset2 = k2 + stepInStep;
            }
        }

        __syncthreads();
    }
}

/*
copy data by index (host code)
>> source - pointer to the source data array
>> blockSize - size of a data block
>> blockNum - number of the blocks (in a grid)
>> gridNum - number of the grids.
Note that a grid may have a number of blocks
>> target - pointer to the target data array
>> index - source block id for each target block (on the device)
>> itemSize - size of each data item
>> myMem - the memory pool
*/
void _CudaCopyBlocksInGrid(void * source, int blockSize, int blockNum, int gridNum, void * target, int * index, int itemSize, XMem * myMem)
{
    CheckNTErrors((myMem != NULL && myMem->devID >= 0), "This code must be run on GPUs!");
    CheckNTErrors((itemSize == sizeof(int)), "TODO!");

    int cudaGrids[3];
    int cudaBlocks[3];
    int threadNum = MIN(MAX(blockSize, blockNum), MAX_CUDA_THREAD_NUM_PER_BLOCK);

    int devIDBackup;
    ProtectCudaDev(myMem->devID, devIDBackup);

    GDevs.GetCudaThread2D(myMem->devID, threadNum, gridNum * blockNum, INT_MAX, cudaGrids, cudaBlocks);

    cudaBlocks[1] = 1;
    cudaGrids[0] = 1;
    cudaGrids[1] = gridNum;

    CheckNTErrors(((SHARED_MEMORY_SIZE / itemSize - 2 * MAX_CUDA_THREAD_NUM_PER_BLOCK) > cudaBlocks[0] * blockNum),
        "No enough shared memory!");

    if (blockNum == 4) {
        if ((SHARED_MEMORY_SIZE / itemSize - 2 * MAX_CUDA_THREAD_NUM_PER_BLOCK) >= 2 * cudaBlocks[0] * blockNum)
            KernelCopyBlocksInGridFast<int, 4, 2> << <dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >> >
                                                    ((int*)source, blockSize, blockNum, gridNum, (int*)target, index);
        else
            KernelCopyBlocksInGridFast<int, 4, 1> << <dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >> >
                                                    ((int*)source, blockSize, blockNum, gridNum, (int*)target, index);
    }
    else if (blockNum == 6) {
        if ((SHARED_MEMORY_SIZE / itemSize - 2 * MAX_CUDA_THREAD_NUM_PER_BLOCK) >= 2 * cudaBlocks[0] * blockNum)
            KernelCopyBlocksInGridFast<int, 6, 2> << <dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >> >
                                                    ((int*)source, blockSize, blockNum, gridNum, (int*)target, index);
        else
            KernelCopyBlocksInGridFast<int, 6, 1> << <dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >> >
                                                    ((int*)source, blockSize, blockNum, gridNum, (int*)target, index);
    }
    else if (blockNum == 8) {
        if ((SHARED_MEMORY_SIZE / itemSize - 2 * MAX_CUDA_THREAD_NUM_PER_BLOCK) >= 2 * cudaBlocks[0] * blockNum)
            KernelCopyBlocksInGridFast<int, 8, 2> << <dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >> >
                                                    ((int*)source, blockSize, blockNum, gridNum, (int*)target, index);
        else
            KernelCopyBlocksInGridFast<int, 8, 1> << <dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >> >
                                                    ((int*)source, blockSize, blockNum, gridNum, (int*)target, index);
    }
    else if (blockNum == 12) {
        if ((SHARED_MEMORY_SIZE / itemSize - 2 * MAX_CUDA_THREAD_NUM_PER_BLOCK) >= 2 * cudaBlocks[0] * blockNum)
            KernelCopyBlocksInGridFast<int, 12, 2> << <dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >> >
                                                     ((int*)source, blockSize, blockNum, gridNum, (int*)target, index);
        else
            KernelCopyBlocksInGridFast<int, 12, 1> << <dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >> >
                                                     ((int*)source, blockSize, blockNum, gridNum, (int*)target, index);
    }
    else {
        KernelCopyBlocksInGrid<int> << <dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >> >
                                      ((int*)source, blockSize, blockNum, gridNum, (int*)target, index);
    }

    BacktoCudaDev(myMem->devID, devIDBackup);
}
#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)