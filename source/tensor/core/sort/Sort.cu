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
#include "../../XUtility.h"
#include "../../XTensor.h"
#include "Sort.h"
#include "Sort.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/*
bitonic sort (for each row in a matrix)
>> data - pointer to the data array
>> index - index data array
>> j - segment/distance for comparsion
>> k - length of the monotonic sequence
>> m - column number of the matrix
>> n - row number of the matrix
*/
template<class T> __global__
void KernelBitonicSort2D(void * data, int j, int k, int m, int n)
{
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx >= m || row >= n)
        return;

    T * items = (T*)data + m * row;

    int ixj = idx^j;
    if (ixj > idx) {
        if ((idx&k) == 0 && items[idx] < items[ixj]) {
            T tmp = items[idx];
            items[idx] = items[ixj];
            items[ixj] = tmp;
        }
        if ((idx&k) != 0 && items[idx] > items[ixj]) {
            T tmp = items[idx];
            items[idx] = items[ixj];
            items[ixj] = tmp;
        }
    }
}

/*
bitonic sort (for each row in a matrix) with index
>> data - pointer to the data array
>> index - index data array
>> j - segment/distance for comparsion
>> k - length of the monotonic sequence
>> m - column number of the matrix
>> n - row number of the matrix
*/
template<class T> __global__
void KernelBitonicSort2D(void * data, int * index, int j, int k, int m, int n)
{
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx >= m || row >= n)
        return;

    T * items = (T*)data + m * row;
    int * indexOnSite = index + m * row;

    int ixj = idx^j;
    if (ixj > idx) {
        if ((idx&k) == 0 && items[idx] < items[ixj]) {
            T tmp = items[idx];
            items[idx] = items[ixj];
            items[ixj] = tmp;
            int tmp2 = indexOnSite[idx];
            indexOnSite[idx] = indexOnSite[ixj];
            indexOnSite[ixj] = tmp2;
        }
        if ((idx&k) != 0 && items[idx] > items[ixj]) {
            T tmp = items[idx];
            items[idx] = items[ixj];
            items[ixj] = tmp;
            int tmp2 = indexOnSite[idx];
            indexOnSite[idx] = indexOnSite[ixj];
            indexOnSite[ixj] = tmp2;
        }
    }
}

/*
reorganize data blocks (in a tensor) into a matrix. In each (source) block
we have stride * strideNum items, where strideNum means the items along the
leading dimension. In the target matrix, each row keeps strideNum items along
the leading dimension in each source block.
>> source - source data array
>> target - target data array
>> srcStride - how many items we need to go over we move to the next
>> srcStrideNum - size of the leading dimension
>> srcBlockNum - number of the source blocks
>> tgtColNum - number of columns in the target matrix
>> tgtRowNum - number of rows in the target matrix
*/
template<class T> __global__
void KernelReorganize(void * source, void * target,
        int srcStride, int srcStrideNum, int srcBlockNum,
        int tgtColNum, int tgtRowNum)
{
    __shared__ int iBlock[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ int iOffset[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    /* index along the "stride" dimension */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* index along the leading dimension */
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= srcStride * srcBlockNum || j >= srcStrideNum)
        return;

    if (threadIdx.y == 0) {
        iBlock[threadIdx.x] = i / srcStride;
        iOffset[threadIdx.x] = i % srcStride;
    }
    __syncthreads();

    T * s = (T*)source + (iBlock[threadIdx.x] * srcStrideNum + j) * srcStride + iOffset[threadIdx.x];
    T * t = (T*)target + (iBlock[threadIdx.x] * srcStride + iOffset[threadIdx.x]) * tgtColNum + j;
    *t = *s;
}

/*
copy back for "KernelReorganize"
>> source - source data array
>> target - target data array
>> srcColNum - number of columns in the source matrix
>> srcRowNum - number of rows in the source matrix
>> tgtStride - how many items we need to go over we move to the next
>> tgtStrideNum - size of the leading dimension
>> tgtBlockNum - number of the target blocks
*/
template<class T> __global__
void KernelReorganizeBack(void * source, void * target,
        int srcColNum, int srcRowNum,
        int tgtStride, int tgtStrideNum, int tgtBlockNum)
{
    __shared__ int iBlock[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ int iOffset[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    /* index along the "stride" dimension */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* index along the leading dimension */
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= tgtStride * tgtBlockNum || j >= tgtStrideNum)
        return;

    if (threadIdx.y == 0) {
        iBlock[threadIdx.x] = i / tgtStride;
        iOffset[threadIdx.x] = i % tgtStride;
    }
    __syncthreads();

    T * s = (T*)source + (iBlock[threadIdx.x] * tgtStride + iOffset[threadIdx.x]) * srcColNum + j;
    T * t = (T*)target + (iBlock[threadIdx.x] * tgtStrideNum + j) * tgtStride + iOffset[threadIdx.x];
    *t = *s;
}

/*
set the data arrry with a default value
>> data - data array
>> value - default value
>> size - size of the array
*/
template<class T> __global__
void KernelSetDataArray(T * data, T value, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size)
        data[i] = value;
}


/*
sort the tensor along a given dimension
>> a - input
>> b - output
>> indexA - input index tensor
>> indexB - output index tensor
>> dim - specified dimension
>> k - top-k results are returned
*/
void _CudaSortBig(const XTensor * a, XTensor * b, XTensor * indexA, XTensor * indexB, int dim, int k)
{
    CheckNTErrors((a && b), "Empty input tensor!");
    CheckNTErrors((a->unitSize == b->unitSize), "Unmatched tensors!");
    CheckNTErrors((a->order > dim && dim >= 0), "Incorrect dimension specified!");
    CheckNTErrors((a->dataType == DEFAULT_DTYPE), "TODO!");

    if (k < 0 || k > b->dimSize[dim])
        k = b->dimSize[dim];

    XMem * mem = a->mem;

    int stride = 1;
    int blockNum = 1;
    int strideNum = a->dimSize[dim];
    for (int i = 0; i < dim; i++)
        blockNum *= a->dimSize[i];

    for (int i = dim + 1; i < a->order; i++)
        stride *= a->dimSize[i];

    int m = GetNextPower2(strideNum);
    int n = stride * blockNum;

    void * buf = mem != NULL ? mem->AllocBuf(a->devID, n * m * a->unitSize) : XMemAlloc(a->devID, n * m * a->unitSize);
    void * bufIndex = NULL;
    if (indexA != NULL && indexB != NULL) {
        bufIndex = mem != NULL ? mem->AllocBuf(a->devID, n * m * sizeof(int)) : XMemAlloc(a->devID, n * m * sizeof(int));
    }

    int cudaGrids[3];
    int cudaBlocks[3];

    GDevs.GetCudaThread(a->devID, m * n, cudaGrids, cudaBlocks);

    int devIDBackup;
    ProtectCudaDev(a->devID, devIDBackup);

    /* set the buffer to the "min" value */
    KernelSetDataArray<DTYPE> << <dim3(cudaGrids[0]), dim3(cudaBlocks[0]) >> >
                                ((DTYPE*)buf, DTYPE_MIN, m * n);

    GDevs.GetCudaThread2D(a->devID, strideNum, n, MAX_INT, cudaGrids, cudaBlocks);

    /* reorganize the data into a matrix */
    KernelReorganize<DTYPE> << <dim3(cudaGrids[1], cudaGrids[0]), dim3(cudaBlocks[1], cudaBlocks[0]) >> >
                               (a->data, buf, stride, strideNum, blockNum, m, n);

    /* reorganize the index into a matrix */
    if (indexA != NULL && indexB != NULL)
        KernelReorganize<int> << <dim3(cudaGrids[1], cudaGrids[0]), dim3(cudaBlocks[1], cudaBlocks[0]) >> >
                                      (indexA->data, bufIndex, stride, strideNum, blockNum, m, n);

    GDevs.GetCudaThread2D(a->devID, m, n, MAX_INT, cudaGrids, cudaBlocks);

    /* bitonic sorting */
    for (int i = 2; i <= m; i <<= 1) {
        for (int j = i >> 1; j > 0; j = j >> 1) {
            if (indexA != NULL && indexB != NULL) {
                KernelBitonicSort2D<DTYPE> << <dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >> >
                                              (buf, (int*)bufIndex, j, i, m, n);
            }
            else {
                KernelBitonicSort2D<DTYPE> << <dim3(cudaGrids[0], cudaGrids[1]), dim3(cudaBlocks[0], cudaBlocks[1]) >> >
                                              (buf, j, i, m, n);
            }
        }
    }

    GDevs.GetCudaThread2D(a->devID, k, n, MAX_INT, cudaGrids, cudaBlocks);

    /* copy result to the output tensor */
    KernelReorganizeBack<DTYPE> << <dim3(cudaGrids[1], cudaGrids[0]), dim3(cudaBlocks[1], cudaBlocks[0]) >> >
        (buf, b->data, m, n, stride, k, blockNum);

    if (indexA != NULL && indexB != NULL)
        KernelReorganizeBack<int> << <dim3(cudaGrids[1], cudaGrids[0]), dim3(cudaBlocks[1], cudaBlocks[0]) >> >
                                      (bufIndex, indexB->data, m, n, stride, k, blockNum);

    if (mem != NULL)
        mem->ReleaseBuf(a->devID, n * m * a->unitSize);
    else
        XMemFree(a->devID, buf);
    if (indexA != NULL && indexB != NULL)
        if (mem != NULL)
            mem->ReleaseBuf(a->devID, n * m * sizeof(int));
        else
            XMemFree(a->devID, bufIndex);

    ProtectCudaDev(a->devID, devIDBackup);
}

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)