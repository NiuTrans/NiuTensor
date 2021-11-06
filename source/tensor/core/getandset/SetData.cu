/* 
 * NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2018, Natural Language Processing Lab, Northeastern University.
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
 * $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-07-18
 * I'm surprised that I did not write this file till today.
 */

#include <curand.h>
#include <time.h>
#include "SetData.cuh"
#include <curand_kernel.h>
#include "../../XDevice.h"
#include "../../XUtility.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/*
set a data array with a fixed value

>> d - pointer to the data array
>> v - the initial value
>> size - size of the array
*/
template<class T>
__global__
void KernelSetDataFixed(T * d, T v, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size)
        d[i] = v;
}
template __global__ void KernelSetDataFixed<int>(int *, int, int);
template __global__ void KernelSetDataFixed<float>(float *, float, int);
template __global__ void KernelSetDataFixed<double>(double *, double, int);
//template __global__ void KernelSetDataFixed<__half>(__half*, __half, int);

/* 
generate data items with a fixed value 

>> tensor - the tensor for initialization
>> value - the initial value
*/
template<class T>
void _CudaSetDataFixed(XTensor * tensor, T value)
{
    int gridSize[3];
    int blockSize[3];

    GDevs.GetCudaThread(tensor->devID, tensor->unitNum, gridSize, blockSize);

    dim3 blocks(gridSize[0]);
    dim3 threads(blockSize[0]);

    int devIDBackup;

    ProtectCudaDev(tensor->devID, devIDBackup);

    if (tensor->dataType == X_INT)
        KernelSetDataFixed << <blocks, threads >> > ((int*)tensor->data, (int)value, tensor->unitNum);
    else if (tensor->dataType == X_FLOAT)
        KernelSetDataFixed << <blocks, threads >> > ((float*)tensor->data, (float)value, tensor->unitNum);
    else if (tensor->dataType == X_DOUBLE)
        KernelSetDataFixed << <blocks, threads >> > ((double*)tensor->data, (double)value, tensor->unitNum);
    //else if (tensor->dataType == X_FLOAT16)
    //    KernelSetDataFixed << <blocks, threads >> > ((__half*)tensor->data, (__half)value, tensor->unitNum);
    else
        ShowNTErrors("TODO! Unsupported datatype!")

    BacktoCudaDev(tensor->devID, devIDBackup);
}
template void _CudaSetDataFixed<int>(XTensor *, int);
template void _CudaSetDataFixed<float>(XTensor *, float);
template void _CudaSetDataFixed<double>(XTensor *, double);

/* 
set a float data array with a fixed value p (in int) only 
if the condition entry is non-zero 
>> d - pointer to the data array
>> c - pointer to the condition array
>> size - size of the array
>> p - the initial value
*/
template<class T>
__global__ 
void KernelSetDataFixedCond(T * d, T * c, T value, int size)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size && c[i] != 0)
        d[i] = value;
}
template __global__ void KernelSetDataFixedCond<int>(int*, int*, int, int);
template __global__ void KernelSetDataFixedCond<float>(float*, float*, float, int);
template __global__ void KernelSetDataFixedCond<double>(double*, double*, double, int);
//template __global__ void KernelSetDataFixedCond<__half>(__half*, __half*, __half, int);

/* 
generate data items with a fixed value p 
only if the condition entry is non-zero 

>> tensor - the tensor for initialization
>> condition - the condition tensor whose entry would be check to
               set the corresponding entry in "tensor"
>> value - the initial value   
*/
template<class T>
void _CudaSetDataFixedCond(XTensor* tensor, XTensor* condition, T value)
{
    int gridSize[3];
    int blockSize[3];

    GDevs.GetCudaThread(tensor->devID, tensor->unitNum, gridSize, blockSize);

    dim3 blocks(gridSize[0]);
    dim3 threads(blockSize[0]);

    int devIDBackup;
    ProtectCudaDev(tensor->devID, devIDBackup);

    if (tensor->dataType == X_INT)
        KernelSetDataFixedCond <<< blocks, threads >>> ((int*)tensor->data, (int*)condition->data,
                                                       (int)value, tensor->unitNum);
    else if (tensor->dataType == X_FLOAT)
        KernelSetDataFixedCond <<< blocks, threads >>> ((float*)tensor->data, (float*)condition->data,
                                                       (float)value, tensor->unitNum);

    else if (tensor->dataType == X_DOUBLE)
        KernelSetDataFixedCond <<< blocks, threads >>> ((double*)tensor->data, (double*)condition->data,
                                                       (double)value, tensor->unitNum);
    //else if (tensor->dataType == X_FLOAT16)
    //    KernelSetDataFixedCond <<< blocks, threads >>> ((__half*)tensor->data, (__half*)condition->data,
    //                                                   (__half)value, tensor->unitNum);
    else
        ShowNTErrors("TODO! Unsupported datatype!")

    BacktoCudaDev(tensor->devID, devIDBackup);
}
template void _CudaSetDataFixedCond<int>(XTensor*, XTensor*, int);
template void _CudaSetDataFixedCond<float>(XTensor*, XTensor*, float);
template void _CudaSetDataFixedCond<double>(XTensor*, XTensor*, double);

/* 
set data array with a uniform distribution in [low, high] 
>> deviceStates - the state of curand
>> d - float datatype pointer to the data array 
>> size - size of the array
>> lower - low value of the range
>> variance - the variance of the range
*/
__global__
void KernelSetDataRandFloat(float * d, int size, DTYPE lower, DTYPE variance)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < size) {
        d[i] = d[i] * variance + lower;
    }
}
/* 
set data array with a uniform distribution in [low, high] 
>> deviceStates - the state of curand
>> d - double datatype pointer to the data array
>> size - size of the array
>> lower - low value of the range
>> variance - the variance of the range
*/
__global__
void KernelSetDataRandDouble(double * d, int size, DTYPE lower, DTYPE variance)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < size){
        d[i] = d[i] * variance + lower;
    }
}

/*
set data items to a pre-defined value if its value >= p, set it to 0 otherwise
>> d - pointer to the data array
>> size - size of the array
>> lower - low value of the range
>> variance - the variance of the range
*/
__global__
void KernelSetDataPCut(DTYPE * d, int size, DTYPE p, DTYPE value)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size) {
        if (d[i] >= p)
            d[i] = value;
        else
            d[i] = 0;
    }
}

/* 
set data items along with a given dimension (and keep the remaining items unchanged) - kernel version
>> tensor - the tensor whose data array would be initialized
>> beg - the beginning position
>> len - length of the segment to be set
>> blockSize - size of a data block
>> blockNum - number of data blocks
*/
template<class T>
__global__
void KernelSetDataDim(T * d, int beg, int len, int blockSize, int blockNum, T p)
{
    /* offset in each block */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* block id */
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i >= blockSize || j > blockNum)
        return;

    if(i < beg || i >= beg + len)
        return;

    d[blockSize * j + i] = p;
}
template __global__ void KernelSetDataDim<int>(int*, int, int, int, int, int);
template __global__ void KernelSetDataDim<float>(float*, int, int, int, int, float);
template __global__ void KernelSetDataDim<double>(double*, int, int, int, int, double);

/* 
set data items along with a given dimension (and keep the remaining items unchanged) - cuda version
>> tensor - the tensor whose data array would be initialized
>> beg - the beginning position
>> len - length along with the given dimension
>> dim - the dimension along which we set the data
e.g., given a 3 * 3 tensor 
      1 2 3
      4 5 6
      7 8 9
      when beg = 1, len = 1, dim = 0 and p = 0, we have
      1 2 3
      0 0 0
      7 8 9
      i.e., we set all entries of row 1 to 0
*/
template<class T>
void _CudaSetDataDim(XTensor * tensor, int beg, int len, int dim, T p)
{
    int n = tensor->order;

    CheckNTErrors(tensor->dataType == DEFAULT_DTYPE, "TODO!");
    CheckNTErrors(dim < n && dim >= 0, "Illegal dimension!");
    CheckNTErrors(beg >= 0 && beg < tensor->GetDim(dim), "Illegal beginning position!");
    CheckNTErrors(beg + len >= 0 && beg + len < tensor->GetDim(dim), "Illegal length!");

    int stride = 1;
    int blockSize = 1;
    int blockNum  = 1;
    for(int i = n - 1; i > dim; i--){
        stride *= tensor->GetDim(i);
    }
    blockSize = stride * tensor->GetDim(dim);
    blockNum = tensor->unitNum / blockSize;

    int cudaGrids[3];
    int cudaBlocks[3];

    GDevs.GetCudaThread2D(tensor->devID, blockSize, blockNum, MAX_INT, cudaGrids, cudaBlocks);

    dim3 blocks(cudaGrids[0], cudaGrids[1]);
    dim3 threads(cudaBlocks[0], cudaBlocks[1]);

    int devIDBackup;
    ProtectCudaDev(tensor->devID, devIDBackup);

    if (tensor->dataType == X_INT)
        KernelSetDataDim << <blocks, threads >> > ((int*)tensor->data, beg * stride,
                                                    len * stride, blockSize, blockNum, (int)p);
    else if (tensor->dataType == X_FLOAT)
        KernelSetDataDim << <blocks, threads >> > ((float*)tensor->data, beg * stride,
                                                    len * stride, blockSize, blockNum, (float)p);

    else if (tensor->dataType == X_DOUBLE)
        KernelSetDataDim << <blocks, threads >> > ((double*)tensor->data, beg * stride,
                                                    len * stride, blockSize, blockNum, (double)p);
    else
        ShowNTErrors("TODO! Unsupported datatype!")

    BacktoCudaDev(tensor->devID, devIDBackup);
}
template void _CudaSetDataDim<int>(XTensor*, int, int, int, int);
template void _CudaSetDataDim<float>(XTensor*, int, int, int, float);
template void _CudaSetDataDim<double>(XTensor*, int, int, int, double);

/* 
modify data items along with a given index and dimension 
(and keep the remaining items unchanged) - kernel version

>> s - the pointer whose data would be modified
>> m - the pointer whose data would be used to modify the data pointed by s
>> blockNum - number of data blocks
>> blockSize - size of a data block
>> stride - stride of a data block
*/
__global__
void KernelSetDataIndexed(DTYPE * s, DTYPE * m, int blockNum, int blockSize, int stride)
{
    /* offset in each block */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* block id */
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    
    if(i >= stride || j >= blockNum)
        return;

    int x = blockSize * j + i;
    int y = stride * j + i;
    s[x] = m[y];
}

/*
modify data items along with a given index and dimension (and keep the remaining items unchanged) 
>> source - the tensor whose data array would be modified
>> modify - the tensor whose data array would be used to modify the source tensor
>> dim - the dimension along which we modify the tensor
>> index - index of the given dimension
e.g., given a source tensor (3, 3)
      1 2 3
      4 5 6
      7 8 9
      given a modified tensor (3)
      1 2 3
      when dim = 0, index = 1, we have
      1 2 3
      1 2 3
      7 8 9
      i.e., we set entries of row 1 to {1, 2, 3}
*/
void _CudaSetDataIndexed(XTensor * source, XTensor * modify, int dim, int index)
{
    int order = source->order;
    int size = source->GetDim(dim);

    CheckNTErrors(source->dataType == DEFAULT_DTYPE, "TODO!");
    CheckNTErrors(dim >= 0 && dim < order, "Illegal dimension!");
    CheckNTErrors(index >= 0 && index < size, "Illegal index!");
    
    int stride = 1;
    int blockSize = 1;
    int blockNum  = 1;

    for(int i = order - 1; i > dim; i--){
        stride *= source->GetDim(i);
    }

    blockSize = stride * source->GetDim(dim);
    blockNum = source->unitNum / blockSize;

    int cudaGrids[3];
    int cudaBlocks[3];

    GDevs.GetCudaThread2D(source->devID, stride, blockNum, MAX_INT, cudaGrids, cudaBlocks);

    dim3 blocks(cudaGrids[0], cudaGrids[1]);
    dim3 threads(cudaBlocks[0], cudaBlocks[1]);

    int devIDBackup;
    ProtectCudaDev(source->devID, devIDBackup);
    
    KernelSetDataIndexed<<<blocks, threads >>>((DTYPE*)source->data + index * stride, (DTYPE*)modify->data, 
                                                blockNum, blockSize, stride);

    BacktoCudaDev(source->devID, devIDBackup);
}

/* 
set lower triangular matrics for each block

>> d - pointer to the data array
>> l - row number (or column number) of each block, i.e, 
       a block is l * l matrix
>> blockSize - size of each block (blockSize = l * l)
>> blockNum - number of the blocks
>> p - the value for each entry of the lower triangular matrics
>> shift - the offset from diagonal
   e.g., for a 3* 3 tensor, 
         when p = 1 ans shift = 0, we have
         1 0 0
         1 1 0
         1 1 1
         when p = 2 and shift = -1, we have
         0 0 0
         2 0 0
         2 2 0
*/
__global__
void KernelSetDataLowTri(DTYPE * d, int l, int blockSize, int blockNum, DTYPE p, int shift)
{
    /* offset in each block */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* block id */
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i >= blockSize || j > blockNum)
        return;

    int row = i / l;
    int col = i % l;
    DTYPE * d2 = d + blockSize * j + row * l + col;

    if(col <= row + shift)
        *d2 = p;
    else
        *d2 = 0;
}

/*
generate data as lower triangular matrics for last two dimensions (cuda version)

>> tensor - the tensor whose data to be set
>> value - the value for each entry of the lower triangular matrics
>> shift - the offset from diagonal

   e.g., for a 3 * 3 tensor,
         when value = 1 ans shift = 0, we have
         1 0 0
         1 1 0
         1 1 1
         when value = 2 and shift = -1, we have
         0 0 0
         2 0 0
         2 2 0
*/
void _CudaSetDataLowTri(XTensor * tensor, DTYPE value, int shift)
{
    int size = tensor->GetDim(-1);
    int blockSize = size * size;
    int blockNum = tensor->unitNum / blockSize;

    int cudaGrids[3];
    int cudaBlocks[3];

    GDevs.GetCudaThread2D(tensor->devID, blockSize, blockNum, MAX_INT, cudaGrids, cudaBlocks);

    dim3 blocks(cudaGrids[0], cudaGrids[1]);
    dim3 threads(cudaBlocks[0], cudaBlocks[1]);

    int devIDBackup;
    ProtectCudaDev(tensor->devID, devIDBackup);

    KernelSetDataLowTri<<<blocks, threads >>>((DTYPE*)tensor->data, size, blockSize, blockNum, value, shift);

    BacktoCudaDev(tensor->devID, devIDBackup);
}

/*
generate data items with a uniform distribution in [lower, upper]
>> tensor - the tensor whose data array would be initialized
>> lower - lower value of the range
>> upper - upper value of the range
*/
void _CudaSetDataRand(const XTensor * tensor, DTYPE lower, DTYPE upper)
{
    CheckNTErrors(upper > lower, "the high value must be greater than low value!");

    int gridSize[3];
    int blockSize[3];

    GDevs.GetCudaThread(tensor->devID, tensor->unitNum, gridSize, blockSize);

    dim3 blocks(gridSize[0]);
    dim3 threads(blockSize[0]);

    int devIDBackup;
    ProtectCudaDev(tensor->devID, devIDBackup);
    
    curandGenerator_t & gen = GDevs.GPUs[tensor->devID].gen;
    curandGenerateUniform(gen, (float*)tensor->data, tensor->unitNum);
    
    DTYPE variance = upper - lower;

    if(variance != 1.0F || lower != 0){
        if (tensor->dataType == X_FLOAT)
            KernelSetDataRandFloat  <<<blocks, threads >>>
                                     ((float*) tensor->data, tensor->unitNum, lower, variance);
        else if (tensor->dataType == X_DOUBLE)
            KernelSetDataRandDouble <<<blocks, threads >>>
                                     ((double*)tensor->data, tensor->unitNum, lower, variance);
    }

    BacktoCudaDev(tensor->devID, devIDBackup);
}

/* 
generate data items with a uniform distribution in [lower, upper] and set
the item to a pre-defined value if the item >= p, set the item to 0 otherwise 
>> tensor - the tensor whose data array would be initialized
>> lower - lower value of the range
>> upper - upper value of the range
>> p - the threshold
>> value - the value we intend to assign to the item
*/
void _CudaSetDataRandP(const XTensor * tensor, DTYPE lower, DTYPE upper, DTYPE p, DTYPE value)
{
    _CudaSetDataRand(tensor, lower, upper);

    int gridSize[3];
    int blockSize[3];

    GDevs.GetCudaThread(tensor->devID, tensor->unitNum, gridSize, blockSize);

    dim3 blocks(gridSize[0]);
    dim3 threads(blockSize[0]);

    int devIDBackup;
    ProtectCudaDev(tensor->devID, devIDBackup);
    
    KernelSetDataPCut << <blocks, threads >> >((float*)tensor->data, tensor->unitNum, p, value);

    BacktoCudaDev(tensor->devID, devIDBackup);
}

/*
set the data with an array of offsets (kernel version)
>> data - pointer to the data array
>> offsets - offset for each data item
>> value - value of the data items
>> num - number of the data items
*/
__global__
void KernelSetDataWithOffset(DTYPE * data, MTYPE * offsets, DTYPE value, MTYPE num)
{
    /* index */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num)
        data[offsets[i]] = value;
}

/*
set the data with an array of offsets (cuda version)
>> tensor - the tensor that keeps the data
>> offsets - offset for each data item
>> value - value of the data items
>> num - number of the data items
*/
void _CudaSetDataWithOffset(XTensor * tensor, MTYPE * offsets, DTYPE value, MTYPE num)
{
    CheckNTErrors(tensor->dataType == X_FLOAT, "Data type is incorrect!");

    int gridSize[3];
    int blockSize[3];

    GDevs.GetCudaThread(tensor->devID, (int)num, gridSize, blockSize);

    dim3 blocks(gridSize[0]);
    dim3 threads(blockSize[0]);

    int devIDBackup;
    ProtectCudaDev(tensor->devID, devIDBackup);

    KernelSetDataWithOffset << <blocks, threads >> > ((DTYPE*)tensor->data, offsets, value, num);

    BacktoCudaDev(tensor->devID, devIDBackup);
}

/*
set the data with an array of offsets (kernel version)
>> data - pointer to the data array
>> offsets - offset for each data item
>> value - value of the data items
>> num - number of the data items
>> dataType - the data type of the data and values
*/
__global__
void KernelSetDataWithOffsetAndValue(void * data, MTYPE * offsets, void * values, MTYPE num, TENSOR_DATA_TYPE dataType)
{
    /* index */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num) {
        if (dataType == X_INT)
            *((int *)data + offsets[i]) = *((int *)values + i);
        else if (dataType == X_FLOAT)
            *((float *)data + offsets[i]) = *((float *)values + i);
    }
}

/*
set the data with an array of values 
>> tensor - the tensor that keeps the data
>> offsets - offset for each data item
>> value - value of the ech data item
>> num - number of the data items
*/
void _CudaSetDataWithOffsetAndValue(XTensor * tensor, MTYPE * offsets, void * values, MTYPE num)
{

    XMem * mem = tensor->mem;
    MTYPE offsetSize = num * sizeof(MTYPE);
    MTYPE valueSize;

    if (tensor->dataType == X_INT)
        valueSize = num * sizeof(int);
    else if (tensor->dataType == X_FLOAT)
        valueSize = num * sizeof(float);
    else
        ShowNTErrors("TO DO!!!");

    int gridSize[3];
    int blockSize[3];

    GDevs.GetCudaThread(tensor->devID, (int)num, gridSize, blockSize);

    dim3 blocks(gridSize[0]);
    dim3 threads(blockSize[0]);

    int devIDBackup;
    ProtectCudaDev(tensor->devID, devIDBackup);

    /*MTYPE * offsetsCuda = mem != NULL ? 
                            (MTYPE*)mem->AllocBuf(mem->devID, offsetSize) : 
                            (MTYPE*)XMemAlloc(tensor->devID, offsetSize);
    void * valuesCuda = mem != NULL ?
                        mem->AllocBuf(mem->devID, valueSize) :
                        XMemAlloc(tensor->devID, valueSize);*/
    MTYPE * offsetsCuda;
    void * valuesCuda; 
    if (mem != NULL) {
        mem->LockBuf();
        offsetsCuda = (MTYPE*)mem->AllocBuf(mem->devID, offsetSize);
        valuesCuda = mem->AllocBuf(mem->devID, valueSize);
    }
    else {
        offsetsCuda = (MTYPE*)XMemAlloc(tensor->devID, offsetSize);
        valuesCuda = XMemAlloc(tensor->devID, valueSize);
    }

    if (mem != NULL) {
        XMemCopy(offsetsCuda, mem->devID, offsets, -1, offsetSize);
        XMemCopy(valuesCuda, mem->devID, values, -1, valueSize);
    }
    else {
        XMemCopy(offsetsCuda, tensor->devID, offsets, -1, offsetSize);
        XMemCopy(valuesCuda, tensor->devID, values, -1, valueSize);
    }

    KernelSetDataWithOffsetAndValue<<<blocks, threads >>> (tensor->data, offsetsCuda, valuesCuda, num, tensor->dataType);

    if (mem != NULL) {
        mem->ReleaseBuf(mem->devID, valueSize);
        mem->ReleaseBuf(mem->devID, offsetSize);
        mem->UnlockBuf();
    }
    else {
        XMemFree(tensor->devID, valuesCuda);
        XMemFree(tensor->devID, offsetsCuda);
    }

    BacktoCudaDev(tensor->devID, devIDBackup);
}

#endif // USE_CUDA
} // namespace nts(NiuTrans.Tensor)
