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
#include "../../XUtility.h"
#include "ReduceMax.h"
#include "ReduceMax.cuh"

namespace nts{ // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA


/*
use PTX code to reduce float data
*/
#define SHLFUNCFLOAT(funcName, reducePTXOp)                         \
__device__ __forceinline__                                     \
float funcName(float input)                                    \
{                                                              \
    float output;                                              \
    asm volatile(                                              \
        "{"                                                    \
        ".reg .f32 r0;"                                        \
        ".reg .pred p;"                                        \
        "shfl.sync.down.b32  r0, %1, 0x10, 0x1f,0xffffffff;"   \
        "setp."#reducePTXOp".f32    p,%1,r0;"                  \
        "@p mov.f32     %1,r0;"                                \
        "shfl.sync.down.b32  r0, %1, 0x8, 0xf,0xffffffff;"     \
        "setp."#reducePTXOp".f32    p,%1,r0;"                  \
        "@p mov.f32     %1,r0;"                                \
        "shfl.sync.down.b32  r0, %1, 0x4, 0x7,0xffffffff;"     \
        "setp."#reducePTXOp".f32    p,%1,r0;"                  \
        "@p mov.f32     %1,r0;"                                \
        "shfl.sync.down.b32  r0, %1, 0x2, 0x3,0xffffffff;"     \
        "setp."#reducePTXOp".f32    p,%1,r0;"                  \
        "@p mov.f32     %1,r0;"                                \
        "shfl.sync.down.b32  r0, %1, 0x1, 0x1,0xffffffff;"     \
        "setp."#reducePTXOp".f32    p, %1, r0; "               \
        "@p mov.f32     %1,r0;"                                \
        "mov.f32        %0,%1;"                                \
        "}"                                                    \
        : "=f"(output) : "f"(input));                          \
    return output;                                             \
}

SHLFUNCFLOAT(shflDownReduceMax, lt)
SHLFUNCFLOAT(shflDownReduceMin, gt)

/*
use PTX code to reduce int data
*/
#define SHLFUNCINT(funcName, reducePTXOp)                      \
__device__ __forceinline__                                     \
int funcName(int input)                                        \
{                                                              \
    int output;                                                \
    asm volatile(                                              \
        "{"                                                    \
        ".reg .s32 r0;"                                        \
        ".reg .pred p;"                                        \
        "shfl.sync.down.b32  r0, %1, 0x10, 0x1f,0xffffffff;"   \
        "setp."#reducePTXOp".s32    p,%1,r0;"                  \
        "@p mov.s32     %1,r0;"                                \
        "shfl.sync.down.b32  r0, %1, 0x8, 0xf,0xffffffff;"     \
        "setp."#reducePTXOp".s32    p,%1,r0;"                  \
        "@p mov.s32     %1,r0;"                                \
        "shfl.sync.down.b32  r0, %1, 0x4, 0x7,0xffffffff;"     \
        "setp."#reducePTXOp".s32    p,%1,r0;"                  \
        "@p mov.s32     %1,r0;"                                \
        "shfl.sync.down.b32  r0, %1, 0x2, 0x3,0xffffffff;"     \
        "setp."#reducePTXOp".s32    p,%1,r0;"                  \
        "@p mov.s32     %1,r0;"                                \
        "shfl.sync.down.b32  r0, %1, 0x1, 0x1,0xffffffff;"     \
        "setp."#reducePTXOp".s32    p, %1, r0; "               \
        "@p mov.s32     %1,r0;"                                \
        "mov.s32        %0,%1;"                                \
        "}"                                                    \
        : "=r"(output) : "r"(input));                          \
    return output;                                             \
}

SHLFUNCINT(shflDownReduceMax, lt)
SHLFUNCINT(shflDownReduceMin, gt)

/* 
reduce a tensor to another that keeps the max value along a dimension  - slow version
Given a block of data, we go over each dimension i in the stride and we have
sum_i = max_{0<=j<strideNum} input_{i,j}
where we can view the block as a matrix and input_{i,j} represent the item at the
crossing of the i-th columne and the j-th row.
>> input - the input array (representing a tensor)
>> output - the sum over each block. NOTE: output is also an array
>> stride - stride that we need to move to the next item
>> strideNum - how many strides we need to finish the reduce
>> reducedStrideNum - the number of strides after reducation 
>> blockSize - size of the block (i.e., stride * strideNum)
>> blockNum - how many blocks
*/
#define KERNELREDUCEFUN3(funName, opName, initData)                                                         \
 __global__                                                                                                 \
void funName(DTYPE * input, DTYPE * output,                                                                 \
                     int stride, int strideNum, int reducedStrideNum,                                       \
                     int blockSize, int blockNum)                                                           \
{                                                                                                           \
    __shared__ DTYPE iData[MAX_CUDA_THREAD_NUM_PER_BLOCK * MIN_CUDA_SHARED_MEM_COL_SIZE/2];                 \
                                                                                                            \
    int idx = threadIdx.x * blockDim.y + threadIdx.y;                                                       \
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;                                                   \
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;                                                   \
                                                                                                            \
    if(i >= stride * blockNum)                                                                              \
        return;                                                                                             \
                                                                                                            \
    __syncthreads();                                                                                        \
                                                                                                            \
    int k = i / stride;                                                                                     \
    int iOffset = i % stride;                                                                               \
                                                                                                            \
    DTYPE value = (i < stride * blockNum && j < strideNum) ?                                                \
                   input[blockSize * k + stride * j + iOffset] : initData;                                  \
                                                                                                            \
    /* load data into the shared mem */                                                                     \
    iData[threadIdx.x * blockDim.y + threadIdx.y] = value;                                                  \
                                                                                                            \
    __syncthreads();                                                                                        \
                                                                                                            \
    /* do reduction in shared mem */                                                                        \
    for (unsigned int s = blockDim.y/2; s > 0; s >>= 1){                                                    \
        if(threadIdx.y < s){                                                                                \
            iData[idx] = opName(iData[idx + s], iData[idx]);                                                \
        }                                                                                                   \
                                                                                                            \
        __syncthreads();                                                                                    \
    }                                                                                                       \
                                                                                                            \
    /* write result for this block to the output array */                                                   \
    if (threadIdx.y == 0 && blockIdx.y < reducedStrideNum)                                                  \
        output[(k * reducedStrideNum + blockIdx.y) * stride + iOffset] = iData[threadIdx.x * blockDim.y];   \
                                                                                                            \
}

KERNELREDUCEFUN3(KernelReduceMax, MAX, FLOAT_MIN)
KERNELREDUCEFUN3(KernelReduceMin, MIN, MAX_FLOAT)

/*
reduce a tensor to another that keeps the max value along a dimension  - slow version
Given a block of data, we go over each dimension i in the stride and we have
sum_i = max_{0<=j<strideNum} input_{i,j}
where we can view the block as a matrix and input_{i,j} represent the item at the
crossing of the i-th columne and the j-th row.
>> input - the input array (representing a tensor)
>> output - the sum over each block. NOTE: output is also an array
>> stride - stride that we need to move to the next item
>> strideNum - how many strides we need to finish the reduce
>> reducedStrideNum - the number of strides after reducation
>> blockSize - size of the block (i.e., stride * strideNum)
>> blockNum - how many blocks
*/
__global__
void KernelReduceMax(__half * input, __half * output,
                     int stride, int strideNum, int reducedStrideNum,
                     int blockSize, int blockNum)
{
    int idx = threadIdx.x * blockDim.y + threadIdx.y;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i >= stride * blockNum)
        return;

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    __shared__ __half iData[MAX_CUDA_THREAD_NUM_PER_BLOCK * MIN_CUDA_SHARED_MEM_COL_SIZE / 2];
#else
    __shared__ DTYPE iData[MAX_CUDA_THREAD_NUM_PER_BLOCK * MIN_CUDA_SHARED_MEM_COL_SIZE / 2];
#endif

    __syncthreads();

    int k = i / stride;
    int iOffset = i % stride;

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    __half value = (i < stride * blockNum && j < strideNum) ?
         input[blockSize * k + stride * j + iOffset] : __half(FLOAT16_MIN);
#else
    DTYPE value = (i < stride * blockNum && j < strideNum) ?
        __half2float(input[blockSize * k + stride * j + iOffset]) : FLOAT_MIN;
#endif

    /* load data into the shared mem */
    iData[threadIdx.x * blockDim.y + threadIdx.y] = value;

    __syncthreads();

    /* do reduction in shared mem */
    for (unsigned int s = blockDim.y / 2; s > 0; s >>= 1) {
        if (threadIdx.y < s && iData[idx] < iData[idx + s]) {
            iData[idx] = iData[idx + s];
        }

        __syncthreads();
    }

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    /* write result for this block to the output array */
    if (threadIdx.y == 0 && blockIdx.y < reducedStrideNum)
        output[(k * reducedStrideNum + blockIdx.y) * stride + iOffset] = iData[threadIdx.x * blockDim.y];
#else
    /* write result for this block to the output array */
    if (threadIdx.y == 0 && blockIdx.y < reducedStrideNum)
        output[(k * reducedStrideNum + blockIdx.y) * stride + iOffset] = __half(iData[threadIdx.x * blockDim.y]);
#endif

 }

/* 
reduce a tensor to another that keeps the max value along a dimension  - fast version
>> input - the input array (representing a tensor)
>> output - the sum over each block. NOTE: output is also an array
>> stride - stride that we need to move to the next item
>> strideNum - how many strides we need to finish the reduce
>> reducedStrideNum - the number of strides after reducation 
>> blockSize - size of the block (i.e., stride * strideNum)
>> blockNum - how many blocks
*/
#define KERNELREDUCEFUN4(funName, opName, opFuncName, initData)                                            \
template <unsigned int goodSize> __global__                                                                \
void funName(DTYPE * input, DTYPE * output,                                                    \
                         int stride, int strideNum, int reducedStrideNum,                                  \
                         int blockSize, int blockNum)                                                      \
{                                                                                                          \
    __shared__ DTYPE iData[MAX_CUDA_THREAD_NUM_PER_BLOCK];                                                 \
                                                                                                           \
    unsigned int tid = threadIdx.y;                                                                        \
    unsigned int j = blockIdx.y * (blockDim.y * 2) + threadIdx.y;                                          \
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;                                                \
                                                                                                           \
    if(i >= stride * blockNum)                                                                             \
        return;                                                                                            \
                                                                                                           \
    __syncthreads();                                                                                       \
                                                                                                           \
    /* first level reduction */                                                                            \
    int k = i / stride;                                                                                    \
    int iOffset = i % stride;                                                                              \
                                                                                                           \
    DTYPE * data = iData + threadIdx.x * blockDim.y;                                                       \
    DTYPE * inputData = input + k * blockSize;                                                             \
    DTYPE value = j < strideNum ? inputData[j * stride + iOffset] : initData;                              \
    DTYPE value2 = j + blockDim.y < strideNum ? inputData[(j + blockDim.y) * stride + iOffset]: initData;  \
                                                                                                           \
    value = opName(value, value2);                                                                         \
    value = opFuncName(value);                                                                             \
    if ((tid & 0x1f) == 0)                                                                                 \
        data[tid / 32] = value;                                                                            \
    __syncthreads();                                                                                       \
                                                                                                           \
    if (tid < 32) {                                                                                        \
        if (tid < blockDim.y / 32)                                                                         \
            value = data[tid];                                                                             \
        else                                                                                               \
            value = initData;                                                                              \
        value = opFuncName(value);                                                                         \
        if (tid == 0 && blockIdx.y < reducedStrideNum)                                                     \
            output[(k * reducedStrideNum + blockIdx.y) * stride + iOffset] = value;                        \
    }                                                                                                      \
}

KERNELREDUCEFUN4(KernelReduceMaxFast, MAX, shflDownReduceMax, FLOAT_MIN)
KERNELREDUCEFUN4(KernelReduceMinFast, MIN, shflDownReduceMin, MAX_FLOAT)

/*
reduce a tensor to another that keeps the max value along a dimension  - fast version
>> input - the input array (representing a tensor)
>> output - the sum over each block. NOTE: output is also an array
>> stride - stride that we need to move to the next item
>> strideNum - how many strides we need to finish the reduce
>> reducedStrideNum - the number of strides after reducation
>> blockSize - size of the block (i.e., stride * strideNum)
>> blockNum - how many blocks
*/
template <unsigned int goodSize> __global__
void KernelReduceMaxFast(__half * input, __half * output,
                         int stride, int strideNum, int reducedStrideNum,
                         int blockSize, int blockNum)
{
    unsigned int tid = threadIdx.y;
    unsigned int j = blockIdx.y * (blockDim.y * 2) + threadIdx.y;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= stride * blockNum)
        return;

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    __shared__ __half iData[MAX_CUDA_THREAD_NUM_PER_BLOCK];
#else
    __shared__ DTYPE iData[MAX_CUDA_THREAD_NUM_PER_BLOCK];
#endif

    __syncthreads();

    /* first level reduction */
    int k = i / stride;
    int iOffset = i % stride;

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    __half * data = iData + threadIdx.x * blockDim.y;
    __half * inputData = input + k * blockSize;
    __half value = j < strideNum ? inputData[j * stride + iOffset] : __half(FLOAT16_MIN);
    __half value2 = j + blockDim.y < strideNum ? inputData[(j + blockDim.y) * stride + iOffset] : __half(FLOAT16_MIN);
#else
    DTYPE * data = iData + threadIdx.x * blockDim.y;
    __half * inputData = input + k * blockSize;
    DTYPE value = j < strideNum ? __half2float(inputData[j * stride + iOffset]) : FLOAT_MIN;
    DTYPE value2 = j + blockDim.y < strideNum ? __half2float(inputData[(j + blockDim.y) * stride + iOffset]) : FLOAT_MIN;
#endif

    /* load data into the shared mem */
    data[tid] = MAX(value, value2);

    __syncthreads();

    /* unroll the warp */

    if (goodSize >= 512) { if (tid < 256) { if (data[tid] < data[tid + 256]) data[tid] = data[tid + 256]; } __syncthreads(); }
    if (goodSize >= 256) { if (tid < 128) { if (data[tid] < data[tid + 128]) data[tid] = data[tid + 128]; } __syncthreads(); }
    if (goodSize >= 128) { if (tid <  64) { if (data[tid] < data[tid +  64]) data[tid] = data[tid +  64]; } __syncthreads(); }
    if (goodSize >=  64) { if (tid <  32) { if (data[tid] < data[tid +  32]) data[tid] = data[tid +  32]; } __syncthreads(); }
    if (goodSize >=  32) { if (tid <  16) { if (data[tid] < data[tid +  16]) data[tid] = data[tid +  16]; } __syncthreads(); }
    if (goodSize >=  16) { if (tid <   8) { if (data[tid] < data[tid +   8]) data[tid] = data[tid +   8]; } __syncthreads(); }
    if (goodSize >=   8) { if (tid <   4) { if (data[tid] < data[tid +   4]) data[tid] = data[tid +   4]; } __syncthreads(); }
    if (goodSize >=   4) { if (tid <   2) { if (data[tid] < data[tid +   2]) data[tid] = data[tid +   2]; } __syncthreads(); }
    if (goodSize >=   2) { if (tid <   1) { if (data[tid] < data[tid +   1]) data[tid] = data[tid +   1]; } __syncthreads(); }

#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
    /* write result for this block to the output array */
    if (threadIdx.y == 0 && blockIdx.y < reducedStrideNum)
        output[(k * reducedStrideNum + blockIdx.y) * stride + iOffset] = data[0];
#else
    /* write result for this block to the output array */
    if (threadIdx.y == 0 && blockIdx.y < reducedStrideNum)
        output[(k * reducedStrideNum + blockIdx.y) * stride + iOffset] = __float2half(data[0]);
#endif
}

/*
reduce a tensor to another that keeps the max value along a dimension  - simple and fast version
*/
__global__
void KernelReduceMaxSimpleFast(DTYPE * input, DTYPE * output, 
                               int stride, int strideNum, int blockSize, int blockNum)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= stride)
        return;

    int blockIndex = i / blockSize;
    int offset = i % blockSize;

    DTYPE * ip = input + blockIndex * blockSize + offset;
    DTYPE * op = output + blockIndex * stride + offset;

    DTYPE max = DTYPE_MIN;
    if(strideNum % 4 == 0){
        int stride2 = stride + stride;
        int stride3 = stride2 + stride;
        int stride4 = stride3 + stride;
        for(int k = 0; k < blockSize; k += stride4){
            DTYPE m = MAX(MAX(ip[k], ip[k + stride]), MAX(ip[k + stride2], ip[k + stride3]));
            max = MAX(max, m);
        }
    }
    else{
        for (int k = 0; k < blockSize; k += stride)
            max = MAX(max, ip[k]);
    }

    __syncthreads();

    op[offset] = max;
}

/*
according the GPU's sm number allocation warp num
*/
inline void continuousStorageThreadAllocation(dim3& grid, dim3& block, long long vectorNum, int vectorSize)
{
    int warpNum = 4;
    if (vectorNum < 20 * 8){
        warpNum = 8;
        if (vectorNum < 20 * 4){
            warpNum = 16;
            if (warpNum < 20 * 2)
                warpNum = 32;
        }
    }
    int minWarpNum = vectorSize / 32;
    if (vectorSize % 32 != 0) minWarpNum++;
    warpNum = min(warpNum, minWarpNum);

    grid.x = (unsigned int)vectorNum;
    grid.y = 1;
    grid.z = 1;
    block.x = 1;
    block.y = warpNum * 32;
    block.z = 1;
}

/*
adjust threads.x number then we can use warp optimization 
*/
inline void adjustThreadForUseWarpOptimization(dim3& blocks, dim3& threads)
{
    if (threads.x > 1) {
        blocks.x *= threads.x;
        threads.x = 1;
    }
    if (threads.y < 32)
        threads.y = 32;
}

/*
In some case,we use less block to imporve efficiency
*/
#define KERNELREDUCEFUN2(funName, opName, opFuncName, initData)                   \
__global__                                                                        \
void funName(DTYPE * input, DTYPE * output, int strideNum, int blockNum)          \
{                                                                                 \
    int idx = threadIdx.x % 32;                                                   \
    int idy = (blockIdx.x * blockDim.x + threadIdx.x) / 32;                       \
                                                                                  \
    int startIndex = idy * strideNum;                                             \
    DTYPE threadMax = initData;                                                   \
    for (int i = idx; i < strideNum; i += 32) {                                   \
        threadMax = opName(input[startIndex + i], threadMax);                     \
    }                                                                             \
    threadMax = opFuncName(threadMax);                                            \
    if (idx == 0)                                                                 \
        output[idy] = threadMax;                                                  \
}

KERNELREDUCEFUN2(KernelReduceMaxOpLessBlocks, MAX, shflDownReduceMax, FLOAT_MIN)
KERNELREDUCEFUN2(KernelReduceMinOpLessBlocks, MIN, shflDownReduceMin, MAX_FLOAT)


/*
we use PTX code reduce
*/
#define KERNELREDUCEFUN1(funName, opName, opFuncName, initData)                          \
__global__                                                                               \
void funName(DTYPE * input, DTYPE * output,int stride, int strideNum,                    \
                       int reducedStrideNum,int blockSize, int blockNum)                 \
{                                                                                        \
    __shared__ DTYPE iData[MAX_CUDA_THREAD_NUM_PER_BLOCK / 32];                          \
                                                                                         \
    unsigned int tid = threadIdx.y;                                                      \
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;                              \
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;                              \
    if (i >= stride * blockNum)                                                          \
        return;                                                                          \
                                                                                         \
    /* first level reduction */                                                          \
    int k = i / stride;                                                                  \
    int iOffset = i % stride;                                                            \
                                                                                         \
    DTYPE threadMax = initData;                                                          \
                                                                                         \
    DTYPE * data = iData + threadIdx.x * blockDim.y;                                     \
    DTYPE * inputData = input + k * blockSize;                                           \
    for (int it = j; it < strideNum; it += blockDim.y){                                  \
        threadMax = opName(inputData[it * stride + iOffset], threadMax);                 \
    }                                                                                    \
                                                                                         \
    __syncthreads();                                                                     \
    threadMax = opFuncName(threadMax);                                                   \
    if ((tid & 0x1f) == 0)                                                               \
        data[tid / 32] = threadMax;                                                      \
                                                                                         \
    __syncthreads();                                                                     \
    /* use one warp to reduce remaining data */                                          \
    if (tid < 32){                                                                       \
        if (tid < blockDim.y / 32)                                                       \
            threadMax = data[tid];                                                       \
        else threadMax = initData;                                                       \
        threadMax = opFuncName(threadMax);                                               \
        if (tid == 0 && blockIdx.y < reducedStrideNum)                                   \
            output[(k * reducedStrideNum + blockIdx.y) * stride + iOffset] = threadMax;  \
    }                                                                                    \
}

KERNELREDUCEFUN1(KernelReduceMaxOp, MAX, shflDownReduceMax, FLOAT_MIN)
KERNELREDUCEFUN1(KernelReduceMinOp, MIN, shflDownReduceMin, MAX_FLOAT)

/* 
get the max-valued items along a dimension of the tensor (cuda version). 
For a 1-dimensional data array a,
sum_i = max_{0<=j<strideNum} input_{i,j}
>> input - the input tensor
>> output - the output tensor
>> dim - which dimension to reduce
*/
#define _CUDAREDUCE(_funcName, _reduceFunc1, _reduceFunc2, _reduceFunc3, _reduceFun4)                                                         \
void _funcName(const XTensor * input, XTensor * output, int dim)                                                                              \
{                                                                                                                                             \
    CheckNTErrors(input && output, "Empty input or output tensors!");                                                                         \
    CheckNTErrors(input->order == output->order + 1, "Incorrect tensor sizes!");                                                              \
    CheckNTErrors(input->order > dim && dim >=0, "Illegal dimension to reduce!");                                                             \
    CheckNTErrors(input->dataType == output->dataType, "Unmatched data types!");                                                              \
                                                                                                                                              \
    for(int i = 0; i < input->order; i++){                                                                                                    \
        if(i < dim){                                                                                                                          \
            CheckNTErrors(input->dimSize[i] == output->dimSize[i], "Unmatched tensors!");                                                     \
        }                                                                                                                                     \
        else if(i > dim){                                                                                                                     \
            CheckNTErrors(input->dimSize[i] == output->dimSize[i - 1], "Unmatched tensors!");                                                 \
        }                                                                                                                                     \
    }                                                                                                                                         \
                                                                                                                                              \
    int cudaGridSize[3];                                                                                                                      \
    int cudaBlockSize[3];                                                                                                                     \
    int iter = 0;                                                                                                                             \
    int stride = 1;                                                                                                                           \
    int strideNum = input->dimSize[dim];                                                                                                      \
    int blockSize = 1;                                                                                                                        \
    int blockNum = 1;                                                                                                                         \
                                                                                                                                              \
    for (int i = 0; i < input->order; i++) {                                                                                                  \
        if (i < dim)                                                                                                                          \
            blockNum *= input->dimSize[i];                                                                                                    \
        else if (i > dim)                                                                                                                     \
            stride *= input->dimSize[i];                                                                                                      \
    }                                                                                                                                         \
    blockSize = stride * strideNum;                                                                                                           \
                                                                                                                                              \
    int devID = input->devID;                                                                                                                 \
    XMem * mem = input->mem;                                                                                                                  \
                                                                                                                                              \
    GDevs.GetCudaThread2D(devID, strideNum, stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);                                         \
                                                                                                                                              \
    int bufSize = sizeof(DTYPE) * cudaGridSize[0] * stride * blockNum * 2;                                                                    \
    DTYPE * buf = mem != NULL ? (DTYPE*)mem->AllocBuf(mem->devID, bufSize) : (DTYPE*)XMemAlloc(input->devID, bufSize);                        \
    DTYPE * buf1 = buf;                                                                                                                       \
    DTYPE * buf2 = buf + cudaGridSize[0] * stride * blockNum;                                                                                 \
                                                                                                                                              \
    int devIDBackup;                                                                                                                          \
    ProtectCudaDev(input->devID, devIDBackup);                                                                                                \
                                                                                                                                              \
    if (stride == 1 && blockNum >= 10) {                                                                                                      \
        dim3 grids;                                                                                                                           \
        dim3 blocks;                                                                                                                          \
        continuousStorageThreadAllocation(grids, blocks, (long long)blockNum, strideNum);                                                     \
        if (blocks.y >= 128) {                                                                                                                \
            _reduceFunc1 <<<grids, blocks >>> ((DTYPE *)input->data, (DTYPE*)output->data, stride, strideNum, grids.y, blockSize, blockNum);  \
        }                                                                                                                                     \
        else {                                                                                                                                \
            if (blockNum % 4 != 0) blockNum = (int)(blockNum / 4) + 1;                                                                        \
            else blockNum = blockNum / 4;                                                                                                     \
            _reduceFunc2 <<<blockNum, 128 >>> ((DTYPE *)input->data, (DTYPE*)output->data, strideNum, blockNum);                              \
        }                                                                                                                                     \
    }                                                                                                                                         \
    else {                                                                                                                                    \
        do {                                                                                                                                  \
            if (input->dataType == DEFAULT_DTYPE) {                                                                                           \
                DTYPE * iData = NULL;                                                                                                         \
                DTYPE * oData = NULL;                                                                                                         \
                if (iter == 0) {                                                                                                              \
                    iData = (DTYPE*)input->data;                                                                                              \
                    oData = buf1;                                                                                                             \
                }                                                                                                                             \
                else if (iter % 2 == 1) {                                                                                                     \
                    iData = buf1;                                                                                                             \
                    oData = buf2;                                                                                                             \
                }                                                                                                                             \
                else {                                                                                                                        \
                    iData = buf2;                                                                                                             \
                    oData = buf1;                                                                                                             \
                }                                                                                                                             \
                                                                                                                                              \
                /* unroll the reduction procedure. The code is messy but it is faster. */                                                     \
                if (strideNum < 32) {                                                                                                         \
                    GDevs.GetCudaThread2D(devID, strideNum, stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);                         \
                    dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);                               \
                    if (cudaGridSize[0] == 1)                                                                                                 \
                        oData = (DTYPE*)output->data;                                                                                         \
                    _reduceFunc3 <<<blocks, threads>>> (iData, oData, stride, strideNum, blocks.y, blockSize, blockNum);                      \
                }                                                                                                                             \
                else if (strideNum < 128) {                                                                                                   \
                    GDevs.GetCudaThread2D(devID, MAX(strideNum / 2 + 1, 64), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);        \
                    dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);                               \
                    if (cudaGridSize[0] == 1)                                                                                                 \
                        oData = (DTYPE*)output->data;                                                                                         \
                    CheckNTErrors(cudaBlockSize[0] >= 64, "Incorrect thread number when calling the cuda kernel!");                           \
                    adjustThreadForUseWarpOptimization(blocks, threads);                                                                      \
                    _reduceFun4<64> <<<blocks, threads>>> (iData, oData, stride, strideNum, blocks.y, blockSize, blockNum);                   \
                }                                                                                                                             \
                else if (strideNum < 256) {                                                                                                   \
                    GDevs.GetCudaThread2D(devID, MAX(strideNum / 2 + 1, 128), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);       \
                    dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);                               \
                    if (cudaGridSize[0] == 1)                                                                                                 \
                        oData = (DTYPE*)output->data;                                                                                         \
                    CheckNTErrors(cudaBlockSize[0] >= 128, "Incorrect thread number when calling the cuda kernel!");                          \
                    adjustThreadForUseWarpOptimization(blocks, threads);                                                                      \
                    _reduceFun4<128> <<<blocks, threads>>> (iData, oData, stride, strideNum, blocks.y, blockSize, blockNum);                  \
                }                                                                                                                             \
                else if (strideNum < 512) {                                                                                                   \
                    GDevs.GetCudaThread2D(devID, MAX(strideNum / 2 + 1, 256), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);       \
                    dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);                               \
                    if (cudaGridSize[0] == 1)                                                                                                 \
                        oData = (DTYPE*)output->data;                                                                                         \
                    CheckNTErrors(cudaBlockSize[0] >= 256, "Incorrect thread number when calling the cuda kernel!");                          \
                    adjustThreadForUseWarpOptimization(blocks, threads);                                                                      \
                    _reduceFun4<256> <<<blocks, threads>>> (iData, oData, stride, strideNum, blocks.y, blockSize, blockNum);                  \
                }                                                                                                                             \
                else {                                                                                                                        \
                    GDevs.GetCudaThread2D(devID, MAX(strideNum / 2 + 1, 512), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);       \
                    dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);                               \
                    if (cudaGridSize[0] == 1)                                                                                                 \
                        oData = (DTYPE*)output->data;                                                                                         \
                    CheckNTErrors(cudaBlockSize[0] >= 512, "Incorrect thread number when calling the cuda kernel!");                          \
                    adjustThreadForUseWarpOptimization(blocks, threads);                                                                      \
                    _reduceFun4<512> <<<blocks, threads>>> (iData, oData, stride, strideNum, blocks.y, blockSize, blockNum);                  \
                }                                                                                                                             \
            }                                                                                                                                 \
            else if (input->dataType == X_FLOAT16) {                                                                                          \
                __half * buf1ft16 = (__half *)buf1;                                                                                           \
                __half * buf2ft16 = (__half *)buf2;                                                                                           \
                __half * iData = NULL;                                                                                                        \
                __half * oData = NULL;                                                                                                        \
                if (iter == 0) {                                                                                                              \
                    iData = (__half*)input->data;                                                                                             \
                    oData = buf1ft16;                                                                                                         \
                }                                                                                                                             \
                else if (iter % 2 == 1) {                                                                                                     \
                    iData = buf1ft16;                                                                                                         \
                    oData = buf2ft16;                                                                                                         \
                }                                                                                                                             \
                else {                                                                                                                        \
                    iData = buf2ft16;                                                                                                         \
                    oData = buf1ft16;                                                                                                         \
                }                                                                                                                             \
                                                                                                                                              \
                /* unroll the reduction procedure. The code is messy but it is faster. */                                                     \
                if (strideNum < 32) {                                                                                                         \
                    GDevs.GetCudaThread2D(devID, strideNum, stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);                         \
                    dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);                               \
                    if (cudaGridSize[0] == 1)                                                                                                 \
                        oData = (__half*)output->data;                                                                                        \
                    KernelReduceMax <<<blocks, threads>>> (iData, oData, stride, strideNum, blocks.y, blockSize, blockNum);                      \
                }                                                                                                                             \
                else if (strideNum < 128) {                                                                                                   \
                    GDevs.GetCudaThread2D(devID, MAX(strideNum / 2 + 1, 64), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);        \
                    dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);                               \
                    if (cudaGridSize[0] == 1)                                                                                                 \
                        oData = (__half*)output->data;                                                                                        \
                    CheckNTErrors(cudaBlockSize[0] >= 64, "Incorrect thread number when calling the cuda kernel!");                           \
                    KernelReduceMaxFast<64> <<<blocks, threads>>> (iData, oData, stride, strideNum, blocks.y, blockSize, blockNum);                   \
                }                                                                                                                             \
                else if (strideNum < 256) {                                                                                                   \
                    GDevs.GetCudaThread2D(devID, MAX(strideNum / 2 + 1, 128), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);       \
                    dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);                               \
                    if (cudaGridSize[0] == 1)                                                                                                 \
                        oData = (__half*)output->data;                                                                                        \
                    CheckNTErrors(cudaBlockSize[0] >= 128, "Incorrect thread number when calling the cuda kernel!");                          \
                    KernelReduceMaxFast<128> <<<blocks, threads>>> (iData, oData, stride, strideNum, blocks.y, blockSize, blockNum);                  \
                }                                                                                                                             \
                else if (strideNum < 512) {                                                                                                   \
                    GDevs.GetCudaThread2D(devID, MAX(strideNum / 2 + 1, 256), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);       \
                    dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);                               \
                    if (cudaGridSize[0] == 1)                                                                                                 \
                        oData = (__half*)output->data;                                                                                        \
                    CheckNTErrors(cudaBlockSize[0] >= 256, "Incorrect thread number when calling the cuda kernel!");                          \
                    KernelReduceMaxFast<256> <<<blocks, threads>>> (iData, oData, stride, strideNum, blocks.y, blockSize, blockNum);                  \
                }                                                                                                                             \
                else {                                                                                                                        \
                    GDevs.GetCudaThread2D(devID, MAX(strideNum / 2 + 1, 512), stride * blockNum, MAX_INT, cudaGridSize, cudaBlockSize);       \
                    dim3 blocks(cudaGridSize[1], cudaGridSize[0]), threads(cudaBlockSize[1], cudaBlockSize[0]);                               \
                    if (cudaGridSize[0] == 1)                                                                                                 \
                        oData = (__half*)output->data;                                                                                        \
                    CheckNTErrors(cudaBlockSize[0] >= 512, "Incorrect thread number when calling the cuda kernel!");                          \
                    KernelReduceMaxFast<512> <<<blocks, threads>>> (iData, oData, stride, strideNum, blocks.y, blockSize, blockNum);                  \
                }                                                                                                                             \
            }                                                                                                                                 \
                                                                                                                                              \
            strideNum = cudaGridSize[0];                                                                                                      \
            blockSize = cudaGridSize[0];                                                                                                      \
                                                                                                                                              \
            iter++;                                                                                                                           \
                                                                                                                                              \
        } while (strideNum > 1);                                                                                                              \
    }                                                                                                                                         \
                                                                                                                                              \
    BacktoCudaDev(input->devID, devIDBackup);                                                                                                 \
                                                                                                                                              \
    if (mem != NULL)                                                                                                                          \
        mem->ReleaseBuf(mem->devID, bufSize);                                                                                                 \
    else                                                                                                                                      \
        XMemFree(input->devID, buf);                                                                                                          \
}

_CUDAREDUCE(_CudaReduceMax, KernelReduceMaxOp, KernelReduceMaxOpLessBlocks, KernelReduceMax, KernelReduceMaxFast)
_CUDAREDUCE(_CudaReduceMin, KernelReduceMinOp, KernelReduceMinOpLessBlocks, KernelReduceMin, KernelReduceMinFast)


#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)