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

#include "../../XTensor.h"
#include "../../XName.h"
#include "../../XBLAS.h"
#include "VectorBuffer.h"
#include "ReduceMax.h"
#include "ReduceMax.cuh"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/* 
get the max value of the items along a dimension of the tensor

>> input - the input tensor
>> output - the output tensor
>> dim - the dimension where the reduction is performed on
*/
#define _REDUCE_CPU_FUNCTION(_funcCPUName, _vectorOp, _reduceOp)                                                    \
void _funcCPUName(const XTensor * input, XTensor * output, int dim)                                                 \
{                                                                                                                   \
    CheckNTErrors((input->devID == output->devID || (input->devID < 0 && output->devID < 0)),                       \
        "This code must be run on the same device!");                                                               \
    CheckNTErrors((input && output), "Empty input or output tensors!");                                             \
    CheckNTErrors((input->order == output->order + 1), "Incorrect tensor sizes!");                                  \
    CheckNTErrors((input->order > dim && dim >= 0), "Illegal dimension to reduce!");                                \
    CheckNTErrors((input->dataType == output->dataType), "Unmatched data types!");                                  \
                                                                                                                    \
    CheckNTErrors(dim < input->order, "Wrong dimension!");                                                          \
                                                                                                                    \
    for (int i = 0; i < input->order; i++) {                                                                        \
                                                                                                                    \
            if (i < dim) {                                                                                          \
                                                                                                                    \
                    CheckNTErrors((input->dimSize[i] == output->dimSize[i]),                                        \
                        "Unmatched tensors!");                                                                      \
            }                                                                                                       \
            else if (i > dim) {                                                                                     \
                        CheckNTErrors((input->dimSize[i] == output->dimSize[i - 1]),                                \
                            "Unmatched tensors!");                                                                  \
                }                                                                                                   \
    }                                                                                                               \
    CheckNTErrors((input->dataType == DEFAULT_DTYPE), "TODO!");                                                     \
    int stride = 1;                                                                                                 \
    int strideNum = input->dimSize[dim];                                                                            \
    int blockSize = 1;                                                                                              \
    int blockNum = 1;                                                                                               \
    for (int i = 0; i < input->order; i++) {                                                                        \
        if (i > dim)                                                                                                \
            stride *= input->dimSize[i];                                                                            \
        else if (i < dim)                                                                                           \
            blockNum *= input->dimSize[i];                                                                          \
    }                                                                                                               \
    blockSize = stride * strideNum;                                                                                 \
                                                                                                                    \
                                                                                                                    \
    if(input->dimSize[input->order - 1] % (4 * 32 / sizeof(DTYPE)) == 0 && input->dimSize[input->order - 1] >= 32){ \
        int vecBufLength =  32 / sizeof(DTYPE);                                                                     \
                                                                                                                    \
        if (dim == input->order - 1) {                                                                              \
            /*data is contiguous in dim 0 */                                                                        \
            for (int i = 0; i < blockNum; i++) {                                                                    \
                DTYPE * ip = (DTYPE*)input->data + blockSize * i;                                                   \
                DTYPE * op = (DTYPE*)output->data + i;                                                              \
                VectorBuffer vecBuf[4];                                                                             \
                for (int j = 0; j < 4; j++) {                                                                       \
                    vecBuf[j] = VectorBuffer::loadu((DTYPE*)(ip)+j * vecBufLength);                                 \
                }                                                                                                   \
                for (int j = 1; j < strideNum / 32; j++) {                                                          \
                    const DTYPE* ptr = (DTYPE*)(ip + j * vecBufLength);                                             \
                    vecBuf[0] = vecBuf[0]._vectorOp(VectorBuffer::loadu(ptr + 0 * vecBufLength));                   \
                    vecBuf[1] = vecBuf[1]._vectorOp(VectorBuffer::loadu(ptr + 1 * vecBufLength));                   \
                    vecBuf[2] = vecBuf[2]._vectorOp(VectorBuffer::loadu(ptr + 2 * vecBufLength));                   \
                    vecBuf[3] = vecBuf[3]._vectorOp(VectorBuffer::loadu(ptr + 3 * vecBufLength));                   \
                }                                                                                                   \
                vecBuf[0] = vecBuf[0]._vectorOp(vecBuf[1]);                                                         \
                vecBuf[0] = vecBuf[0]._vectorOp(vecBuf[2]);                                                         \
                vecBuf[0] = vecBuf[0]._vectorOp(vecBuf[3]);                                                         \
                DTYPE maxN = vecBuf[0][0];                                                                          \
                for (int k = 1; k < vecBufLength; k++) {                                                            \
                    maxN = _reduceOp(maxN, vecBuf[0][k]);                                                           \
                }                                                                                                   \
                *op = maxN;                                                                                         \
            }                                                                                                       \
                                                                                                                    \
        }                                                                                                           \
        else {                                                                                                      \
            /* data is separated */                                                                                 \
            for(int i = 0; i < blockNum; i++){                                                                      \
                for(int j = 0; j < input->dimSize[input->order - 1] / 32; j++){                                     \
                    DTYPE * ip = (DTYPE*)input->data + blockSize * i;                                               \
                    DTYPE * op = (DTYPE*)output->data + stride * i;                                                 \
                    VectorBuffer vecBuf[4];                                                                         \
                    for(int k = 0; k < 4; k++){                                                                     \
                        vecBuf[k] = VectorBuffer::loadu((DTYPE*)(ip) + (j * 4 + k) * 32 / sizeof(DTYPE));           \
                                                                                                                    \
                    }                                                                                               \
                    for(int k = 1; k < strideNum; k++){                                                             \
                        DTYPE * ptr = ip + k * stride + (j * 4) * vecBufLength;                                     \
                        vecBuf[0] = vecBuf[0]._vectorOp(VectorBuffer::loadu(ptr + 0 * vecBufLength));               \
                        vecBuf[1] = vecBuf[1]._vectorOp(VectorBuffer::loadu(ptr + 1 * vecBufLength));               \
                        vecBuf[2] = vecBuf[2]._vectorOp(VectorBuffer::loadu(ptr + 2 * vecBufLength));               \
                        vecBuf[3] = vecBuf[3]._vectorOp(VectorBuffer::loadu(ptr + 3 * vecBufLength));               \
                    }                                                                                               \
                    for(int k = 0; k < 4; k++){                                                                     \
                        for(int l = 0; l < vecBufLength; l++)                                                       \
                            *(op + j * 32 + 8 * k + l) = vecBuf[k][l];                                              \
                    }                                                                                               \
                }                                                                                                   \
            }                                                                                                       \
        }                                                                                                           \
    }/* run vector buffer */                                                                                        \
    else{                                                                                                           \
        for(int k = 0; k < blockNum; k++){                                                                          \
            DTYPE * ip = (DTYPE*)input->data + blockSize * k;                                                       \
            DTYPE * op = (DTYPE*)output->data + stride * k;                                                         \
            for(int i = 0; i < stride; i++){                                                                        \
                DTYPE * ipe = ip + blockSize;                                                                       \
                DTYPE tmpData = *(ip + i);                                                                          \
                for(DTYPE * ipb = ip + i + stride; ipb < ipe; ipb += stride){                                       \
                    DTYPE v = *ipb;                                                                                 \
                    tmpData = _reduceOp(tmpData, v);                                                                \
                }                                                                                                   \
                *(op + i) = tmpData;                                                                                \
            }                                                                                                       \
        }                                                                                                           \
    }                                                                                                               \
}

_REDUCE_CPU_FUNCTION(reduceMaxCPU, maxData, MAX)
_REDUCE_CPU_FUNCTION(reduceMinCPU, minData, MIN)

#ifdef USE_CUDA            
#define _REDUCE_FUNCTION(_funcName, _cudaFuncName)                                                                   \
void _funcName(const XTensor * input, XTensor * output, int dim)                                                     \
{                                                                                                                    \
    if(input->devID >= 0){                                                                                           \
        _cudaFuncName(input, output, dim);                                                                           \
    }                                                                                                                \
    else{                                                                                                            \
        reduceMaxCPU(input, output, dim);                                                                            \
    }                                                                                                                \
}
_REDUCE_FUNCTION(_ReduceMax, _CudaReduceMax)
_REDUCE_FUNCTION(_ReduceMin, _CudaReduceMin)
#else
#define _REDUCE_FUNCTION(_funcName, reduceNameCPU)                                                                   \
void _funcName(const XTensor * input, XTensor * output, int dim)                                                     \
{                                                                                                                    \
    CheckNTErrors((input->devID < 0), "This code must be run on the CPU!");                                          \
    reduceNameCPU(input, output, dim);                                                                               \
}
    _REDUCE_FUNCTION(_ReduceMax, reduceMaxCPU)
    _REDUCE_FUNCTION(_ReduceMin, reduceMinCPU)
#endif 

/* 
get the max value of the items along a dimension of the tensor (return an XTensor structure).
make a new tensor to keep the result and return it

>> input - the input tensor
>> dim - the dimension where the reduction is performed on
<< return - the max value of the items along a dimension of the tensor
*/
#define REDUCE_FUNCTION(funcName, funcOp)                                                                           \
XTensor funcName(const XTensor & input, int dim)                                                                    \
{                                                                                                                   \
    CheckNTErrors(dim >= 0 && dim < input.order, "Illegal dimension to reduce!");                                   \
	                                                                                                                \
    int order = input.order - 1;                                                                                    \
    int * dimSize = new int[order];                                                                                 \
    for(int i = 0; i < order; i++){                                                                                 \
        if(i < dim)                                                                                                 \
            dimSize[i] = input.dimSize[i];                                                                          \
        else if(i >= dim)                                                                                           \
            dimSize[i] = input.dimSize[i + 1];                                                                      \
    }                                                                                                               \
                                                                                                                    \
    float dr = (!input.isSparse) ? 1.0F : input.denseRatio;                                                         \
    XTensor output(order, dimSize, input.dataType, dr, input.devID, input.mem);                                     \
    output.SetTMPFlag();                                                                                            \
                                                                                                                    \
    /* call _ReduceMax function */                                                                                  \
    funcOp(&input, &output, dim);                                                                                   \
                                                                                                                    \
    /* tensor connection */                                                                                         \
    if(input.enableGrad)                                                                                            \
    {                                                                                                               \
        XLink::MakeLink(&input, NULL, &output, REDUCE_REDUCEMAX);                                                   \
        XLink::AddParamToHeadInt(&output, dim);                                                                     \
    }                                                                                                               \
                                                                                                                    \
    /* destroy variables */                                                                                         \
    delete[] dimSize;                                                                                               \
                                                                                                                    \
    return output;                                                                                                  \
}

REDUCE_FUNCTION(ReduceMax, _ReduceMax)
REDUCE_FUNCTION(ReduceMin, _ReduceMin)

} // namespace nts(NiuTrans.Tensor)
