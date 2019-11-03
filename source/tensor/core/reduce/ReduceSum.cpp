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

#include <math.h>
#include "ReduceSum.h"
#include "ReduceSum.cuh"
#include "../shape/IsSameShaped.h"
#include "../../XName.h"
#include "../../XBLAS.h"
#include "VectorBuffer.h"
#include <iostream>

namespace nts{ // namespace nts(NiuTrans.Tensor)

/* 
sum the items along a dimension of the tensor

For a 1-dimensional data array a,
sum = \sum_i (a_i - shift)^power if isExp == false
sum = \sum_i exp((a_i - shift)^power) if isExp == true

>> input - the input tensor
>> output - the output tensor
>> dim - the dimension where the reduction is performed on
>> shift - shift the input
>> ieExp - specify if the exp() is performed
>> power - we perform pow(item_i, power) on each item in the array
*/
void _ReduceSum(const XTensor * input, XTensor * output, int dim, const XTensor * shift, DTYPE power, bool isExp)
{
    CheckNTErrors((input->devID == output->devID || (input->devID < 0 && output->devID < 0)), 
                  "This code must be run on the same device!");
    CheckNTErrors((input && output), "Empty input or output tensors!");
    CheckNTErrors((input->order == output->order + 1), "Incorrect tensor sizes!");
    CheckNTErrors((input->order > dim && dim >=0), "Illegal dimension to reduce!");
    CheckNTErrors((input->dataType == output->dataType), "Unmatched data types!");
    CheckNTErrors((shift == NULL || _IsSameShaped(output, shift)), "Incorrect shift tensor size!");

    CheckNTErrors(dim < input->order, "Wrong dimension!");

    for(int i = 0; i < input->order; i++){
        if(i < dim){
            CheckNTErrors((input->dimSize[i] == output->dimSize[i]), "Unmatched tensors!");
        }
        else if(i > dim){
            CheckNTErrors((input->dimSize[i] == output->dimSize[i - 1]), "Unmatched tensors!");
        }
    }

    if(input->devID >= 0){
#ifdef USE_CUDA
        _CudaReduceSum(input, output, dim, shift, power, isExp);
#endif
    }
    else{
        CheckNTErrors((input->dataType == DEFAULT_DTYPE), "TODO!");

        int stride = 1;
        int strideNum = input->dimSize[dim];
        int blockSize = 1;
        int blockNum = 1;
        for (int i = 0; i < input->order; i++) {
            if (i < dim)
                blockNum *= input->dimSize[i];
            else if (i > dim)
                stride *= input->dimSize[i];
        }
        blockSize = stride * strideNum;

        if(input->dimSize[input->order - 1] % (4 * 32 / sizeof(DTYPE)) == 0 && input->dimSize[input->order - 1] >= 32){
            int vecBufLength =  32 / sizeof(DTYPE);

            if(dim == input->order - 1){
                //data is contiguous in dim 0
                for(int i = 0; i < blockNum; i++){
                    // stride = 1
                    DTYPE * ip = (DTYPE*)input->data + blockSize * i;
                    DTYPE * op = (DTYPE*)output->data + i;
                    DTYPE * sp = shift != NULL ? (DTYPE*)shift->data + i : NULL;
                    DTYPE bias[32 / sizeof(DTYPE)] = {0};
                    if(shift != NULL){
                        for(int k = 0; k < 32 / sizeof(DTYPE); k++)
                            bias[k] = *(sp);
                    }
                    VectorBuffer vecBuf[4];
                    for(int j = 0; j < 4; j++){
                        vecBuf[j] = VectorBuffer::loadu((DTYPE*)(ip) + j * vecBufLength, isExp, power, bias);
                    }
                    for(int j = 1; j < strideNum / 32; j++){
                        const DTYPE* ptr = (DTYPE*)(ip + j * vecBufLength);
                        vecBuf[0] = vecBuf[0] + VectorBuffer::loadu(ptr + 0 * vecBufLength, isExp, power, bias);
                        vecBuf[1] = vecBuf[1] + VectorBuffer::loadu(ptr + 1 * vecBufLength, isExp, power, bias);
                        vecBuf[2] = vecBuf[2] + VectorBuffer::loadu(ptr + 2 * vecBufLength, isExp, power, bias);
                        vecBuf[3] = vecBuf[3] + VectorBuffer::loadu(ptr + 3 * vecBufLength, isExp, power, bias);
                    }
                    vecBuf[0] = ((vecBuf[0] + vecBuf[1]) + (vecBuf[2] + vecBuf[3]));
                    DTYPE sum = (DTYPE) 0.0;
                    for(int k = 0; k < vecBufLength; k++){
                        sum = sum + vecBuf[0][k];
                    }
                    *op = sum;
                }

            } else{
                //data is separated
                for(int i = 0; i < blockNum; i++){
                    for(int j = 0; j < input->dimSize[input->order - 1] / 32; j++){
                        DTYPE * ip = (DTYPE*)input->data + blockSize * i;
                        DTYPE * op = (DTYPE*)output->data + stride * i;
                        DTYPE * sp = shift != NULL ? (DTYPE*)shift->data + stride * i : NULL;
                        DTYPE bias[4 * 32 / sizeof(DTYPE)] = {0};
                        if(shift != NULL){
                            for(int k = 0; k < 4 * 32 / sizeof(DTYPE); k++)
                                bias[k] = *(sp + k);
                        }
                        VectorBuffer vecBuf[4];
                        for(int k = 0; k < 4; k++){
                            vecBuf[k] = VectorBuffer::loadu((DTYPE*)(ip) + (j * 4 + k) * 32 / sizeof(DTYPE), isExp, power, bias + j * 32 / sizeof(DTYPE));

                        }
                        for(int k = 1; k < strideNum; k++){
                            DTYPE * ptr = ip + k * stride + (j * 4) * vecBufLength;
                            vecBuf[0] = vecBuf[0] + VectorBuffer::loadu(ptr + 0 * vecBufLength, isExp, power, bias);
                            vecBuf[1] = vecBuf[1] + VectorBuffer::loadu(ptr + 1 * vecBufLength, isExp, power, bias + 1 * vecBufLength);
                            vecBuf[2] = vecBuf[2] + VectorBuffer::loadu(ptr + 2 * vecBufLength, isExp, power, bias + 2 * vecBufLength);
                            vecBuf[3] = vecBuf[3] + VectorBuffer::loadu(ptr + 3 * vecBufLength, isExp, power, bias + 3 * vecBufLength);
                        }
                        for(int k = 0; k < 4; k++){
                            for(int l = 0; l < vecBufLength; l++)
                                *(op + j * 32 + 8 * k + l) = vecBuf[k][l];
                        }
                    }
                }
            }
        }//run vector buffer
        else{

            for(int k = 0; k < blockNum; k++){
                DTYPE * ip = (DTYPE*)input->data + blockSize * k;
                DTYPE * op = (DTYPE*)output->data + stride * k;
                DTYPE * sp = shift != NULL ? (DTYPE*)shift->data + stride * k : NULL;
                for(int i = 0; i < stride; i++){
                    DTYPE sum = 0;
                    DTYPE bias = shift != NULL ? *(sp + i) : 0;
                    DTYPE * ipe = ip + blockSize;
                    if(isExp){
                        if(bias == 0){
                            if(power == (DTYPE)1.0){
                                for(DTYPE * ipb = ip + i; ipb < ipe; ipb += stride)
                                    sum += (DTYPE)exp(*ipb);
                            }
                            else if(power == (DTYPE)2.0){
                                for(DTYPE * ipb = ip + i; ipb < ipe; ipb += stride){
                                    DTYPE value = (*ipb);
                                    sum += (DTYPE)exp(value * value);
                                }
                            }
                            else if(power == (DTYPE)0.5){
                                for(DTYPE * ipb = ip + i; ipb < ipe; ipb += stride){
                                    DTYPE value = (*ipb);
                                    sum += (DTYPE)exp(sqrt(value));
                                }
                            }
                            else{
                                for(DTYPE * ipb = ip + i; ipb < ipe; ipb += stride){
                                    DTYPE value = (*ipb);
                                    sum += (DTYPE)exp(pow(value, power));
                                }
                            }
                        }
                        else{
                            if(power == (DTYPE)1.0){
                                for(DTYPE * ipb = ip + i; ipb < ipe; ipb += stride)
                                    sum += (DTYPE)exp(*ipb - bias);
                            }
                            else if(power == (DTYPE)2.0){
                                for(DTYPE * ipb = ip + i; ipb < ipe; ipb += stride){
                                    DTYPE value = (*ipb) - bias;
                                    sum += (DTYPE)exp(value * value);
                                }
                            }
                            else if(power == (DTYPE)0.5){
                                for(DTYPE * ipb = ip + i; ipb < ipe; ipb += stride){
                                    DTYPE value = (*ipb) - bias;
                                    sum += (DTYPE)exp(sqrt(value));
                                }
                            }
                            else{
                                for(DTYPE * ipb = ip + i; ipb < ipe; ipb += stride){
                                    DTYPE value = (*ipb) - bias;
                                    sum += (DTYPE)exp(pow(value, power));
                                }
                            }
                        }
                    }
                    else{
                        if(bias == 0){
                            if(power == (DTYPE)1.0){
                                    for(DTYPE * ipb = ip + i; ipb < ipe; ipb += stride)
                                        sum += *ipb;
                            }
                            else if(power == (DTYPE)2.0){
                                    for(DTYPE * ipb = ip + i; ipb < ipe; ipb += stride){
                                        DTYPE value = (*ipb);
                                        sum += value * value;
                                    }
                            }
                            else if(power == (DTYPE)0.5){
                                for(DTYPE * ipb = ip + i; ipb < ipe; ipb += stride){
                                    DTYPE value = (*ipb);
                                    sum += (DTYPE)sqrt(value);
                                }
                            }
                            else{
                                for(DTYPE * ipb = ip + i; ipb < ipe; ipb += stride){
                                    DTYPE value = (*ipb);
                                    sum += (DTYPE)pow(value, power);
                                }
                            }
                        }
                        else{
                            if(power == (DTYPE)1.0){
                                    for(DTYPE * ipb = ip + i; ipb < ipe; ipb += stride)
                                        sum += *ipb;
                                sum -= strideNum * bias;
                            }
                            else if(power == (DTYPE)2.0){
                                for(DTYPE * ipb = ip + i; ipb < ipe; ipb += stride){
                                    DTYPE value = (*ipb) - bias;
                                    sum += value * value;
                                }
                            }
                            else if(power == (DTYPE)0.5){
                                for(DTYPE * ipb = ip + i; ipb < ipe; ipb += stride){
                                    DTYPE value = (*ipb) - bias;
                                    sum += (DTYPE)sqrt(value);
                                }
                            }
                            else{
                                for(DTYPE * ipb = ip + i; ipb < ipe; ipb += stride){
                                    DTYPE value = (*ipb) - bias;
                                    sum += (DTYPE)pow(value, power);
                                }
                            }
                        }
                    }
                    *(op + i) = sum;
                }
            }
        }

    }
}

/* 
sum the items along a dimension of the tensor (return an XTensor structure)
make a new tensor to keep the result and return it

For a 1-dimensional data array a,
sum = \sum_i (a_i - shift)^power if isExp == false
sum = \sum_i exp((a_i - shift)^power) if isExp == true

>> input - the input tensor
>> dim - the dimension where the reduction is performed on
>> shift - shift the input
>> ieExp - specify if the exp() is performed
>> power - we perform pow(item_i, power) on each item in the array
<< return - the sum along a dimension of the tensor
*/
XTensor ReduceSum(const XTensor &input, int dim, const XTensor &shift, DTYPE power, bool isExp)
{
    CheckNTErrors(dim >= 0 && dim < input.order, "Illegal dimension to reduce!");
    
    int order = input.order - 1;
    int * dimSize = new int[order];
    for(int i = 0; i < order; i++){
        if(i < dim)
            dimSize[i] = input.dimSize[i];
        else if(i >= dim)
            dimSize[i] = input.dimSize[i + 1];
    }

    float dr = (!input.isSparse) ? 1.0F : input.denseRatio;
    XTensor output(order, dimSize, input.dataType, dr, input.devID, input.mem);
    output.SetTMPFlag();

    /* call _ReduceSum function */
    _ReduceSum(&input, &output, dim, &shift, power, isExp);
            
    /* tensor connection */
    if (input.enableGrad) {
        XLink::MakeLink(&input, &shift, &output, REDUCE_REDUCESUM);
        XLink::AddParamToHeadInt(&output, dim);
        XLink::AddParamToHead(&output, power);
        XLink::AddParamToHeadBool(&output, isExp);
    }

    /* destroy variables */
    delete[] dimSize;

    return output;
}

void ReduceSum(const XTensor &input, XTensor &output, int dim, const XTensor &shift, DTYPE power, bool isExp)
{
    CheckNTErrors(dim >= 0 && dim < input.order, "Illegal dimension to reduce!");

    if (!output.isInit || !XTensor::IsReduceShaped(&input, &output, dim)) {
        int order = input.order - 1;
        int * dimSize = new int[order];
        for (int i = 0; i < order; i++) {
            if (i < dim)
                dimSize[i] = input.dimSize[i];
            else if (i >= dim)
                dimSize[i] = input.dimSize[i + 1];
        }

        float dr = (!input.isSparse) ? 1.0F : input.denseRatio;
        InitTensorV2(&output, order, dimSize, input.dataType, dr, input.devID, input.mem);

        /* destroy variables */
        delete[] dimSize;
    }

    /* call _ReduceSum function */
    _ReduceSum(&input, &output, dim, &shift, power, isExp);

    if (input.enableGrad) {
        /* tensor connections */
        XLink::MakeLink(&input, &shift, &output, REDUCE_REDUCESUM);
        XLink::AddParamToHeadInt(&output, dim);
        XLink::AddParamToHead(&output, power);
        XLink::AddParamToHeadBool(&output, isExp);
    }
}

/* 
sum the items along a dimension of the tensor (return an XTensor structure)
make a new tensor to keep the result and return it

For a 1-dimensional data array a,
sum = \sum_i (a_i)^power if isExp == false
sum = \sum_i exp((a_i)^power) if isExp == true

>> input - the input tensor
>> dim - the dimension where the reduction is performed on
>> ieExp - specify if the exp() is performed
>> power - we perform pow(item_i, power) on each item in the array
<< return - the sum along a dimension of the tensor
*/
XTensor ReduceSum(const XTensor &input, int dim, DTYPE power, bool isExp)
{
    CheckNTErrors(dim >= 0 && dim < input.order, "Illegal dimension to reduce!");
    
    int order = input.order - 1;
    int * dimSize = new int[order];
    for(int i = 0; i < order; i++){
        if(i < dim)
            dimSize[i] = input.dimSize[i];
        else if(i >= dim)
            dimSize[i] = input.dimSize[i + 1];
    }

    float dr = (!input.isSparse) ? 1.0F : input.denseRatio;
    XTensor output(order, dimSize, input.dataType, dr, input.devID, input.mem);
    output.SetTMPFlag();

    /* call _ReduceSum function */
    _ReduceSum(&input, &output, dim, NULL, power, isExp);
            
    /* tensor connection */
    if (input.enableGrad) {
        XLink::MakeLink(&input, NULL, &output, REDUCE_REDUCESUM);
        XLink::AddParamToHeadInt(&output, dim);
        XLink::AddParamToHead(&output, power);
        XLink::AddParamToHeadBool(&output, isExp);
    }

    /* destroy variables */
    delete[] dimSize;

    return output;
}

/* 
sum the items along a dimension of the tensor

For a 1-dimensional data array a,
sum = \sum_i (a_i - shift)^power if isExp == false
sum = \sum_i exp((a_i - shift)^power) if isExp == true

>> input - the input tensor
>> output - the output tensor
>> dim - the dimension where the reduction is performed on
>> shift - shift the input
>> ieExp - specify if the exp() is performed
>> power - we perform pow(item_i, power) on each item in the array
*/
void ReduceSum(const XTensor &input, XTensor &output, int dim, DTYPE power, bool isExp)
{
    CheckNTErrors(dim >= 0 && dim < input.order, "Illegal dimension to reduce!");

    if (!output.isInit || !XTensor::IsReduceShaped(&input, &output, dim)) {
        int order = input.order - 1;
        int * dimSize = new int[order];
        for (int i = 0; i < order; i++) {
            if (i < dim)
                dimSize[i] = input.dimSize[i];
            else if (i >= dim)
                dimSize[i] = input.dimSize[i + 1];
        }

        float dr = (!input.isSparse) ? 1.0F : input.denseRatio;
        InitTensorV2(&output, order, dimSize, input.dataType, dr, input.devID, input.mem);

        /* destroy variables */
        delete[] dimSize;
    }

    /* call _ReduceSum function */
    _ReduceSum(&input, &output, dim, NULL, power, isExp);

    if (input.enableGrad) {
        /* tensor connections */
        XLink::MakeLink(&input, NULL, &output, REDUCE_REDUCESUM);
        XLink::AddParamToHeadInt(&output, dim);
        XLink::AddParamToHead(&output, power);
        XLink::AddParamToHeadBool(&output, isExp);
    }
}

} // namespace nts(NiuTrans.Tensor)
