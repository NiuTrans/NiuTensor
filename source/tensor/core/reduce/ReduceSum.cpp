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
#include "../../XName.h"

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
    CheckNTErrors((shift == NULL || XTensor::IsSameShaped(output, shift)), "Incorrect shift tensor size!");

	int dimRDI = input->order - dim - 1;
    CheckNTErrors(dimRDI >= 0, "Wrong dimension!");

    for(int i = 0; i < input->order; i++){
        if(i < dimRDI){
            CheckNTErrors((input->dimSizeRDI[i] == output->dimSizeRDI[i]), "Unmatched tensors!");
        }
        else if(i > dimRDI){
            CheckNTErrors((input->dimSizeRDI[i] == output->dimSizeRDI[i - 1]), "Unmatched tensors!");
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
        int strideNum = input->dimSizeRDI[dimRDI];
        int blockSize = 1;
        int blockNum = 1;
        for (int i = 0; i < input->order; i++) {
            if (i < dimRDI)
                stride *= input->dimSizeRDI[i];
            else if (i > dimRDI)
                blockNum *= input->dimSizeRDI[i];
        }
        blockSize = stride * strideNum;

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
    XLink::MakeLink(&input, &shift, &output, REDUCE_REDUCESUM);
    XLink::AddParamToHeadInt(&output, dim);
    XLink::AddParamToHead(&output, power);
    XLink::AddParamToHeadBool(&output, isExp);

    /* destroy variables */
    delete[] dimSize;

    return output;
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
    XLink::MakeLink(&input, NULL, &output, REDUCE_REDUCESUM);
    XLink::AddParamToHeadInt(&output, dim);
    XLink::AddParamToHead(&output, power);
    XLink::AddParamToHeadBool(&output, isExp);

    /* destroy variables */
    delete[] dimSize;

    return output;
}

} // namespace nts(NiuTrans.Tensor)
