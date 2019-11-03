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
#include "../../XTensor.h"
#include "../../XName.h"
#include "../shape/IsSameShaped.h"
#include "Normalize.h"
#include "Normalize.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
normalized the data with normal distribution

For an input x, y = a * (x-mean)/sqrt(variance+\epsilon) + b
where a and b are the scalar and bias respectively, and \epsilon is the adjustment parameter.

>> input - the input tensor
>> output - the output tensor
>> dim - dimension alone which we generate the mean and variance
>> mean - the mean of the input
>> var - the variance of the input
>> a - the scalar
>> b - the bias
>> epsilon - a parameter
*/
void _Normalize(const XTensor * input, XTensor * output, int dim, 
                const XTensor * mean, const XTensor * var, 
                const XTensor * a, const XTensor * b, DTYPE epsilon)
{
    CheckNTErrors((_IsSameShaped(input, output)), "Unmatched input tensors!");
    CheckNTErrors((_IsSameShaped(a, b)), "Unmatched input tensors");
    CheckNTErrors((_IsSameShaped(mean, var)), "Unmatched input tensors");
    CheckNTErrors((input && output && mean && var && a && b), "Empty input tensors!");
    CheckNTErrors((dim >= 0 && dim < input->order), "Incorrect reduction dimension!");
    CheckNTErrors((input->order == mean->order + 1), "Incorrect reduction dimension!");

    int stride = 1;
    int strideNum = input->dimSize[dim];
    int blockSize = 1;
    int blockNum = 1;
    for (int i = 0; i < input->order; i++) {
        if (i < dim) {
            CheckNTErrors((input->dimSize[i] == mean->dimSize[i]), "Wrong size!");
            blockNum *= input->dimSize[i];
        }
        else if (i > dim) {
            CheckNTErrors((input->dimSize[i] == mean->dimSize[i - 1]), "Wrong size!");
            stride *= input->dimSize[i];
        }
    }
    blockSize = stride * strideNum;

    if (input->devID >= 0 || output->devID >= 0) {
#ifdef USE_CUDA
        _CudaNormalize(input, output, dim, mean, var, a, b, epsilon);
#else
        ShowNTErrors("Please specify USE_CUDA and recompile the code!");
#endif
    }
    else {
        CheckNTErrors((input->dataType == DEFAULT_DTYPE), "TODO!");
        for (int k = 0; k < blockNum; k++) {
            DTYPE * ip = (DTYPE*)input->data + k * blockSize;
            DTYPE * op = (DTYPE*)output->data + k * blockSize;
            DTYPE * mp = (DTYPE*)mean->data + k * stride;
            DTYPE * vp = (DTYPE*)var->data + k * stride;
            DTYPE * ap = (DTYPE*)a->data;
            DTYPE * bp = (DTYPE*)b->data;
            for (int i = 0; i < strideNum; i++) {
                for (int j = 0; j < stride; j++) {
                    int offset = i * stride + j;
                    op[offset] = ap[offset] * (ip[offset] - mp[j]) / (DTYPE)sqrt(vp[j] + epsilon) + bp[offset];
                }
            }
        }
    }
}

/*
normalized the data with normal distribution (do it on site)
keep the result in the input tensor and return nothing

For an input x, x = a * (x-mean)/sqrt(variance+\epsilon) + b
where a and b are the scalar and bias respectively, and \epsilon is the adjustment parameter.

>> input - the input tensor
>> dim - dimension alone which we generate the mean and variance
>> mean - the mean of the input
>> var - the variance of the input
>> a - the scalar
>> b - the bias
>> epsilon - a parameter
*/
void _NormalizeMe(XTensor * input, int dim, 
                  const XTensor * mean, const XTensor * var, 
                  const XTensor * a, const XTensor * b, DTYPE epsilon)
{
    _Normalize(input, input, dim, mean, var, a, b, epsilon);
}

/*
normalized the data with normal distribution (do it on site)
keep the result in the input tensor and return nothing

For an input x, x = a * (x-mean)/sqrt(variance+\epsilon) + b
where a and b are the scalar and bias respectively, and \epsilon is the adjustment parameter.

>> input - the input tensor
>> dim - dimension alone which we generate the mean and variance
>> mean - the mean of the input
>> var - the variance of the input
>> a - the scalar
>> b - the bias
>> epsilon - a parameter
*/
void NormalizeMe(XTensor& input, int dim, 
                 const XTensor& mean, const XTensor& var, 
                 const XTensor& a, const XTensor& b, DTYPE epsilon)
{
    _Normalize(&input, &input, dim, &mean, &var, &a, &b, epsilon);
}

/*
normalized the data with normal distribution (return an XTensor structure)
make a new tensor to keep the result and return it 

For an input x, y = a * (x-mean)/sqrt(variance+\epsilon) + b
where a and b are the scalar and bias respectively, and \epsilon is the adjustment parameter.

>> input - the input tensor
>> dim - dimension alone which we generate the mean and variance
>> mean - the mean of the input
>> var - the variance of the input
>> a - the scalar
>> b - the bias
>> epsilon - a parameter
<< return - the result of normalized the data with normal distribution
*/
XTensor Normalize(const XTensor &input, int dim, 
                  const XTensor &mean, const XTensor &var, 
                  const XTensor &a, const XTensor &b, DTYPE epsilon)
{
    XTensor output(&input);
    output.SetTMPFlag();

    /* call _Normalize function */
    _Normalize(&input, &output, dim, &mean, &var, &a, &b, epsilon);

    /* tensor connections */
    TensorList list(5);
    list.Add((XTensor*)&input);
    list.Add((XTensor*)&mean);
    list.Add((XTensor*)&var);
    list.Add((XTensor*)&a);
    list.Add((XTensor*)&b);
    if (input.enableGrad) {
        XLink::MakeLink(&list, &output, MATH_NORMALIZE);
        XLink::AddParamToHeadInt(&output, dim);
        XLink::AddParamToHead(&output, epsilon);
    }

    return output;
}

/*
normalized the data with normal distribution (return an XTensor structure)
make a new tensor to keep the result and return it 

For an input x, y = a * (x-mean)/sqrt(variance+\epsilon) + b
where a and b are the scalar and bias respectively, and \epsilon is the adjustment parameter.

>> input - the input tensor
>> output - the output tensor
>> dim - dimension alone which we generate the mean and variance
>> mean - the mean of the input
>> var - the variance of the input
>> a - the scalar
>> b - the bias
>> epsilon - a parameter
<< return - the result of normalized the data with normal distribution
*/
void Normalize(const XTensor &input, XTensor &output, int dim, 
               const XTensor &mean, const XTensor &var, 
               const XTensor &a, const XTensor &b, DTYPE epsilon)
{
    if (!output.isInit || !IsSameShaped(input, output)) {
        InitTensorV2(&output, &input);
    }

    /* call _Normalize function */
    _Normalize(&input, &output, dim, &mean, &var, &a, &b, epsilon);

    if (input.enableGrad == true) {
        /* tensor connections */
        TensorList list(5);
        list.Add((XTensor*)&input);
        list.Add((XTensor*)&mean);
        list.Add((XTensor*)&var);
        list.Add((XTensor*)&a);
        list.Add((XTensor*)&b);
        XLink::MakeLink(&list, &output, MATH_NORMALIZE);
        XLink::AddParamToHeadInt(&output, dim);
        XLink::AddParamToHead(&output, epsilon);
    }
}


} // namespace nts(NiuTrans.Tensor)
