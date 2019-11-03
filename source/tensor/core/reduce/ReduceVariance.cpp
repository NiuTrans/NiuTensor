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

#include "../../XName.h"
#include "../math/ScaleAndShift.h"
#include "ReduceSum.h"
#include "ReduceVariance.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/* 
variance of the items along a dimension of the tensor

For a 1-dimensional data array a, variance = 1/n * \sum_i (a_i - mean)^2

>> input - the input tensor
>> output - the output tensor
>> dim - the dimension where the reduction is performed on
>> mean - the mean value
*/
void _ReduceVariance(const XTensor * input, XTensor * output, int dim, const XTensor * mean)
{
    int num = input->dimSize[dim];
    _ReduceSum(input, output, dim, mean, 2.0F);
    _ScaleAndShiftMe(output, (DTYPE)1 / num, 0);
}

/* 
variance of the items along a dimension of the tensor (return an XTensor structure)
make a new tensor to keep the result and return it

For a 1-dimensional data array a, variance = 1/n * \sum_i (a_i - mean)^2

>> input - the input tensor
>> dim - the dimension where the reduction is performed on
>> mean - the mean value
<< return - the variance of the items along a dimension of the tensor
*/
XTensor ReduceVariance(const XTensor &input, int dim, const XTensor &mean)
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

    /* call _ReduceVariance function */
    _ReduceVariance(&input, &output, dim, &mean);
                
    /* tensor connection */
    if (input.enableGrad) {
        XLink::MakeLink(&input, &mean, &output, REDUCE_REDUCEVARIANCE);
        XLink::AddParamToHeadInt(&output, dim);
    }

    /* destroy variables */
    delete[] dimSize;

    return output;
}

/* 
variance of the items along a dimension of the tensor

For a 1-dimensional data array a, variance = 1/n * \sum_i (a_i - mean)^2

>> input - the input tensor
>> output - the output tensor
>> dim - the dimension where the reduction is performed on
>> mean - the mean value
*/
void ReduceVariance(const XTensor &input, XTensor &output, int dim, const XTensor &mean)
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

    /* call _ReduceVariance function */
    _ReduceVariance(&input, &output, dim, &mean);

    if (input.enableGrad) {
        /* tensor connection */
        XLink::MakeLink(&input, &mean, &output, REDUCE_REDUCEVARIANCE);
        XLink::AddParamToHeadInt(&output, dim);
    }
}

} // namespace nts(NiuTrans.Tensor)
