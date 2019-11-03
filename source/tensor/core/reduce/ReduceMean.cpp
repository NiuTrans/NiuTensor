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

#include "../math/ScaleAndShift.h"
#include "../../XName.h"
#include "ReduceSum.h"
#include "ReduceMean.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/* 
get the mean value along a dimension of the tensor

For a 1-dimensional data array a, mean = (1/n) * sum_i input_i

>> input - the input tensor
>> output - the output tensor
>> dim - the dimension where the reduction is performed on
*/
void _ReduceMean(const XTensor * input, XTensor * output, int dim)
{
    CheckNTErrors((input->order > dim), "Illegal dimension specified!");

    int num = input->dimSize[dim];

    _ReduceSum(input, output, dim);
    _ScaleAndShiftMe(output, (DTYPE)1/num, 0);
}

/* 
get the mean value along a dimension of the tensor (return an XTensor structure)
make a new tenosr to keep the result and return it

For a 1-dimensional data array a, mean = (1/n) * sum_i input_i

>> input - the input tensor
>> dim - the dimension where the reduction is performed on
<< return - the mean value along a dimension of the tensor
*/
XTensor ReduceMean(const XTensor &input, int dim)
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

    /* call _ReduceMean function */
    _ReduceMean(&input, &output, dim);
        
    /* tensor connection */
    if (input.enableGrad) {
        XLink::MakeLink(&input, NULL, &output, REDUCE_REDUCEMEAN);
        XLink::AddParamToHeadInt(&output, dim);
    }

    /* destroy variables */
    delete[] dimSize;

    return output;
}

/* 
get the mean value along a dimension of the tensor

For a 1-dimensional data array a, mean = (1/n) * sum_i input_i

>> input - the input tensor
>> output - the output tensor
>> dim - the dimension where the reduction is performed on
*/
void ReduceMean(const XTensor &input, XTensor &output, int dim)
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

    /* call _ReduceMean function */
    _ReduceMean(&input, &output, dim);

    if (input.enableGrad) {
        /* tensor connections */
        XLink::MakeLink(&input, NULL, &output, REDUCE_REDUCEMEAN);
        XLink::AddParamToHeadInt(&output, dim);
    }
}

} // namespace nts(NiuTrans.Tensor)