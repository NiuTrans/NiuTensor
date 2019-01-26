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
#include "ReduceMax.h"
#include "ReduceMax.cuh"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/* 
get the max value of the items along a dimension of the tensor

>> input - the input tensor
>> output - the output tensor
>> dim - the dimension where the reduction is performed on
*/
void _ReduceMax(const XTensor * input, XTensor * output, int dim)
{
    CheckNTErrors((input->devID == output->devID || (input->devID < 0 && output->devID < 0)), 
                  "This code must be run on the same device!");
    CheckNTErrors((input && output), "Empty input or output tensors!");
    CheckNTErrors((input->order == output->order + 1), "Incorrect tensor sizes!");
    CheckNTErrors((input->order > dim && dim >=0), "Illegal dimension to reduce!");
    CheckNTErrors((input->dataType == output->dataType), "Unmatched data types!");
	
	int dimRDI = input->order - dim - 1;
    CheckNTErrors(dimRDI >= 0, "Wrong dimension!");

    for(int i = 0; i < input->order; i++){
        if(i < dimRDI){
            CheckNTErrors((input->dimSizeRDI[i] == output->dimSizeRDI[i]), 
                          "Unmatched tensors!");
        }
        else if(i > dimRDI){
            CheckNTErrors((input->dimSizeRDI[i] == output->dimSizeRDI[i - 1]), 
                          "Unmatched tensors!");
        }
    }

    if(input->devID >= 0){
#ifdef USE_CUDA
        _CudaReduceMax(input, output, dim);
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
            for(int i = 0; i < stride; i++){
                DTYPE max = FLOAT_MIN;
                DTYPE * ipe = ip + blockSize;
                for(DTYPE * ipb = ip + i; ipb < ipe; ipb += stride){
                    DTYPE v = *ipb;
                    if(max < v)
                        max = v;
                }
                *(op + i) = max;
            }
        }
    }
}

/* 
get the max value of the items along a dimension of the tensor (return an XTensor structure).
make a new tensor to keep the result and return it

>> input - the input tensor
>> dim - the dimension where the reduction is performed on
<< return - the max value of the items along a dimension of the tensor
*/
XTensor ReduceMax(const XTensor &input, int dim)
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

    /* call _ReduceMax function */
    _ReduceMax(&input, &output, dim);
    
    /* tensor connection */
    XLink::MakeLink(&input, NULL, &output, REDUCE_REDUCEMAX);
    XLink::AddParamToHeadInt(&output, dim);

    /* destroy variables */
    delete[] dimSize;

    return output;
}

} // namespace nts(NiuTrans.Tensor)
