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
* $Created by: LI Yinqiao (li.yin.qiao.2012@hotmail.com) 2018-7-11
*/

#include "../../XTensor.h"
#include "ConvertDataType.h"
#include "ConvertDataType.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
convert data type
>> input - input tensor
>> output - output tensor
*/
void _ConvertDataType(const XTensor * input, XTensor * output)
{
    //CheckNTErrors((input->unitSize == output->unitSize), "Input and Output must be same in size!");

    if (input->dataType == output->dataType)
        return;
    
#ifdef USE_CUDA
    /* run it on GPUs */
    if (input->devID >= 0) {
        _CudaConvertDataType(input, output);
        return;
    }
#endif

    if (input->dataType == X_FLOAT && output->dataType == X_INT) {
        float * inputData = (float*)input->data;
        int * outputData = (int*)output->data;
        for (int i = 0; i < input->unitNum; i++) 
            outputData[i] = (int)inputData[i];
    }
    else if (input->dataType == X_INT && output->dataType == X_FLOAT) {
        int * inputData = (int*)input->data;
        float * outputData = (float*)output->data;
        for (int i = 0; i < input->unitNum; i++) 
            outputData[i] = (float)inputData[i];
    }
    else
        ShowNTErrors("Unsupported data types for conversion!");

}
} // namespace nts(NiuTrans.Tensor)