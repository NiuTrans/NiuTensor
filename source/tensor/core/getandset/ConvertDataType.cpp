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
#include "../../XName.h"
#include "ConvertDataType.h"
#include "ConvertDataType.cuh"
#include "../movement/CopyValues.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* 
data type conversion
>> devID - device id
>> s - source data array
>> typeS - source data type
>> t - target data array
>> typeT - target data type
>> size - number of the items in s (and t)
*/
void ConvertDataType(int devID, 
                     void * s, TENSOR_DATA_TYPE typeS, 
                     void * t, TENSOR_DATA_TYPE typeT, 
                     int size)
{
    CheckNTErrors((devID < 0), "This code must be run on CPUs!");

    if(typeS == typeT)
        return;

    if(typeS == X_FLOAT && typeT == X_FLOAT16){
        for(int i = 0; i < size; i++){
            ((unsigned short*)t)[i] = FloatToFloat16(((float*)s)[i]);
        }
    }
    else if(typeS == X_FLOAT16 && typeT == X_FLOAT){
        for(int i = 0; i < size; i++){
            ((float*)t)[i] = Float16ToFloat(((unsigned short*)s)[i]);
        }
    }
    else{
        ShowNTErrors("Unsupported data types for conversion!");
    }
}

/*
convert data type

>> input - the input tensor
>> output - the output tensor
*/
void _ConvertDataType(const XTensor * input, XTensor * output)
{
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

/*
convert data type (return an XTensor structure) 
make a new tensor to keep the result and return it

>> input - the input tensor
<< return - the output tensor with the specified data type
*/
XTensor ConvertDataType(const XTensor & input, TENSOR_DATA_TYPE dataType)
{
    if (input.dataType == dataType) {
        XTensor output;
        output = CopyValues(input);

        return output;
    }

    int order = input.order;
    
    float dr = (!input.isSparse) ? 1.0F : input.denseRatio;
    XTensor output(order, input.dimSize, dataType, dr, input.devID, input.mem);
    output.SetTMPFlag();

    _ConvertDataType(&input, &output);

    /* tensor connection */
    if(input.enableGrad)
        XLink::MakeLink(&input, NULL, &output, GETANDSET_CONVERTDATATYPE);

    return output;
}

void ConvertDataType(const XTensor & input, XTensor & output, TENSOR_DATA_TYPE dataType)
{
    if (!output.isInit || input.dataType != output.dataType) {
        float dr = (!input.isSparse) ? 1.0F : input.denseRatio;
        InitTensorV2(&output, input.order, input.dimSize, dataType, dr, input.devID, input.mem);
    }

    _ConvertDataType(&input, &output);

    /* tensor connection */
    if (input.enableGrad)
        XLink::MakeLink(&input, NULL, &output, GETANDSET_CONVERTDATATYPE);
}

} // namespace nts(NiuTrans.Tensor)