/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2018, Natural Language Processing Lab, Northeastern University.
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

#include "Embedding.h"
#include "LayerNorm.h"
#include "../../../tensor/core/CHeader.h"

/* the nmt namespace */
namespace nmt
{

/* set the training flag */
void LayerNorm::SetTrainingFlag(bool myIsTraining)
{
    isTraining = myIsTraining;
}

/* constructor */
LayerNorm::LayerNorm()
{
    d = 0;
    devID = -1;
    isTraining = false;
    isL1Normed = false;
}

/* de-constructor */
LayerNorm::~LayerNorm()
{
}

/*
initialize the model
>> config - configuration of the model
>> myDevID - the device id
>> hiddenSize - the hidden size of layer normalization
>> myL1Normed - whether use L1-Norm
*/
void LayerNorm::InitModel(NMTConfig& config, int myDevID, int hiddenSize, bool myL1Normed)
{
    SetTrainingFlag(config.training.isTraining);
    d = hiddenSize;
    devID = myDevID;
    isL1Normed = myL1Normed;

    InitTensor1D(&weight, d, X_FLOAT, devID);
    InitTensor1D(&bias, d, X_FLOAT, devID);
    if (isTraining) {
        bias.SetZeroAll();
        weight.SetDataFixed(1);
    }
}

/*
initialize the model
>> myDevID - the device id
>> hiddenSize - the hidden size of layer normalization
>> myL1Normed - whether use L1-Norm
*/
XTensor LayerNorm::Run(XTensor& input)
{
    if (isL1Normed)
        return RunL1Norm(input);
    else
        return RunL2Norm(input);
}



/*
run layernorm for inference
>> input - the input tensor
>> return - layer normalization output
*/
XTensor LayerNorm::RunL2Norm(XTensor& input)
{
    XTensor& x = input;
    XTensor xn;
    XTensor mean;
    XTensor variance;
    XTensor standard;
    XTensor meanFilled;
    XTensor standardFilled;

    TENSOR_DATA_TYPE dataType = input.dataType;

    if (dataType == X_FLOAT16) {
        /* reduce functions can only run with FP32 */
        x = ConvertDataType(input, X_FLOAT);
    }

    /* \mu = (sum_i x_i)/m */
    mean = ReduceMean(x, x.order - 1);

    /* \sigma = (sum_i (x_i - \mu)^2)/m */
    variance = ReduceVariance(x, x.order - 1, mean, false);

    if (!weight.enableGrad)
        return Normalize(x, x.order - 1, mean, variance, weight, bias, 0.0F);

    /* standard = sqrt(variance) */
    standard = Power(variance, 0.5F);

    /* unsqueeze mean and standard deviation to fit them into
       the same shape of x */
    meanFilled = Unsqueeze(mean, x.order - 1, x.GetDim(-1));
    standardFilled = Unsqueeze(standard, x.order - 1, x.GetDim(-1));

    /* x' = (x - \mu)/standard */
    xn = (x - meanFilled) / standardFilled;

    if (dataType != mean.dataType) {
        x = ConvertDataType(x, dataType);
        xn = ConvertDataType(xn, dataType);
    }

    /* result = x' * w + b   */
    xn = xn * weight;

    xn = Sum(xn, bias, /*inplace=*/true);

    return xn;
}

/*
run layernorm-l1 for inference
>> input - the input tensor
>> return - layer normalization output
*/
XTensor LayerNorm::RunL1Norm(XTensor& input)
{
    XTensor& x = input;
    XTensor mean;
    XTensor variance;

    /* \mu = (sum_i x_i)/m */
    mean = ReduceMean(x, x.order - 1);

    /* \sigma = (sum_i |(x_i - \mu)|)/m */
    variance = ReduceVariance(x, x.order - 1, mean, true);

    return L1Normalize(x, x.order - 1, mean, variance, weight, bias);
}

} /* end of the nmt namespace */