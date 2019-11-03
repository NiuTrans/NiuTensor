/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2018, Natural Language Processing Lab, Northestern University.
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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-31
 */

#include <math.h>
#include "T2TLayerNormal.h"
#include "T2TUtility.h"
#include "T2TEmbedding.h"
#include "../../tensor/core/CHeader.h"

namespace transformer
{

/* constructor */
T2TLN::T2TLN()
{
    devID = -1;
    d = 0;
}

/* de-constructor */
T2TLN::~T2TLN()
{
}

/*
initialize the model
>> argc - number of arguments
>> argv - list of pointers to the arguments
>> myDevID - device id
*/
void T2TLN::InitModel(int argc, char ** argv, int myDevID)
{
    devID = myDevID;

    d = 0;
    LoadParamInt(argc, argv, "d", &d, DEFAULT_EMBEDDING_SIZE);

    InitTensor1D(&w, d, X_FLOAT, devID);
    InitTensor1D(&b, d, X_FLOAT, devID);

    w.SetDataRand(1.0F, 1.0F);
    b.SetZeroAll();
}

/*
make the network
for each layer representation x, we have
y =
>> input - the input tensor
>> return - layer normalization output
*/
XTensor T2TLN::Make(XTensor &input)
{
    XTensor &x = input;
    XTensor xn;
    XTensor mean;
    XTensor variance;
    XTensor standard;
    XTensor meanFilled;
    XTensor standardFilled;

    /* \mu = (sum_i x_i)/m */
    mean = ReduceMean(x, x.order - 1);

    /* \sigma = (sum_i (x_i - \mu)^2)/m */
    variance = ReduceVariance(x, x.order - 1, mean);

    /* standard = sqrt(variance) */
    standard = Power(variance, 0.5F);

    /* unsqueeze mean and standard deviation to fit them into
       the same shape of x */
    meanFilled = Unsqueeze(mean, x.order - 1, x.GetDim(-1));
    standardFilled = Unsqueeze(standard, x.order - 1, x.GetDim(-1));

    /* x' = (x - \mu)/standard */
    xn = (x - meanFilled) / standardFilled;

    /* result = x' * w + b   */
    return xn * w + b;
}

}
