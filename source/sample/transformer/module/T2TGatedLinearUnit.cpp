/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2020, Natural Language Processing Lab, Northeastern University.
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
 * $Created by: Bei Li (libei_neu@outlook.com) 2020-02-03
 */


#include <cmath>

#include "T2TUtility.h"
#include "T2TEmbedding.h"
#include "T2TGatedLinearUnit.h"
#include "../../../tensor/core/CHeader.h"
#include "../../../tensor/function/FHeader.h"

namespace transformer
{

/* constructor */
GLU::GLU()
{
    inSize = -1;
    outSize = -1;
    hSize = -1;
}

/* de-constructor */
GLU::~GLU()
{
}

/*
initialize the model
>> config - configurations of the model
*/
void GLU::InitModel(T2TConfig& config)
{
    devID = config.devID;

    float minmax = 0;

    inSize = config.modelSize;
    outSize = config.modelSize;

    InitTensor2D(&w1, hSize, outSize, X_FLOAT, devID);
    InitTensor1D(&b1, outSize, X_FLOAT, devID);

    InitTensor2D(&w2, hSize, outSize, X_FLOAT, devID);
    InitTensor1D(&b2, outSize, X_FLOAT, devID);
}

/*
make the network
y = W1 * x + b1 * sigmod(W2 * x + b2)
>> input - the input tensor, size = 2 * hSize
>> return - the output tensor, size = hSize
*/
XTensor GLU::Make(XTensor& input)
{
    XTensor t1;
    XTensor t2;
    TensorList input_list;

    /* split the input into two vectors with the dim hSize */
    Split(input, input_list, -1, 2);

    /* t1 = W1 * x + b1 */
    t1 = MulAndShift(input_list.GetItem(0), w1, b1);

    /* t2 = W2 * x + b2 */
    t2 = MulAndShift(input_list.GetItem(1), w2, b2);

    return t1 * Sigmoid(t2);
}

}