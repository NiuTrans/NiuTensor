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
#include "T2TOutput.h"
#include "T2TUtility.h"
#include "T2TEmbedding.h"
#include "../../tensor/core/CHeader.h"

namespace transformer
{
/* constructor */
T2TOutput::T2TOutput()
{
    devID = -1;
    vSize = -1;
    inSize = -1;
    hSize = -1;
}

/* de-constructor */
T2TOutput::~T2TOutput()
{
}

/*
initialize the model 
>> argc - number of arguments
>> argv - list of pointers to the arguments
>> myDevID - device id
*/
void T2TOutput::InitModel(int argc, char ** argv, int myDevID)
{
    devID = myDevID;

    float minmax = 0;

    LoadParamInt(argc, argv, "vsizetgt", &vSize, -1);
    LoadParamInt(argc, argv, "d", &inSize, DEFAULT_EMBEDDING_SIZE);
    LoadParamInt(argc, argv, "d", &hSize, DEFAULT_EMBEDDING_SIZE);
    LoadParamFloat(argc, argv, "outputminmax", &minmax, 0.08F);

    InitTensor2D(&w, hSize, vSize, X_FLOAT, devID);
    
    float scale = 1.0F;
    float finfout = (float)sqrt(6.0F * scale/(hSize + vSize));
    w.SetDataRand(-finfout, finfout);

    DTYPE v = 1.0F/(float)sqrt((float)hSize);
    w.SetDataRandn(0, v);
}

/* 
make the network 
y = softmax(x * w)
>> input - input tensor
<< return - output tensor 
*/
XTensor T2TOutput::Make(XTensor &input)
{
    XTensor &x = input;

    return LogSoftmax(MMul(x, w), -1);
}

/* 
make the network (redefined output tensor) 
>> input - input tensor
>> output - output tensor 
*/
void T2TOutput::Make(XTensor &input, XTensor &output)
{
    XTensor &x = input;

    //output = LogSoftmax(MMul(x, w), -1);
    output = Softmax(MMul(x, w), -1);
    output.SetName(OUTPUT_NAME);
}

}
