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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-31
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04
 */

#include <cmath>

#include "T2TOutput.h"
#include "T2TUtility.h"
#include "T2TEmbedding.h"
#include "../../../tensor/core/CHeader.h"

namespace transformer
{

/* constructor */
T2TOutput::T2TOutput()
{
    devID = -1;
    vSize = -1;
    hSize = -1;
}

/* de-constructor */
T2TOutput::~T2TOutput()
{
}

/*
initialize the model
>> config - configurations of the model
*/
void T2TOutput::InitModel(T2TConfig& config)
{
    devID = config.devID;
    hSize = config.modelSize;
    vSize = config.tgtVocabSize;

    InitTensor2D(&w, vSize, hSize, X_FLOAT, devID);

    DTYPE v = 1.0F / (float)sqrt((float)hSize);
    w.SetDataRandn(0, v);
}

/*
make the network (redefined output tensor)
>> input - input tensor
>> output - output tensor
>> isTraining - whether it is used for training
>> normalized - whether ignore the log-softmax
*/
void T2TOutput::Make(XTensor& input, XTensor& output, bool isTraining, bool normalized)
{
    XTensor& x = input;

    output = MMul(x, X_NOTRANS, w, X_TRANS);

    /* use softmax for training */
    if (isTraining) {
        output = Softmax(output, -1);
        return;
    }

    /* normalize the output for beam search */
    if (normalized) {
        auto dataType = output.dataType;
        if (dataType == X_FLOAT16)
            output = ConvertDataType(output, X_FLOAT);

        output = LogSoftmax(output, -1);

        if (output.dataType != dataType)
            output = ConvertDataType(output, dataType);
    }
}

}