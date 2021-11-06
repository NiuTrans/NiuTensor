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

/*
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-31
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04
 */

#include "Output.h"
#include "Embedding.h"
#include "../../../tensor/core/CHeader.h"

/* the nmt namespace */
namespace nmt
{

/* set the training flag */
void OutputLayer::SetTrainingFlag(bool myIsTraining)
{
    isTraining = myIsTraining;
}

/* constructor */
OutputLayer::OutputLayer()
{
    devID = -1;
    vSize = -1;
    hSize = -1;
    isTraining = false;
    shareDecInputOutputEmb = false;
}

/* de-constructor */
OutputLayer::~OutputLayer()
{
    if (!shareDecInputOutputEmb)
        DelTensor(w);
}

/*
initialize the model
>> config - configurations of the model
*/
void OutputLayer::InitModel(NMTConfig& config)
{
    SetTrainingFlag(config.training.isTraining);
    devID = config.common.devID;
    hSize = config.model.decEmbDim;
    vSize = config.model.tgtVocabSize;
    shareDecInputOutputEmb = config.model.shareDecInputOutputEmb;

    if (!shareDecInputOutputEmb) {
        w = NewTensor2D(vSize, hSize, X_FLOAT, devID);

        DTYPE v = 1.0F / (float)sqrt((float)hSize);
        if (isTraining) {
            w->SetDataRandn(0, v);
        }
    }
}

/*
make the network
>> input - the input tensor, (batch, srcLen, hiddenDim)
>> normalized - whether ignore the log-softmax
<< output - the output tensor, (batch, tgtLen, hiddenDim)
*/
XTensor OutputLayer::Make(XTensor& input, bool normalized)
{
    XTensor output;

    output = MMul(input, X_NOTRANS, *w, X_TRANS);

    /* use softmax for training */
    if (w->enableGrad)
        return Softmax(output, -1);

    /* normalize the output for beam search */
    if (normalized) {
        TENSOR_DATA_TYPE dataType = output.dataType;
        if (dataType == X_FLOAT16)
            output = ConvertDataType(output, X_FLOAT);

        output = LogSoftmax(output, -1);

        if (output.dataType != dataType)
            output = ConvertDataType(output, dataType);
    }
    return output;
}

} /* end of the nmt namespace */