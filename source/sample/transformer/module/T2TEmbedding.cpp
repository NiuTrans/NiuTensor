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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-08-01
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-07
 */

#include <cmath>

#include "T2TUtility.h"
#include "T2TEmbedding.h"
#include "../../../tensor/core/CHeader.h"

namespace transformer
{

/* constructor */
T2TEmbedder::T2TEmbedder()
{
    devID = -1;
    vSize = -1;
    maxLength = -1;
}

/* de-constructor */
T2TEmbedder::~T2TEmbedder()
{
}

/*
initialize the model
>> config - configurations of the model
>> isEnc - indicates if it is used for the encoder
*/
void T2TEmbedder::InitModel(T2TConfig& config, bool isEnc)
{
    devID = config.devID;
    d = config.modelSize;
    padIdx = config.padID;
    eSize = config.embSize;
    maxLength = config.maxPosLen;
    vSize = (isEnc) ? config.srcVocabSize : config.tgtVocabSize;

    InitTensor2D(&w, vSize, eSize, X_FLOAT, devID);

    maxLength = maxLength + 1 + 1;
    DTYPE v = 1.0F / (float)sqrt((float)eSize);
    w.SetDataRandn(0, v);

    /* create the positional embedding matrix */
    MakePosEmbedding(maxLength);
}

/*
make positional embeddings (of size eSize * length)
>> length - length of the sequence
*/
void T2TEmbedder::MakePosEmbedding(int length)
{
    InitTensor2D(&posEmbeddingBase, length, eSize, X_FLOAT, devID);

    float* data = new float[posEmbeddingBase.unitNum];

    for (int pos = 0; pos < length; pos++) {
        float* dp = data + pos * eSize;

        int channelSize = eSize / 2;
        int offset = 0;
        for (int i = 0; i < channelSize; i++) {
            dp[offset++] = (float)sin(pos * exp(-i * log(10000.0F) / (channelSize - 1)));
        }
        for (int i = 0; i < channelSize; i++) {
            dp[offset++] = (float)cos(pos * exp(-i * log(10000.0F) / (channelSize - 1)));
        }
    }

    /* padding zeros */
    int padStart = padIdx * eSize;
    for (int i = padStart; i < padStart + eSize; i++)
        data[i] = 0.F;

    posEmbeddingBase.SetData(data, posEmbeddingBase.unitNum);

    if (w.dataType != posEmbeddingBase.dataType)
        posEmbeddingBase = ConvertDataType(posEmbeddingBase, w.dataType);

    delete[] data;
}

/*
make the network
>> input - the word indices
>> nstep - the length of current sequence
>> isDec - indicates whether it is decoder
>> isTraining - indicates whether it is training
<< return - word & position embeddings of the input
*/
XTensor T2TEmbedder::Make(XTensor& input, bool isDec, bool isTraining, int nstep)
{
    /* make sure the padding index is 1 */
    CheckNTErrors(input.order > 1, "Wrong input tensor size!");
    CheckNTErrors(input.dimSize[input.order - 1] < maxLength, "The sequence is too long!");
    CheckNTErrors(vSize > 0, "set vocabulary size by \"-vsize\"");
    CheckNTErrors(eSize > 0, "set embedding size by \"-esize\"");

    XTensor wordEmbedding, position, posEmbedding;
    InitTensor(&position, &input);

    int* posData = new int[input.unitNum];

    XTensor inputCPU;
    InitTensorOnCPU(&inputCPU, &input);
    _CopyValues(&input, &inputCPU);

    if (!isDec)
    {
        /* encoder embeddings */
        for (int i = 0; i < inputCPU.dimSize[0]; i++) {
            int startNoPad = 1 + 1;
            int* p = ((int*)inputCPU.data) + i * inputCPU.dimSize[1];
            for (int j = 0; j < inputCPU.dimSize[1]; j++) {
                if (p[j] == 1) {
                    posData[i * inputCPU.dimSize[1] + j] = 1;
                }
                else {
                    posData[i * inputCPU.dimSize[1] + j] = startNoPad++;
                }
            }
        }
        position.SetData(posData, position.unitNum);
    }
    else
    {
        /* decoder embeddings */
        position.SetDataFixed(nstep + 2);
    }

    delete[] posData;

    /* we make positional embeddings first */
    posEmbedding = Gather(posEmbeddingBase, position);

    /* then we make word embeddings */
    wordEmbedding = Gather(w, input);

    wordEmbedding = Linear(wordEmbedding, (float)sqrt((float)eSize));

    /* we sum over the two embeddings */
    return wordEmbedding + posEmbedding;
}

}