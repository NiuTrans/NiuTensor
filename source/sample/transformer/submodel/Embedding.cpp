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
#include "../Config.h"
#include "../../../tensor/core/CHeader.h"

/* the nmt namespace */
namespace nmt
{

/* set the training flag */
void Embedder::SetTrainingFlag(bool myIsTraining)
{
    isTraining = myIsTraining;
}

/* constructor */
Embedder::Embedder()
{
    fp16 = false;
    w = NULL;
    shareEncDecEmb = false;
    devID = -1;
    vSize = -1;
    eSize = -1;
    padIdx = -1;
    maxLength = -1;
    isTraining = false;
}

/* de-constructor */
Embedder::~Embedder()
{
    if (w)
        DelTensor(w);
    w = NULL;
}

/*
initialize the model
>> config - configurations of the model
>> isEnc - indicates if it is a encoder module
*/
void Embedder::InitModel(NMTConfig& config, bool isEnc)
{
    SetTrainingFlag(config.training.isTraining);
    fp16 = config.common.useFP16;
    shareEncDecEmb = config.model.shareEncDecEmb;
    padIdx = config.model.pad;
    devID = config.common.devID;
    eSize = isEnc ? config.model.encEmbDim : config.model.decEmbDim;
    maxLength = config.model.maxTgtLen; // TODO: reset the maxLength for src emb
    vSize = isEnc ? config.model.srcVocabSize : config.model.tgtVocabSize;

    if (!w) {
        w = NewTensor2D(vSize, eSize, fp16 ? X_FLOAT16 : X_FLOAT, devID);

        maxLength = maxLength + 1 + 1;
        DTYPE v = 1.0F / (float)sqrt((float)eSize);

        if (isTraining) {
            w->SetDataRandn(0, v);
            for (int i = 0; i < eSize; i++) {
                w->Set2D(0.0F, padIdx, i);
            }
        }
    }

    /* create the positional embedding matrix */
    MakePosEmbedding(maxLength);
}

/*
make positional embeddings (of size eSize * length)
>> length - length of the sequence
*/
void Embedder::MakePosEmbedding(int length)
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

    delete[] data;
}

/*
make the network
>> input - the word indices
>> nstep - the length of current sequence
>> isDec - indicates whether it is decoder
<< return - word & position embeddings of the input
*/
XTensor Embedder::Make(XTensor& input, bool isDec, int nstep)
{
    /* make sure the padding index is 1 */
    CheckNTErrors(input.order > 1, "Wrong input tensor size!");
    CheckNTErrors(input.dimSize[input.order - 1] < maxLength, "The sequence is too long!");
    CheckNTErrors(vSize > 0, "Set vocabulary size by \"-vsize\"");
    CheckNTErrors(eSize > 0, "Set embedding size by \"-esize\"");

    XTensor wordEmbedding, position, posEmbedding;

    InitTensor1D(&position, input.GetDim(-1), X_INT, devID);

    if (!isDec || isTraining || input.GetDim(-1) > 1) {
        position.Range(0, position.unitNum, 1);
        ScaleAndShiftMe(position, 1.0F, float(padIdx + 1));
    }
    else {
        /* decoder embeddings during decoding */
        position.SetDataFixed(nstep + padIdx + 1);
    }

    /* we make positional embeddings first */
    XTensor embTMP;
    embTMP = Gather(posEmbeddingBase, position);
    posEmbedding = Unsqueeze(embTMP, 0, input.GetDim(0));

    /* then we make word embeddings */
    wordEmbedding = Gather(*w, input);

    if (isTraining)
        wordEmbedding = Linear(wordEmbedding, sqrtf((float)eSize), 0.0F, true);
    else
        ScaleMe(wordEmbedding, sqrtf((float)eSize));

    /* we sum over the two embeddings */
    SumMe(wordEmbedding, posEmbedding);

    return wordEmbedding;
}

} /* end of the nmt namespace */