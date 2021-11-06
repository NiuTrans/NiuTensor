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

#include "Encoder.h"
#include "submodel/LayerNorm.h"
#include "submodel/CommonModules.h"
#include "../../tensor/core/CHeader.h"

/* the nmt namespace */
namespace nmt
{

/* set the training flag */
void AttEncoder::SetTrainingFlag(bool myIsTraining)
{
    isTraining = myIsTraining;

    /* initialize the stacked layers */
    embedder.SetTrainingFlag(myIsTraining);
    for (int i = 0; i < nlayer; i++) {
        if (ffns != NULL)
            ffns[i].SetTrainingFlag(myIsTraining);
        if (selfAtts != NULL)
            selfAtts[i].SetTrainingFlag(myIsTraining);
        if (attLayerNorms != NULL)
            attLayerNorms[i].SetTrainingFlag(myIsTraining);
        if (fnnLayerNorms != NULL)
            fnnLayerNorms[i].SetTrainingFlag(myIsTraining);
    }
    if (history != NULL)
        history->SetTrainingFlag(myIsTraining);
    if (encoderLayerNorm != NULL)
        encoderLayerNorm->SetTrainingFlag(myIsTraining);
}

/* constructor */
AttEncoder::AttEncoder()
{
    devID = -1;
    selfAtts = NULL;
    ffns = NULL;
    attLayerNorms = NULL;
    fnnLayerNorms = NULL;
    encoderLayerNorm = NULL;
    useHistory = false;
    history = NULL;
    dropoutP = 0.0;
    embDim = -1;
    finalNorm = false;
    ignored = -1;
    nlayer = -1;
    preLN = false;
    vSize = -1;
    isTraining = false;
}

/* de-constructor */
AttEncoder::~AttEncoder()
{
    delete[] selfAtts;
    delete[] ffns;
    delete[] attLayerNorms;
    delete[] fnnLayerNorms;
    if (finalNorm)
        delete encoderLayerNorm;
    if (useHistory)
        delete history;
}

/*
initialize the model
>> config - configurations for the model
*/
void AttEncoder::InitModel(NMTConfig& config)
{
    SetTrainingFlag(config.training.isTraining);
    devID = config.common.devID;
    preLN = config.model.encPreLN;
    dropoutP = config.model.dropout;
    embDim = config.model.encEmbDim;
    nlayer = config.model.encLayerNum;
    vSize = config.model.srcVocabSize;
    finalNorm = config.model.encFinalNorm;
    useHistory = config.model.useEncHistory;
    
    CheckNTErrors(vSize > 1, "Set vocabulary size by \"-vsize\"");
    CheckNTErrors(nlayer >= 1, "We have one encoding layer at least!");

    ffns = new FFN[nlayer];
    selfAtts = new Attention[nlayer];
    attLayerNorms = new LayerNorm[nlayer];
    fnnLayerNorms = new LayerNorm[nlayer];

    if (useHistory) {
        history = new LayerHistory;
        history->InitModel(config, true);
    }

    if (finalNorm) {
        encoderLayerNorm = new LayerNorm;
        encoderLayerNorm->InitModel(config, devID, embDim, config.model.encoderL1Norm);
    }

    /* initialize the stacked layers */
    embedder.InitModel(config);
    for (int i = 0; i < nlayer; i++) {
        ffns[i].InitModel(config, true);
        selfAtts[i].InitModel(config, true, true);
        attLayerNorms[i].InitModel(config, devID, embDim, config.model.encoderL1Norm);
        fnnLayerNorms[i].InitModel(config, devID, embDim, config.model.encoderL1Norm);
    }
}

/*
make the encoding network
>> input - the input tensor of the encoder
>> mask - the mask that indicate each position is valid
>> maskEncDec - a place-holder, not used
<< return - the output tensor of the encoder
*/
XTensor AttEncoder::Make(XTensor& input, XTensor* mask, XTensor& maskEncDec)
{
    /* clear the history */
    if (useHistory)
        history->ClearHistory();

    XTensor x;
    x = embedder.Make(input, false, 0);

    /* dropout */
    if (isTraining && dropoutP > 0)
        x = Dropout(x, dropoutP, /*inplace=*/true);

    if (useHistory)
        history->Add(x);

    for (int i = 0; i < nlayer; i++) {

        if (useHistory)
            x = history->Pop();

        XTensor att;
        XTensor fnn;
        XTensor res;
        XTensor attnBefore;
        XTensor attnAfter;
        XTensor fnnBefore;

        /* layer normalization with pre-norm for self-attn */
        attnBefore = LN(x, attLayerNorms[i], preLN, true, false);

        /* self attention */
        att = selfAtts[i].Make(attnBefore, attnBefore, attnBefore, mask, NULL, SELF_ATT);

        /* dropout */
        if (isTraining && dropoutP > 0)
            att = Dropout(att, dropoutP, /*inplace=*/true);

        /* residual connection */
        res = Sum(att, x, /*inplace=*/true);

        /* layer normalization with post-norm for self-attn */
        attnAfter = LN(res, attLayerNorms[i], preLN, false, true);

        /* layer normalization with pre-norm for fnn */
        fnnBefore = LN(attnAfter, fnnLayerNorms[i], preLN, true, false);

        /* fnn */
        fnn = ffns[i].Make(fnnBefore);

        /* dropout */
        if (isTraining && dropoutP > 0)
            fnn = Dropout(fnn, dropoutP, /*inplace=*/true);

        /* residual connection */
        res = Sum(fnn, attnAfter, /*inplace=*/true);

        /* layer normalization with post-norm for fnn */
        x = LN(res, fnnLayerNorms[i], preLN, false, true);

        if (useHistory)
            history->Add(x);
    }

    if (useHistory)
        x = history->Pop();

    /* clear the history while not training */
    if (useHistory && !isTraining)
        history->ClearHistory();

    if (finalNorm)
        return encoderLayerNorm->Run(x);

    return x;
}

/*
make the encoding network (wrapper)
>> input - the input tensor of the encoder
>> mask - the mask that indicate each position is valid
<< return - the output tensor of the encoder
*/
XTensor AttEncoder::Make(XTensor& input, XTensor* mask)
{
    XTensor nothing;

    return Make(input, mask, nothing);
}

/* 
run encoding for inference with pre-norm
>> input - the input tensor of the encoder
>> mask - the input mask on self-attention
<< return - the output tensor of the encoder
*/
XTensor AttEncoder::RunFastPreNorm(XTensor& input, XTensor* mask)
{
    /* clear the history */
    if (useHistory)
        history->ClearHistory();

    XTensor x;
    x = embedder.Make(input, false, 0);

    if (useHistory)
        history->Add(x);

    for (int i = 0; i < nlayer; i++) {

        XTensor xn;

        if (useHistory)
            x = history->Pop();

        /* layer normalization with pre-norm for self-attn */
        xn = attLayerNorms[i].Run(x);

        /* self attention */
        xn = selfAtts[i].Make(xn, xn, xn, mask, NULL, SELF_ATT);

        /* residual connection */
        SumMe(xn, x);

        /* layer normalization with pre-norm for ffn */
        x = fnnLayerNorms[i].Run(xn);

        /* ffn */
        x = ffns[i].Make(x);

        /* residual connection */
        SumMe(x, xn);

        if (useHistory)
            history->Add(x);
    }

    if (useHistory)
        x = history->Pop();

    if (finalNorm)
        return encoderLayerNorm->Run(x);

    return x;
}

/*
run encoding for inference with post-norm
>> input - the input tensor of the encoder
>> mask - the mask that indicate each position is valid
<< return - the output tensor of the encoder
*/
XTensor AttEncoder::RunFastPostNorm(XTensor& input, XTensor* mask)
{
    /* clear the history */
    if (useHistory)
        history->ClearHistory();

    XTensor x;
    x = embedder.Make(input, false, 0);

    if (useHistory)
        history->Add(x);

    for (int i = 0; i < nlayer; i++) {

        if (useHistory)
            x = history->Pop();

        XTensor selfAtt;

        /* self attention */
        selfAtt = selfAtts[i].Make(x, x, x, mask, NULL, SELF_ATT);

        /* residual connection */
        SumMe(selfAtt, x);

        /* layer normalization with post-norm for self-attn */
        selfAtt = attLayerNorms[i].Run(selfAtt);

        /* ffn */
        x = ffns[i].Make(selfAtt);

        /* residual connection */
        SumMe(x, selfAtt);

        /* layer normalization with post-norm for ffn */
        x = fnnLayerNorms[i].Run(x);

        if (useHistory)
            history->Add(x);
    }

    if (useHistory)
        x = history->Pop();

    /* clear the history while not training */
    if (useHistory && !isTraining)
        history->ClearHistory();

    if (finalNorm)
        return encoderLayerNorm->Run(x);

    return x;
}

} /* end of the nmt namespace */