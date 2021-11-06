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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-10-09
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04
 */

#include "Config.h"
#include "Decoder.h"
#include "submodel/LayerNorm.h"
#include "submodel/CommonModules.h"
#include "../../tensor/core/CHeader.h"

/* the nmt namespace */
namespace nmt
{
/* set the training flag */
void AttDecoder::SetTrainingFlag(bool myIsTraining)
{
    isTraining = myIsTraining;

    /* disable caching during training */
    if (isTraining) {
        for (int i = 0; i < nlayer; i++) {
            if (selfAttCache != NULL)
                selfAttCache[i].enable = false;
            if (enDeAttCache != NULL)
                enDeAttCache[i].enable = false;
        }
    }

    for (int i = 0; i < nlayer; i++) {
        if (ffns != NULL)
            ffns[i].SetTrainingFlag(myIsTraining);
        if (selfAtts != NULL)
            selfAtts[i].SetTrainingFlag(myIsTraining);
        if (enDeAtts != NULL)
            enDeAtts[i].SetTrainingFlag(myIsTraining);
        if (ffnLayerNorms != NULL)
            ffnLayerNorms[i].SetTrainingFlag(myIsTraining);
        if (selfAttLayerNorms != NULL)
            selfAttLayerNorms[i].SetTrainingFlag(myIsTraining);
        if (enDeAttLayerNorms != NULL)
            enDeAttLayerNorms[i].SetTrainingFlag(myIsTraining);
    }
    if (embedder != NULL)
        embedder->SetTrainingFlag(myIsTraining);
    if (history != NULL)
        history->SetTrainingFlag(myIsTraining);
    if (decoderLayerNorm != NULL)
        decoderLayerNorm->SetTrainingFlag(myIsTraining);
}

/* constructor */
AttDecoder::AttDecoder()
{
    devID = -1;
    vSize = -1;
    embDim = -1;
    nlayer = -1;
    dropoutP = 0.0F;
    preLN = true;
    finalNorm = false;
    useHistory = false;
    isTraining = false;
    shareEncDecEmb = false;
    ffns = NULL;
    history = NULL;
    embedder = NULL;
    selfAtts = NULL;
    enDeAtts = NULL;
    selfAttCache = NULL;
    enDeAttCache = NULL;
    ffnLayerNorms = NULL;
    decoderLayerNorm = NULL;
    selfAttLayerNorms = NULL;
    enDeAttLayerNorms = NULL;    
}

/* de-constructor */
AttDecoder::~AttDecoder()
{
    delete[] selfAtts;
    delete[] enDeAtts;
    delete[] selfAttCache;
    delete[] enDeAttCache;
    delete[] ffnLayerNorms;
    delete[] enDeAttLayerNorms;
    delete[] selfAttLayerNorms;

    if (ffns != NULL)
        delete[] ffns;
    if (useHistory && history != NULL)
        delete history;
    if (!shareEncDecEmb && embedder != NULL)
        delete embedder;
    if (finalNorm && decoderLayerNorm != NULL)
        delete decoderLayerNorm;
}

/*
initialize the model
>> config - configurations of the model
*/
void AttDecoder::InitModel(NMTConfig& config)
{
    SetTrainingFlag(config.training.isTraining);
    devID = config.common.devID;
    preLN = config.model.decPreLN;
    dropoutP = config.model.dropout;
    embDim = config.model.decEmbDim;
    vSize = config.model.tgtVocabSize;
    nlayer = config.model.decLayerNum;
    finalNorm = config.model.decFinalNorm;
    useHistory = config.model.useDecHistory;
    shareEncDecEmb = config.model.shareEncDecEmb;

    CheckNTErrors(vSize > 1, "Set vocabulary size by \"-vsizetgt\"");
    CheckNTErrors(nlayer >= 1, "We have one encoding layer at least!");

    /* remove the FFN modules in some Transformer variants */
    if (config.model.decFFNHiddenDim > 0)
        ffns = new FFN[nlayer];
    selfAtts = new Attention[nlayer];
    enDeAtts = new Attention[nlayer];
    selfAttCache = new Cache[nlayer];
    enDeAttCache = new Cache[nlayer];
    ffnLayerNorms = new LayerNorm[nlayer];
    selfAttLayerNorms = new LayerNorm[nlayer];
    enDeAttLayerNorms = new LayerNorm[nlayer];

    if (useHistory) {
        history = new LayerHistory;
        history->InitModel(config, false);
    }
    if (!config.model.shareEncDecEmb) {
        embedder = new Embedder();
        embedder->InitModel(config, false);
    }
    if (finalNorm) {
        decoderLayerNorm = new LayerNorm;
        decoderLayerNorm->InitModel(config, devID, embDim, config.model.decoderL1Norm);
    }

    /* initialize the stacked layers */
    for (int i = 0; i < nlayer; i++) {
        if (ffns != NULL)
            ffns[i].InitModel(config, false);
        selfAtts[i].InitModel(config, false, true);
        enDeAtts[i].InitModel(config, false, false);
        ffnLayerNorms[i].InitModel(config, devID, embDim, config.model.decoderL1Norm);
        selfAttLayerNorms[i].InitModel(config, devID, embDim, config.model.decoderL1Norm);
        enDeAttLayerNorms[i].InitModel(config, devID, embDim, config.model.decoderL1Norm);
    }
}

/*
make the decoding network
>> inputDec - the input tensor of the decoder
>> outputEnc - the output tensor of the encoder
>> mask - mask that indicates which position is valid
>> maskEncDec - mask for the encoder-decoder attention
>> nstep - the current length of the decoder input
<< return - the output tensor of the decoder
*/
XTensor AttDecoder::Make(XTensor& inputDec, XTensor& outputEnc, 
                         XTensor* mask, XTensor* maskEncDec, int nstep)
{
    /* clear the history */
    if (useHistory)
        history->ClearHistory();

    XTensor x;
    x = embedder->Make(inputDec, true, nstep);

    /* dropout */
    if (isTraining && dropoutP > 0)
        x = Dropout(x, dropoutP, /*inplace=*/true);

    if (useHistory)
        history->Add(x);

    for (int i = 0; i < nlayer; i++) {

        if (useHistory)
            x = history->Pop();

        XTensor att;
        XTensor ffn;
        XTensor res;
        XTensor ende;
        XTensor ffnBefore;
        XTensor selfAttnBefore;
        XTensor selfAttnAfter;
        XTensor endeAttnBefore;
        XTensor endeAttnAfter;

        /* layer normalization with pre-norm for self-attn */
        selfAttnBefore = LN(x, selfAttLayerNorms[i], preLN, true, false);

        /******************/
        /* self attention */
        att = selfAtts[i].Make(selfAttnBefore, selfAttnBefore, selfAttnBefore, 
                               mask, &selfAttCache[i], SELF_ATT);

        /* dropout */
        if (isTraining && dropoutP > 0)
            att = Dropout(att, dropoutP, /*inplace=*/true);

        /* residual connection */
        res = Sum(att, x, /*inplace=*/true);

        /* layer normalization with post-norm for self-attention */
        selfAttnAfter = LN(res, selfAttLayerNorms[i], preLN, false, true);

        /* layer normalization with pre-norm for encoder-decoder attention */
        endeAttnBefore = LN(selfAttnAfter, enDeAttLayerNorms[i], preLN, true, false);

        /* encoder-decoder attention */
        ende = enDeAtts[i].Make(outputEnc, endeAttnBefore, outputEnc, maskEncDec, 
                                &enDeAttCache[i], EN_DE_ATT);

        /* dropout */
        if (isTraining && dropoutP > 0)
            ende = Dropout(ende, dropoutP, /*inplace=*/true);

        /* residual connection */
        res = Sum(ende, selfAttnAfter, /*inplace=*/true);

        /* layer normalization with post-norm for encoder-decoder attention */
        endeAttnAfter = LN(res, enDeAttLayerNorms[i], preLN, false, true);

        /* layer normalization with pre-norm for ffn */
        ffnBefore = LN(endeAttnAfter, ffnLayerNorms[i], preLN, true, false);

        /* ffn */
        ffn = ffns[i].Make(ffnBefore);

        /* dropout */
        if (isTraining && dropoutP > 0)
            ffn = Dropout(ffn, dropoutP, /*inplace=*/true);

        /* residual connection */
        res = Sum(ffn, endeAttnAfter, /*inplace=*/true);

        /* layer normalization with post-norm for ffn */
        x = LN(res, ffnLayerNorms[i], preLN, false, true);

        if (useHistory)
            history->Add(x);
    }

    if (useHistory)
        x = history->Pop();

    /* clear the history while not training */
    if (useHistory && !isTraining)
        history->ClearHistory();

    if (finalNorm)
        return decoderLayerNorm->Run(x);

    return x;
}

/*
run decoding for inference with pre-norm
>> inputDec - the input tensor of the decoder
>> outputEnc - the output tensor of the encoder
>> mask - mask that indicates which position is valid
>> maskEncDec - mask for the encoder-decoder attention
>> nstep - the current length of the decoder input
<< return - the output tensor of the decoder
*/
XTensor AttDecoder::RunFastPreNorm(XTensor& inputDec, XTensor& outputEnc, XTensor* maskEncDec, int nstep)
{
    /* clear the history */
    if (useHistory)
        history->ClearHistory();

    XTensor x;

    x = embedder->Make(inputDec, true, nstep);

    if (useHistory)
        history->Add(x);

    for (int i = 0; i < nlayer; i++) {

        if (useHistory)
            x = history->Pop();

        XTensor xn;

        /* layer normalization with pre-norm for self-attn */
        xn = selfAttLayerNorms[i].Run(x);

        /* self attention */
        xn = selfAtts[i].Make(xn, xn, xn, NULL, &selfAttCache[i], SELF_ATT);

        /* residual connection */
        SumMe(xn, x);

        /* layer normalization with pre-norm for encoder-decoder attention */
        x = enDeAttLayerNorms[i].Run(xn);

        /* encoder-decoder attention */
        x = enDeAtts[i].Make(outputEnc, x, outputEnc, maskEncDec,
                             &enDeAttCache[i], EN_DE_ATT);

        /* residual connection */
        SumMe(x, xn);

        /* layer normalization with pre-norm for ffn */
        xn = ffnLayerNorms[i].Run(x);

        /* ffn */
        if (ffns != NULL)
            xn = ffns[i].Make(xn);

        /* residual connection */
        SumMe(x, xn);

        if (useHistory)
            history->Add(x);
    }

    if (useHistory)
        x = history->Pop();

    if (finalNorm)
        return decoderLayerNorm->Run(x);

    return x;
}

/*
run decoding for inference with post-norm
>> inputDec - the input tensor of the decoder
>> outputEnc - the output tensor of the encoder
>> mask - mask that indicates which position is valid
>> maskEncDec - mask for the encoder-decoder attention
>> nstep - the current length of the decoder input
<< return - the output tensor of the decoder
*/
XTensor AttDecoder::RunFastPostNorm(XTensor& inputDec, XTensor& outputEnc, XTensor* maskEncDec, int nstep)
{
    /* clear the history */
    if (useHistory)
        history->ClearHistory();

    XTensor x;

    x = embedder->Make(inputDec, true, nstep);

    if (useHistory)
        history->Add(x);

    for (int i = 0; i < nlayer; i++) {
        XTensor xn;

        if (useHistory)
            x = history->Pop();

        /******************/
        /* self attention */
        xn = selfAtts[i].Make(x, x, x, NULL, &selfAttCache[i], SELF_ATT);

        /* residual connection */
        SumMe(xn, x);

        /* layer normalization with post-norm for self-attn */
        xn = selfAttLayerNorms[i].Run(xn);

        /* encoder-decoder attention */
        x = enDeAtts[i].Make(outputEnc, xn, outputEnc, maskEncDec,
                             &enDeAttCache[i], EN_DE_ATT);

        /* residual connection */
        SumMe(x, xn);

        /* layer normalization with pre-norm for ffn */
        xn = enDeAttLayerNorms[i].Run(x);

        /* ffn */
        if (ffns != NULL)
            x = ffns[i].Make(xn);

        /* residual connection */
        SumMe(x, xn);

        x = ffnLayerNorms->Run(x);

        if (useHistory)
            history->Add(x);
    }

    if (useHistory)
        x = history->Pop();

    /* clear the history while not training */
    if (useHistory && !isTraining)
        history->ClearHistory();

    if (finalNorm)
        return decoderLayerNorm->Run(x);

    return x;
}

}