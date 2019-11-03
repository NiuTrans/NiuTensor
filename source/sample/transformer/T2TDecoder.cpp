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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-10-09
 */

#include <math.h>
#include "T2TDecoder.h"
#include "T2TUtility.h"
#include "T2TLayerNormal.h"
#include "../../tensor/core/CHeader.h"

namespace transformer
{

/* constructor */
AttDecoder::AttDecoder()
{
    attentions = NULL;
    fnns = NULL;
    attLayerNorms = NULL;
    fnnLayerNorms = NULL;
    attentionsEnde = NULL;
    attEndeLayerNorms = NULL;
}

/* de-constructor */
AttDecoder::~AttDecoder()
{
    delete[] attentions;
    delete[] fnns;
    delete[] attLayerNorms;
    delete[] fnnLayerNorms;
    delete[] attentionsEnde;
    delete[] attEndeLayerNorms;
}

/* 
initialize the model 
>> argc - number of arguments
>> argv - list of pointers to the arguments
>> myIsMasked - indicates whether the masked attention is employed
>> myIgnored - number of positions ignored in attention (from the start)
>> myDevID - device id
*/
void AttDecoder::InitModel(int argc, char ** argv, 
                           bool myIsMasked, int myIgnored, 
                           int myDevID)
{
    //AttEncoder::InitModel(argc, argv, myIsMasked, myIgnored, myDevID);

    devID = myDevID;
    ignored = myIgnored;

    LoadParamInt(argc, argv, "nlayer", &nlayer, 6);
    LoadParamInt(argc, argv, "hsize", &hSize, DEFAULT_EMBEDDING_SIZE);
    LoadParamInt(argc, argv, "esize", &eSize, DEFAULT_EMBEDDING_SIZE);
    LoadParamInt(argc, argv, "vsizetgt", &vSize, -1);
    LoadParamFloat(argc, argv, "dropout", &dropoutP, 0);

    CheckNTErrors(nlayer >= 1, "We have one encoding layer at least!");
    CheckNTErrors(vSize > 1, "set vocabulary size by \"-vsizetgt\"");

    /* embedding model */
    embedder.InitModel(argc, argv, devID, false);

    attentions = new T2TAttention[nlayer];
    fnns = new T2TFNN[nlayer];
    attLayerNorms = new T2TLN[nlayer];
    fnnLayerNorms = new T2TLN[nlayer];
    attentionsEnde = new T2TAttention[nlayer];
    attEndeLayerNorms = new T2TLN[nlayer];

    /* initialize the stacked layers */
    for (int i = 0; i < nlayer; i++) {
        attentions[i].InitModel(argc, argv, myIsMasked, myIgnored, myDevID);
        fnns[i].InitModel(argc, argv, myDevID);
        attLayerNorms[i].InitModel(argc, argv, myDevID);
        fnnLayerNorms[i].InitModel(argc, argv, myDevID);
        attentionsEnde[i].InitModel(argc, argv, true, myIgnored, myDevID);
        attEndeLayerNorms[i].InitModel(argc, argv, myDevID);
    }
}

/* 
make the decoding network
>> inputDec - the input tensor of the decoder
>> outputEnc - the output tensor of the encoder
>> mask - mask that indicates which position is valid
>> maskEncDec - mask for the encoder-decoder attention
>> isTraining - indicates whether the model is used for training
<< return - the output tensor of the encoder
*/
XTensor AttDecoder::Make(XTensor &inputDec, XTensor &outputEnc, XTensor &mask, XTensor &maskEncDec, bool isTraining)
{
    XTensor x;

    x = embedder.Make(inputDec);

    /* dropout */
    if(isTraining && dropoutP > 0)
        x = Dropout(x, dropoutP);

    for(int i = 0; i < nlayer; i++){
        XTensor att;
        XTensor ende;
        XTensor ln;
        XTensor fnn;
        XTensor res;

        /******************/
        /* self attention */
        att = attentions[i].MakeBig(x, mask, isTraining);

        /* dropout */
        if(isTraining && dropoutP > 0)
            att = Dropout(att, dropoutP);

        /* residual connection */
        res = Sum(att, x);

        /* layer normalization */
        x = attLayerNorms[i].Make(res);

        /*****************************/
        /* encoder-decoder attention */
        ende = attentionsEnde[i].Make(outputEnc, x, outputEnc, maskEncDec, isTraining);

        /* dropout */
        if(isTraining && dropoutP > 0)
            ende = Dropout(ende, dropoutP);

        /* residual connection */
        res = Sum(ende, x);

        /* layer normalization */
        x = attEndeLayerNorms[i].Make(res);

        /*******/
        /* fnn */
        fnn = fnns[i].Make(x, isTraining);

        /* dropout */
        if(isTraining && dropoutP > 0)
            fnn = Dropout(fnn, dropoutP);

        /* residual connection */
        res = Sum(fnn, x);

        /* layer normalization */
        x = fnnLayerNorms[i].Make(res);
    }
    
    x.SetName(DECODING_NAME);

    return x;
}


}
