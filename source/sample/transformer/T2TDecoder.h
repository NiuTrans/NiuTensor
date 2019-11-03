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

#ifndef __T2TDECODER_H__
#define __T2TDECODER_H__

#include "T2TEncoder.h"

namespace transformer
{
    
#define DECODING_NAME "decoding"
#define DECODING_INPUT_NAME "decoding_input"

class AttDecoder
{
public:

    /* device id */
    int devID;

    /* layer number */
    int nlayer;

    /* hidden layer size of the FNN layer */
    int hSize;

    /* embedding size */
    int eSize;

    /* vocabulary size */
    int vSize;

    /* dropout probability */
    DTYPE dropoutP;

    /* some positions can be ignored in attention. this is useful in lm where the first position needs
 *     special design for the attention model. */
    int ignored;

    /* embedding of word at each position */
    T2TEmbedder embedder;

    /* FNN model of each layer */
    T2TFNN * fnns;

    /* attention model of each layer */
    T2TAttention * attentions;

    /* layer normalization for fnn */
    T2TLN * fnnLayerNorms;

    /* layer normalization for attention */
    T2TLN * attLayerNorms;

    /* input tensor of the encoder */
    XTensor * input;

    /* output tensor of the encoder */
    XTensor * output;

    /* encoder-decoder attention model of each layer */
    T2TAttention * attentionsEnde;

    /* layer normalization for encoder-decoder attention */
    T2TLN * attEndeLayerNorms;
public:
    /* constructor */
    AttDecoder();

    /* deconstructor */
    ~AttDecoder();

    /* initialize the model */
    void InitModel(int argc, char ** argv, 
                   bool myIsMasked, int myIgnored, 
                   int myDevID = -1);

    /* make the decoding network */
    XTensor Make(XTensor &inputDec, XTensor &outputEnc, XTensor &mask, XTensor &maskEncDec, bool isTraining);
};

}

#endif