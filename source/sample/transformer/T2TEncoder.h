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

#ifndef __T2TENCODER_H__
#define __T2TENCODER_H__

#include "T2TFNN.h"
#include "T2TAttention.h"
#include "T2TEmbedding.h"
#include "T2TLayerNormal.h"
#include "../../network/XNet.h"

using namespace nts;

namespace transformer
{
    
#define ENCODING_NAME "encoding"
#define ENCODING_INPUT_NAME "encoding_input"

/* 
base class of the encoder 
*/
class T2TEncoder
{
public:
    virtual
    XTensor Make(XTensor &input, XTensor &mask, XTensor &mask2, bool isTraining) = 0;
};

/* 
the encoder based on RNN 
*/
class RNNEncoder : T2TEncoder
{
public:
    XTensor Make(XTensor &input, XTensor &mask, XTensor &mask2, bool isTraining);
};


/* 
the encoder based on self-attention 
*/
class AttEncoder : T2TEncoder
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
       special design for the attention model. */
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
    
public:
    /* constructor */
    AttEncoder();

    /* de-constructor */
    ~AttEncoder();

    /* initialize the model */
    void InitModel(int argc, char ** argv, 
                   bool myIsMasked, int myIgnored, 
                   int myDevID = -1);

    /* make the encoding network */
    XTensor Make(XTensor &input, XTensor &mask, XTensor &maskEncDec, bool isTraining);

    /* make the encoding network (wrapper) */
    XTensor Make(XTensor &input, XTensor &mask, bool isTraining);
};


}

#endif
