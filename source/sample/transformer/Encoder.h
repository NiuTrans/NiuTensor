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

#ifndef __ENCODER_H__
#define __ENCODER_H__

#include "Config.h"
#include "submodel/FFN.h"
#include "submodel/Attention.h"
#include "submodel/Embedding.h"
#include "submodel/LayerNorm.h"
#include "submodel/LayerHistory.h"
#include "../../network/XNet.h"

using namespace nts;

/* the nmt namespace */
namespace nmt
{

/*
base class of the encoder
*/
class Encoder
{
public:
    virtual XTensor Make(XTensor& input, XTensor* mask, XTensor& mask2) = 0;
};

/*
the encoder based on self-attention
*/
class AttEncoder : Encoder
{
public:
    /* indicates whether train the model */
    bool isTraining;

    /* device id */
    int devID;

    /* layer number */
    int nlayer;

    /* embedding size */
    int embDim;

    /* vocabulary size */
    int vSize;

    /* dropout probability */
    DTYPE dropoutP;

    /* some positions can be ignored in attention. this is useful in lm where the first position needs
       special design for the attention model. */
    int ignored;

    /* embedding of word at each position */
    Embedder embedder;

    /* FNN model of each layer */
    FFN* ffns;

    /* attention model of each layer */
    Attention* selfAtts;

    /* layer normalizations for attention */
    LayerNorm* attLayerNorms;

    /* layer normalization for fnn */
    LayerNorm* fnnLayerNorms;

    /* layer normalization for encoder */
    LayerNorm* encoderLayerNorm;

    /* dynamic layer history */
    LayerHistory* history;

    /* if true, put layer normalization inside the residual blocks, 
       else put it between the residual blocks */
    bool preLN;

    /* add LN to the encoder output or not */
    bool finalNorm;

    /* reserve history for layers or not */
    bool useHistory;

public:
    /* set the training flag */
    void SetTrainingFlag(bool myIsTraining);

    /* constructor */
    AttEncoder();

    /* de-constructor */
    ~AttEncoder();

    /* initialize the model */
    void InitModel(NMTConfig& config);

    /* make the encoding network */
    XTensor Make(XTensor& input, XTensor* mask, XTensor& maskEncDec) override;

    /* make the encoding network (wrapper) */
    XTensor Make(XTensor& input, XTensor* mask);

    /* run encoding for inference with pre-norm */
    XTensor RunFastPreNorm(XTensor& input, XTensor* mask);

    /* run encoding for inference with post-norm */
    XTensor RunFastPostNorm(XTensor& input, XTensor* mask);
};

} /* end of the nmt namespace */

#endif
