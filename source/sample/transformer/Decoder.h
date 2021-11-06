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

#ifndef __DECODER_H__
#define __DECODER_H__

#include "Config.h"
#include "Encoder.h"

 /* end of the nmt namespace */
namespace nmt
{
/* todo: refactor the type of embedder and its weight */
class AttDecoder
{
public:
    /* indicates whether share encoder and decoder embeddings */
    bool shareEncDecEmb;

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

    /* embedding of word at each position */
    Embedder* embedder;

    /* FNN model of each layer */
    FFN* ffns;

    /* attention model of each layer */
    Attention* selfAtts;

    /* layer normalization for attention */
    LayerNorm* selfAttLayerNorms;

    /* layer normalization for fnn */
    LayerNorm* ffnLayerNorms;

    /* layer normalization for decoder */
    LayerNorm* decoderLayerNorm;

    /* encoder-decoder attention model of each layer */
    Attention* enDeAtts;

    /* layer normalization for encoder-decoder attention */
    LayerNorm* enDeAttLayerNorms;

    /* dynamic layer history */
    LayerHistory* history;

    /* layer cache list */
    Cache* selfAttCache;

    /* layer cache list */
    Cache* enDeAttCache;

    /* the location of layer normalization */
    bool preLN;

    /* add LN to the decoder output or not */
    bool finalNorm;

    /* reserve history for layers or not */
    bool useHistory;

public:
    /* set the training flag */
    void SetTrainingFlag(bool myIsTraining);

    /* constructor */
    AttDecoder();

    /* de-constructor */
    ~AttDecoder();

    /* initialize the model */
    void InitModel(NMTConfig& myConfig);

    /* make the decoding network */
    XTensor Make(XTensor& inputDec, XTensor& outputEnc, XTensor* mask,
                 XTensor* maskEncDec, int nstep);

    /* run decoding for inference with pre-norm */
    XTensor RunFastPreNorm(XTensor& inputDec, XTensor& outputEnc, XTensor* maskEncDec, int nstep);

    /* run decoding for inference with post-norm */
    XTensor RunFastPostNorm(XTensor& inputDec, XTensor& outputEnc, XTensor* maskEncDec, int nstep);
};

} /* end of the nmt namespace */

#endif /* __DECODER_H__ */