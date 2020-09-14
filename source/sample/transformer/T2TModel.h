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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-31
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04
 */

#ifndef __T2TMODEL_H__
#define __T2TMODEL_H__

#include "T2TEncoder.h"
#include "T2TDecoder.h"
#include "module/T2TFNN.h"
#include "module/T2TOutput.h"
#include "module/T2TUtility.h"
#include "module/T2TAttention.h"

namespace transformer
{

/* a transformer model that keeps parameters of the encoder,
   the decoder and the output layer (softmax). Also, it creates
   the network used in transformer. */
class T2TModel
{
public:
    /* device id */
    int devID;

    /* the encoder */
    AttEncoder* encoder;

    /* the decoder */
    AttDecoder* decoder;

    /* output layer */
    T2TOutput* outputLayer;

    /* indicates whether the model is running for language modeling */
    bool isLM;

    /* indicates whether the model is running for machine translation */
    bool isMT;

    /* indicates whether the model is running with FP16 data type */
    bool useFP16;

    /* number of heads in the attention model */
    int nhead;

    /* indicates whether share encoders embeddings with decoders */
    int shareAllEmbeddings;

    /* indicates whether share decoder embeddings with output weights */
    int shareDecInputOutputWeight;

public:
    /* constructor */
    T2TModel();

    /* de-constructor */
    ~T2TModel();

    /* initialize the model */
    void InitModel(T2TConfig& config);

    /* make the encoding network */
    XTensor MakeEncoder(XTensor& input, XTensor* mask, bool isTraining);

    /* make the encoding network */
    XTensor MakeDecoder(XTensor& inputEnc, XTensor& inputDec, XTensor* mask,
        XTensor& MaskEncDec, bool isTraining);

    /* make the network for language modeling (with the output softmax layer) */
    void MakeLM(XTensor& input, XTensor& output, XTensor& padding, bool isTraining);

    /* make the network for machine translation (with the output softmax layer) */
    void MakeMT(XTensor& inputEnc, XTensor& inputDec, XTensor& output,
        XTensor& paddingEnc, XTensor& paddingDec, bool isTraining);

    /* make the mask for training MT models */
    void MakeMTMask(XTensor& inputEnc, XTensor& inputDec,
        XTensor& paddingEnc, XTensor& paddingDec,
        XTensor& maskEnc, XTensor& maskDec, XTensor& maskEncDec);

    /* make the mask of the encoder */
    void MakeMTMaskEnc(XTensor& paddingEnc, XTensor& maskEnc);

    /* make the mask of the decoder */
    void MakeMTMaskDec(XTensor& paddingEnc, XTensor& paddingDec,
        XTensor& maskDec, XTensor& maskEncDec);

    /* get parameter matrices */
    void GetParams(TensorList& list);

    /* dump the model to a file */
    void Dump(const char* fn);

    /* read the parameters */
    void Read(FILE* file);
};

}

#endif
