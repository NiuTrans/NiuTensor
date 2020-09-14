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

#include <cstdint>

#include "T2TModel.h"
#include "module/T2TUtility.h"
#include "../../tensor/XUtility.h"
#include "../../tensor/core/CHeader.h"

namespace transformer
{

/* constructor */
T2TModel::T2TModel()
{
    devID = -1;
    isLM = false;
    isMT = false;
    useFP16 = false;
    shareAllEmbeddings = false;
    shareDecInputOutputWeight = false;
    nhead = 1;

    encoder = new AttEncoder();
    decoder = new AttDecoder();
    outputLayer = new T2TOutput();
}

/* de-constructor */
T2TModel::~T2TModel()
{
    delete encoder;
    delete decoder;
    delete outputLayer;
}

/*
initialize the model
>> config - configurations of the model
*/
void T2TModel::InitModel(T2TConfig& config)
{
    devID = config.devID;
    isMT = config.isMT;
    isLM = !isMT;
    useFP16 = config.useFP16;

    /* configurations for the model */
    int* metaInfo[] = {
        &config.nEncLayer, &config.nDecLayer,
        &config.fnnHiddenSize, &config.modelSize,
        &config.embSize, &config.srcVocabSize,
        &config.tgtVocabSize, &config.nhead,
        &config.maxRP, &shareAllEmbeddings,
        &shareDecInputOutputWeight,
        &config.maxPosLen
    };

    FILE* modelFile = NULL;

    /* read model configurations */
    if (!config.isTraining) {
        modelFile = fopen(config.modelFN, "rb");
        for (auto& meta : metaInfo)
            fread(meta, sizeof(int), 1, modelFile);
    }
    nhead = config.nhead;

    encoder->InitModel(config);
    outputLayer->InitModel(config);

    if (isMT)
        decoder->InitModel(config);

    TensorList params(10);
    GetParams(params);

    /* load parameters */
    if (!config.isTraining)
        Read(modelFile);
    else {
        for (int i = 0; i < params.Size(); i++)
            params[i]->SetVarFlag();
    }

    if (modelFile != NULL)
        fclose(modelFile);
}

/*
make the encoding network
>> input - input tensor
>> mask - the mask for positions that are/not involved in computation
>> isTraining - indicates whether we are training the model
<< return - encoding result
*/
XTensor T2TModel::MakeEncoder(XTensor& input, XTensor* mask, bool isTraining)
{
    XTensor nothing;

    return encoder->Make(input, mask, nothing, isTraining);
}

/*
make the decoding network
>> inputDec - input tensor of the decoder
>> outputEnc - output tensor of the encoder
>> output - output tensor (distribution)
>> mask - mask for positions that are/not involved in computation
>> maskEncDec - mask for the encoder-decoder attention
>> isTraining - indicates whether we are training the model
<< return - encoding result
*/
XTensor T2TModel::MakeDecoder(XTensor& inputDec, XTensor& outputEnc, 
                              XTensor* mask, XTensor& maskEncDec, bool isTraining)
{
    return decoder->Make(inputDec, outputEnc, mask, &maskEncDec, 
                         inputDec.GetDim(1), isTraining);
}

/*
make the network for language modeling (with the output softmax layer)
>> input - input tensor
>> output - output tensor (distribution)
>> padding - padding of the sequences
>> isTraining - indicates whether the model is for training
*/
void T2TModel::MakeLM(XTensor& input, XTensor& output, XTensor& padding, bool isTraining)
{
    int len = padding.GetDim(padding.order - 1);
    int* dims = new int[padding.order + 2];
    for (int i = 0; i < padding.order; i++)
        dims[i + 1] = padding.GetDim(i);
    dims[0] = nhead;
    dims[padding.order + 1] = len;
    XTensor mask;
    InitTensor(&mask, padding.order + 2, dims, X_FLOAT, padding.devID);

    delete[] dims;

    /* a upper triangular matrix where the cells of the upper triangular are set to -1e-9.
        this matrix can be used to prevent the attention to current or following words in
        a given sequence. */
    _SetDataLowTri(&mask, 1e9F, 0);
    ScaleAndShiftMe(mask, 1.0F, -1e9F);

    /* forward */
    XTensor encoding;

    encoding = MakeEncoder(input, &mask, isTraining);
    outputLayer->Make(encoding, output, true, true);
}

/*
make the network for machine translation (with the output softmax layer)
>> inputEnc - input tensor of the encoder
>> inputDec - input tensor of the decoder
>> output - output tensor (distribution)
>> paddingEnc - padding of the sequences (on the encoder side)
>> paddingDec - padding of the sequences (on the decoder side)
>> isTraining - indicates whether the model is for training
*/
void T2TModel::MakeMT(XTensor& inputEnc, XTensor& inputDec, XTensor& output,
    XTensor& paddingEnc, XTensor& paddingDec,
    bool isTraining)
{
    XTensor encoding;
    XTensor decoding;
    XTensor maskEnc;
    XTensor maskDec;
    XTensor maskEncDec;

    /* encoder mask */
    MakeMTMaskEnc(paddingEnc, maskEnc);

    /* decoder mask */
    MakeMTMaskDec(paddingEnc, paddingDec, maskDec, maskEncDec);

    encoding = MakeEncoder(inputEnc, &maskEnc, isTraining);

    decoding = MakeDecoder(inputDec, encoding, &maskDec, maskEncDec, isTraining);

    outputLayer->Make(decoding, output, true, true);
}

/*
make the mask for training MT models
>> inputEnc - input of the encoder
>> inputDec - input of the decoder
>> paddingEnc - padding of the encoder input
>> paddingDec - padding of the decoder input
>> maskEnc - mask of the encoder self-attention
>> maksDec - mask of the decoder self-attention
>> maksEncDec - mask of the decoder enc-dec attention
*/
void T2TModel::MakeMTMask(XTensor& inputEnc, XTensor& inputDec,
    XTensor& paddingEnc, XTensor& paddingDec,
    XTensor& maskEnc, XTensor& maskDec, XTensor& maskEncDec)
{
    int len = inputDec.GetDim(inputDec.order - 1);
    int* dims = new int[inputDec.order + 2];
    for (int i = 0; i < inputDec.order; i++)
        dims[i + 1] = inputDec.GetDim(i);
    dims[0] = nhead;
    dims[inputDec.order + 1] = len;
    InitTensor(&maskDec, inputDec.order + 2, dims, X_FLOAT, paddingDec.devID);

    /* an upper triangular matrix where the cells of the upper triangular are set to -1e-9.
       this matrix can be used to prevent the attention to current or following words in
       a given sequence. */
    _SetDataLowTri(&maskDec, 1e9F, 0);
    ScaleAndShiftMe(maskDec, 1.0F, -1e9F);

    /* encoder-decoder mask that prevents the attention to padding dummy words */
    dims[inputDec.order + 1] = inputEnc.GetDim(inputEnc.order - 1);
    InitTensor(&maskEncDec, inputDec.order + 2, dims, X_FLOAT, paddingEnc.devID);

    XTensor* maskEncDecTMPEnc = NewTensorBuf(paddingEnc.order + 1, dims + 1, 
                                paddingEnc.dataType, paddingEnc.devID);
    XTensor* maskEncDecTMPDec = NewTensorBuf(maskEncDecTMPEnc, paddingEnc.devID);

    _Unsqueeze(&paddingEnc, maskEncDecTMPEnc, paddingEnc.order - 1, paddingDec.GetDim(-1));
    _ScaleAndShiftMe(maskEncDecTMPEnc, 1e9F, -1e9F);
    _Unsqueeze(maskEncDecTMPEnc, &maskEncDec, 0, dims[0]);

    DelTensorBuf(maskEncDecTMPDec);
    DelTensorBuf(maskEncDecTMPEnc);

    /* padding on the source side */
    int* dimsPadding = new int[paddingEnc.order + 2];
    for (int i = 0; i < paddingEnc.order - 1; i++)
        dimsPadding[i] = paddingEnc.GetDim(i);
    dimsPadding[paddingEnc.order - 1] = paddingEnc.GetDim(-1);
    dimsPadding[paddingEnc.order] = paddingEnc.GetDim(-1);

    XTensor* padding2 = NewTensorBuf(paddingEnc.order + 1, dimsPadding, paddingEnc.dataType,
        paddingEnc.devID);

    for (int i = 0; i < padding2->order; i++)
        dimsPadding[i + 1] = padding2->GetDim(i);
    dimsPadding[0] = nhead;

    XTensor* padding3 = NewTensorBuf(paddingEnc.order + 2, dimsPadding, paddingEnc.dataType,
        paddingEnc.devID);

    /* mask of the padding */
    _Unsqueeze(&paddingEnc, padding2, paddingEnc.order - 1, paddingEnc.GetDim(-1));
    _Unsqueeze(padding2, padding3, 0, nhead);

    _ScaleAndShiftMe(padding3, 1e9F, -1e9F);

    InitTensor(&maskEnc, padding3);
    maskEnc.SetZeroAll();

    /* generate the mask on the source language side (for padding) */
    _Sum(&maskEnc, padding3, &maskEnc);

    delete[] dims;
    delete[] dimsPadding;

    DelTensorBuf(padding3);
    DelTensorBuf(padding2);
}

/*
make the mask of the encoder
>> inputEnc - input of the encoder
>> paddingEnc - padding of the encoder input
>> maskEnc - mask of the encoder self-attention
*/
void T2TModel::MakeMTMaskEnc(XTensor& paddingEnc, XTensor& maskEnc)
{
    XTensor padding2;
    XTensor padding3;

    /* mask of the padding */
    Unsqueeze(paddingEnc, padding2, paddingEnc.order - 1, paddingEnc.GetDim(-1));
    Unsqueeze(padding2, padding3, 0, nhead);
    ScaleAndShiftMe(padding3, 1e9F, -1e9F);

    InitTensor(&maskEnc, &padding3);
    maskEnc.SetZeroAll();

    /* generate the mask on the source language side (for padding) */
    SumMe(maskEnc, padding3);
}

/*
make the mask of the decoder
>> inputEnc - input of the encoder
>> inputDec - input of the decoder
>> paddingEnc - padding of the encoder input
>> paddingDec - padding of the decoder input
>> maksDec - mask of the decoder self-attention
>> maksEncDec - mask of the decoder enc-dec attention
*/
void T2TModel::MakeMTMaskDec(XTensor& paddingEnc, XTensor& paddingDec,
                             XTensor& maskDec, XTensor& maskEncDec)
{
    int len = paddingDec.GetDim(paddingDec.order - 1);
    int* dims = new int[paddingDec.order + 2];
    for (int i = 0; i < paddingDec.order; i++)
        dims[i + 1] = paddingDec.GetDim(i);
    dims[0] = nhead;
    dims[paddingDec.order + 1] = len;
    InitTensor(&maskDec, paddingDec.order + 2, dims, X_FLOAT, paddingDec.devID);

    /* An upper triangular matrix where the cells of the upper triangular are set to -1e-9.
       This matrix can be used to block the attention to current or following words in
       a given sequence. */
    _SetDataLowTri(&maskDec, 1e9F, 0);
    ScaleAndShiftMe(maskDec, 1.0F, -1e9F);

    /* encoder-decoder mask that prevents the attention to padding dummy words */
    XTensor maskEncDecTMP;

    Unsqueeze(paddingEnc, maskEncDecTMP, paddingEnc.order - 1, paddingDec.GetDim(-1));
    ScaleAndShiftMe(maskEncDecTMP, 1e9F, -1e9F);
    Unsqueeze(maskEncDecTMP, maskEncDec, 0, dims[0]);

    delete[] dims;
}
/*
get parameter matrices
>> list - the list that keeps the parameter matrics
*/
void T2TModel::GetParams(TensorList& list)
{
    list.Clear();

    /* encoder parameters */
    for (int i = 0; i < encoder->nlayer; i++) {
        list.Add(&encoder->selfAtt[i].wq);
        list.Add(&encoder->selfAtt[i].wk);
        list.Add(&encoder->selfAtt[i].wv);
        list.Add(&encoder->selfAtt[i].bq);
        list.Add(&encoder->selfAtt[i].bk);
        list.Add(&encoder->selfAtt[i].bv);
        if (encoder->selfAtt[i].useRPR)
            list.Add(&encoder->selfAtt[i].RPEmbK);
        list.Add(&encoder->selfAtt[i].wo);
        list.Add(&encoder->selfAtt[i].bo);
        list.Add(&encoder->fnns[i].w1);
        list.Add(&encoder->fnns[i].b1);
        list.Add(&encoder->fnns[i].w2);
        list.Add(&encoder->fnns[i].b2);
        list.Add(&encoder->attLayerNorms[i].w);
        list.Add(&encoder->attLayerNorms[i].b);
        list.Add(&encoder->fnnLayerNorms[i].w);
        list.Add(&encoder->fnnLayerNorms[i].b);
    }
    if (encoder->preNorm) {
        list.Add(&encoder->encoderLayerNorm->w);
        list.Add(&encoder->encoderLayerNorm->b);
    }

    if (isMT) {
        /* decoder parameters */
        for (int i = 0; i < decoder->nlayer; i++) {
            list.Add(&decoder->selfAtt[i].wq);
            list.Add(&decoder->selfAtt[i].wk);
            list.Add(&decoder->selfAtt[i].wv);
            list.Add(&decoder->selfAtt[i].bq);
            list.Add(&decoder->selfAtt[i].bk);
            list.Add(&decoder->selfAtt[i].bv);
            if (decoder->selfAtt[i].useRPR)
                list.Add(&decoder->selfAtt[i].RPEmbK);
            list.Add(&decoder->selfAtt[i].wo);
            list.Add(&decoder->selfAtt[i].bo);
            list.Add(&decoder->selfAttLayerNorms[i].w);
            list.Add(&decoder->selfAttLayerNorms[i].b);
            list.Add(&decoder->enDeAtt[i].wq);
            list.Add(&decoder->enDeAtt[i].wk);
            list.Add(&decoder->enDeAtt[i].wv);
            list.Add(&decoder->enDeAtt[i].bq);
            list.Add(&decoder->enDeAtt[i].bk);
            list.Add(&decoder->enDeAtt[i].bv);
            list.Add(&decoder->enDeAtt[i].wo);
            list.Add(&decoder->enDeAtt[i].bo);
            list.Add(&decoder->enDeAttLayerNorms[i].w);
            list.Add(&decoder->enDeAttLayerNorms[i].b);
            list.Add(&decoder->fnns[i].w1);
            list.Add(&decoder->fnns[i].b1);
            list.Add(&decoder->fnns[i].w2);
            list.Add(&decoder->fnns[i].b2);
            list.Add(&decoder->fnnLayerNorms[i].w);
            list.Add(&decoder->fnnLayerNorms[i].b);
        }
        if (decoder->preNorm) {
            list.Add(&decoder->decoderLayerNorm->w);
            list.Add(&decoder->decoderLayerNorm->b);
        }
    }

    list.Add(&encoder->embedder.w);

    if (isMT && (shareAllEmbeddings == 0)) {
        list.Add(&decoder->embedder.w);
    }

    if (shareDecInputOutputWeight == 0)
        list.Add(&outputLayer->w);
}

/*
dump the model to a file
>> fn - where to save the model
>> model - the model
*/
void T2TModel::Dump(const char* fn)
{
    double startT = GetClockSec();

    FILE* file = fopen(fn, "wb");
    CheckNTErrors(file, "Cannot open the model file");

    TensorList params(100);

    GetParams(params);

    int metaInfo[]{
        encoder->nlayer, decoder->nlayer,
        encoder->fnns->hSize, encoder->selfAtt->d,
        encoder->embedder.eSize, encoder->embedder.vSize,
        decoder->embedder.vSize, encoder->selfAtt->nhead,
        encoder->selfAtt->maxRP, shareAllEmbeddings,
        shareDecInputOutputWeight, encoder->embedder.maxLength - 1 - 1,
    };

    /* part 1: hyper-parameters */
    fwrite(metaInfo, sizeof(int), sizeof(metaInfo) / sizeof(int), file);

    /* part 2: model parameters */
    for (int i = 0; i < params.Size(); i++) {
        params[i]->BinaryDump(file);
    }

    fclose(file);

    double elapsed = GetClockSec() - startT;

    XPRINT1(0, stderr, "[INFO] model saved (took %.1fs)\n", elapsed);
}

/* read the parameters */
void T2TModel::Read(FILE* file)
{
    double startT = GetClockSec();

    TensorList params(100);
    GetParams(params);

    /* convert parameters to FP16 */
    if (useFP16) {
        for (int i = 0; i < params.Size(); i++) {
            XTensor* p = params[i];
            InitTensorV2(p, p->order, p->dimSize, X_FLOAT16, 1, p->devID);
        }

        auto& encEmb = encoder->embedder.posEmbeddingBase;
        auto& decEmb = decoder->embedder.posEmbeddingBase;
        encEmb = ConvertDataType(encEmb, X_FLOAT16);
        decEmb = ConvertDataType(decEmb, X_FLOAT16);
    }

    for (int i = 0; i < params.Size(); i++)
        params[i]->BinaryRead(file);

    /* share all embeddings */
    if (shareAllEmbeddings == 1) {
        decoder->embedder.w = CopyValues(encoder->embedder.w);
        XPRINT(0, stderr, "[INFO] sharing encoder decoder embeddings\n");
    }

    /* share embeddings with output weights */
    if (shareDecInputOutputWeight == 1) {
        outputLayer->w = CopyValues(decoder->embedder.w);
        XPRINT(0, stderr, "[INFO] sharing decoder embeddings with output weights\n");
    }

    double elapsed = GetClockSec() - startT;
    XPRINT1(0, stderr, "[INFO] model loaded (took %.1fs)\n", elapsed);
}

}