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

#include <cstdint>
#include "Model.h"
#include "../../tensor/XUtility.h"
#include "../../tensor/core/CHeader.h"

/* the nmt namespace */
namespace nmt
{

/* constructor */
NMTModel::NMTModel()
{
    devID = -1;
    config = NULL;
    encoder = new AttEncoder();
    decoder = new AttDecoder();
    outputLayer = new OutputLayer();
}

/* de-constructor */
NMTModel::~NMTModel()
{
    delete encoder;
    delete decoder;
    delete outputLayer;
}

/* return a list to keep the configurations (interger) */
vector<int*> NMTModel::GetIntConfigs()
{
    vector<int*> intConfig = {
        &(config->model.encEmbDim),
        &(config->model.encLayerNum),
        &(config->model.encSelfAttHeadNum),
        &(config->model.encFFNHiddenDim),
        &(config->model.decEmbDim),
        &(config->model.decLayerNum),
        &(config->model.decSelfAttHeadNum),
        &(config->model.encDecAttHeadNum),
        &(config->model.decFFNHiddenDim),
        &(config->model.maxRelativeLength),
        &(config->model.maxSrcLen),
        &(config->model.maxTgtLen),
        &(config->model.sos),
        &(config->model.eos),
        &(config->model.pad),
        &(config->model.unk),
        &(config->model.srcVocabSize),
        &(config->model.tgtVocabSize),
    };

    return intConfig;
}

/*
initialize the model
>> myConfig - configuration of the model
*/
void NMTModel::InitModel(NMTConfig& myConfig)
{
    config = &myConfig;
    devID = config->common.devID;

    /* configurations for the model */
    vector<int*> intConfig = GetIntConfigs();

    FILE* modelFile = NULL;
    modelFile = fopen(config->common.modelFN, "rb");

    /* read model configurations */
    if (modelFile) {

        CheckNTErrors(modelFile, "Failed to open the model file");

        LOG("loading configurations from the model file...");

        fread(&(config->model.encoderL1Norm), sizeof(bool), 1, modelFile);
        fread(&(config->model.decoderL1Norm), sizeof(bool), 1, modelFile);
        fread(&(config->model.useBigAtt), sizeof(bool), 1, modelFile);
        fread(&(config->model.encFinalNorm), sizeof(bool), 1, modelFile);
        fread(&(config->model.decFinalNorm), sizeof(bool), 1, modelFile);
        fread(&(config->model.encPreLN), sizeof(bool), 1, modelFile);
        fread(&(config->model.decPreLN), sizeof(bool), 1, modelFile);
        fread(&(config->model.useEncHistory), sizeof(bool), 1, modelFile);
        fread(&(config->model.useDecHistory), sizeof(bool), 1, modelFile);
        fread(&(config->model.shareEncDecEmb), sizeof(bool), 1, modelFile);
        fread(&(config->model.shareDecInputOutputEmb), sizeof(bool), 1, modelFile);

        int maxSrcLen = config->model.maxSrcLen;
        for (auto c : intConfig) {
            fread(c, sizeof(int), 1, modelFile);
        }
        /* reset the maximum source sentence length */
        config->model.maxSrcLen = MIN(maxSrcLen, config->model.maxSrcLen);
    }

    if (config->training.isTraining) {

        /* currently we do not support training with FP16 */
        config->common.useFP16 = false;

        /* read the source & target vocab size and special tokens from the training file */
        FILE* trainF = fopen(config->training.trainFN, "rb");
        CheckNTErrors(trainF, "Failed to open the training file");

        LOG("loading configurations of the training data...");

        fread(&(config->model.srcVocabSize), sizeof(int), 1, trainF);
        fread(&(config->model.tgtVocabSize), sizeof(int), 1, trainF);
        fread(&(config->model.pad), sizeof(int), 1, trainF);
        fread(&(config->model.sos), sizeof(int), 1, trainF);
        fread(&(config->model.eos), sizeof(int), 1, trainF);
        fread(&(config->model.unk), sizeof(int), 1, trainF);
        CheckNTErrors(config->model.srcVocabSize > 0, "Invalid source vocabulary size");
        CheckNTErrors(config->model.tgtVocabSize > 0, "Invalid target vocabulary size");
        fclose(trainF);

        /* start incremental training from a checkpoint */
        if (modelFile) {
            config->training.incremental = true;
        }
    }

    encoder->InitModel(*config);
    decoder->InitModel(*config);
    outputLayer->InitModel(*config);

    /* share encoder&decoder embeddings */
    if (config->model.shareEncDecEmb) {
        decoder->embedder = &(encoder->embedder);
        LOG("share encoder decoder embeddings");
    }

    /* share embeddings with output weights */
    if (config->model.shareDecInputOutputEmb) {
        outputLayer->w = decoder->embedder->w;
        LOG("share decoder embeddings with output weights");
    }

    ShowModelConfig();

    /* load parameters for translation or incremental training */
    if (config->training.incremental || (!config->training.isTraining))
        LoadFromFile(modelFile);

    if (config->training.isTraining) {
        TensorList params;
        GetParams(params);
        for (int i = 0; i < params.Size(); i++)
            AddParam(params[i]);
    }

    if (modelFile)
        fclose(modelFile);
}

/*
print model configurations
*/
void NMTModel::ShowModelConfig()
{
    LOG("model configuration:");
    if (config->model.encPreLN)
        LOG("encoder pre-norm with %s", config->model.encoderL1Norm ? "l1-norm" : "l2-norm");
    else
        LOG("encoder post-norm with %s", config->model.encoderL1Norm ? "l1-norm" : "l2-norm");
    if (config->model.decPreLN)
        LOG("decoder pre-norm with %s", config->model.decoderL1Norm ? "l1-norm" : "l2-norm");
    else
        LOG("decoder post-norm with %s", config->model.decoderL1Norm ? "l1-norm" : "l2-norm");
    if (config->model.maxRelativeLength > 0)
        LOG("rpr length: %d", config->model.maxRelativeLength);
    LOG("encoder embedding dim: %d", config->model.encEmbDim);
    LOG("encoder layers: %d", config->model.encLayerNum);
    LOG("encoder heads: %d", config->model.encSelfAttHeadNum);
    LOG("encoder ffn hidden dim: %d", config->model.encFFNHiddenDim);
    LOG("decoder embedding dim: %d", config->model.decEmbDim);
    LOG("decoder layers: %d", config->model.decLayerNum);
    LOG("decoder self-att heads: %d", config->model.decSelfAttHeadNum);
    LOG("decoder en-de-att heads: %d", config->model.encDecAttHeadNum);
    LOG("decoder ffn hidden dim: %d", config->model.decFFNHiddenDim);
    LOG("number of parameters: %zu", GetParamNum());
}

/*
make the encoding network
>> input - input tensor, (batchSize, srcLen)
>> mask - the mask for encoder self-attention, (headNum, batchSize, srcLen, srcLen)
<< return - encoding result, (batchSize, srcLen, hiddenDim)
*/
XTensor NMTModel::MakeEncoder(XTensor& input, XTensor* mask)
{
    XTensor nothing;

    return encoder->Make(input, mask, nothing);
}

/*
make the decoding network
>> inputDec - input tensor of the decoder, (batchSize, tgtLen)
>> outputEnc - output tensor of the encoder, (batchSize, srcLen, hiddenDim)
>> mask - mask for decoder self-attention, (headNum, batchSize, tgtLen, tgtLen)
>> maskEncDec - mask for the encoder-decoder attention, (headNum, batchSize, tgtLen, srcLen)
<< return - decoding result, (batchSize, tgtLen, hiddenDim)
*/
XTensor NMTModel::MakeDecoder(XTensor& inputDec, XTensor& outputEnc,
                              XTensor* mask, XTensor& maskEncDec)
{
    return decoder->Make(inputDec, outputEnc, mask, &maskEncDec,
                         inputDec.GetDim(1));
}

/*
make the network for language modeling (with the output softmax layer)
>> input - input tensor
>> padding - padding of the sequences
<< output - output tensor (distribution)
*/
XTensor NMTModel::MakeLM(XTensor& input, XTensor& padding)
{
    int len = padding.GetDim(padding.order - 1);
    int* dims = new int[padding.order + 2];
    for (int i = 0; i < padding.order; i++)
        dims[i + 1] = padding.GetDim(i);
    dims[0] = config->model.encSelfAttHeadNum;
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

    encoding = MakeEncoder(input, &mask);
    return outputLayer->Make(encoding, true);
}

/*
make the network for machine translation (with the output softmax layer)
>> inputEnc - input tensor of the encoder, (batchSize, srcLen)
>> inputDec - input tensor of the decoder, (batchSize, tgtLen)
>> paddingEnc - padding of the sequences (on the encoder side), (batchSize, srcLen)
>> paddingDec - padding of the sequences (on the decoder side), (batchSize, tgtLen)
<< output - output tensor (distribution), (batchSize, tgtLen, hiddenDim)
*/
XTensor NMTModel::MakeMT(XTensor& inputEnc, XTensor& inputDec,
                         XTensor& paddingEnc, XTensor& paddingDec)
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

    encoding = MakeEncoder(inputEnc, &maskEnc);

    decoding = MakeDecoder(inputDec, encoding, &maskDec, maskEncDec);

    return outputLayer->Make(decoding, true);
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
void NMTModel::MakeMTMask(XTensor& inputEnc, XTensor& inputDec,
                          XTensor& paddingEnc, XTensor& paddingDec,
                          XTensor& maskEnc, XTensor& maskDec, XTensor& maskEncDec)
{
    int len = inputDec.GetDim(inputDec.order - 1);
    int* dims = new int[inputDec.order + 2];
    for (int i = 0; i < inputDec.order; i++)
        dims[i + 1] = inputDec.GetDim(i);
    dims[0] = config->model.decSelfAttHeadNum;
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

    GMems.GetMem(paddingEnc.devID)->LockBuf();
    XTensor* maskEncDecTMPEnc = NewTensorBufV2(paddingEnc.order + 1, dims + 1,
        paddingEnc.dataType, 1.0F, paddingEnc.devID, paddingEnc.mem);
    XTensor* maskEncDecTMPDec = NewTensorBufV2(maskEncDecTMPEnc, paddingEnc.devID, paddingEnc.mem);

    _Unsqueeze(&paddingEnc, maskEncDecTMPEnc, paddingEnc.order - 1, paddingDec.GetDim(-1));
    _ScaleAndShiftMe(maskEncDecTMPEnc, 1e9F, -1e9F);
    _Unsqueeze(maskEncDecTMPEnc, &maskEncDec, 0, dims[0]);

    DelTensorBuf(maskEncDecTMPDec);
    DelTensorBuf(maskEncDecTMPEnc);
    GMems.GetMem(paddingEnc.devID)->UnlockBuf();

    /* padding on the source side */
    int* dimsPadding = new int[paddingEnc.order + 2];
    for (int i = 0; i < paddingEnc.order - 1; i++)
        dimsPadding[i] = paddingEnc.GetDim(i);
    dimsPadding[paddingEnc.order - 1] = paddingEnc.GetDim(-1);
    dimsPadding[paddingEnc.order] = paddingEnc.GetDim(-1);

    GMems.GetMem(paddingEnc.devID)->LockBuf();
    XTensor* padding2 = NewTensorBufV2(paddingEnc.order + 1, dimsPadding, paddingEnc.dataType, 1.0F,
        paddingEnc.devID, paddingEnc.mem);

    for (int i = 0; i < padding2->order; i++)
        dimsPadding[i + 1] = padding2->GetDim(i);
    dimsPadding[0] = config->model.decSelfAttHeadNum;

    XTensor* padding3 = NewTensorBufV2(paddingEnc.order + 2, dimsPadding, paddingEnc.dataType, 1.0F, paddingEnc.devID, paddingEnc.mem);

    /* mask of the padding */
    _Unsqueeze(&paddingEnc, padding2, paddingEnc.order - 1, paddingEnc.GetDim(-1));
    _Unsqueeze(padding2, padding3, 0, config->model.decSelfAttHeadNum);

    _ScaleAndShiftMe(padding3, 1e9F, -1e9F);

    InitTensor(&maskEnc, padding3);
    maskEnc.SetZeroAll();

    /* generate the mask on the source language side (for padding) */
    _Sum(&maskEnc, padding3, &maskEnc);

    delete[] dims;
    delete[] dimsPadding;

    DelTensorBuf(padding3);
    DelTensorBuf(padding2);
    GMems.GetMem(paddingEnc.devID)->UnlockBuf();
}

/*
make the mask of the encoder
>> paddingEnc - padding of the encoder input, (batchSize, srcLen)
>> maskEnc - mask of the encoder self-attention, (headNum, batchSize, srcLen, srcLen)
*/
void NMTModel::MakeMTMaskEnc(XTensor& paddingEnc, XTensor& maskEnc)
{
    XTensor padding2;

    /* mask of the padding */
    Unsqueeze(paddingEnc, padding2, paddingEnc.order - 1, paddingEnc.GetDim(-1));
    Unsqueeze(padding2, maskEnc, 0, config->model.encSelfAttHeadNum);
    ScaleAndShiftMe(maskEnc, 1e9F, -1e9F);
}

/*
make the mask of the decoder
>> paddingEnc - padding of the encoder input, (batchSize, srcLen)
>> paddingDec - padding of the decoder input, (batchSize, tgtLen)
>> maksDec - mask of the decoder self-attention, (headNum, batchSize, tgtLen, tgtLen)
>> maksEncDec - mask of the decoder enc-dec attention, (headNum, batchSize, tgtLen, srcLen)
*/
void NMTModel::MakeMTMaskDec(XTensor& paddingEnc, XTensor& paddingDec,
                             XTensor& maskDec, XTensor& maskEncDec)
{
    if (config->training.isTraining) {
        int len = paddingDec.GetDim(paddingDec.order - 1);
        int* dims = new int[paddingDec.order + 2];
        for (int i = 0; i < paddingDec.order; i++)
            dims[i + 1] = paddingDec.GetDim(i);
        dims[0] = config->model.decSelfAttHeadNum;
        dims[paddingDec.order + 1] = len;
        InitTensor(&maskDec, paddingDec.order + 2, dims, X_FLOAT, paddingDec.devID);

        /* An upper triangular matrix where the cells of the upper triangular are set to -1e-9.
           This matrix can be used to block the attention to current or following words in
           a given sequence. */
        _SetDataLowTri(&maskDec, 1e9F, 0);
        ScaleAndShiftMe(maskDec, 1.0F, -1e9F);
        delete[] dims;
    }

    /* encoder-decoder mask that prevents the attention to padding dummy words */
    XTensor maskEncDecTMP;

    Unsqueeze(paddingEnc, maskEncDecTMP, paddingEnc.order - 1, paddingDec.GetDim(-1));
    if (config->model.encDecAttHeadNum > 1)
        Unsqueeze(maskEncDecTMP, maskEncDec, 0, config->model.encDecAttHeadNum);
    else
        maskEncDec = maskEncDecTMP;
    ScaleAndShiftMe(maskEncDec, 1e9F, -1e9F);
}

/*
make the mask of the decoder
>> paddingEnc - padding of the encoder input, (batchSize, srcLen)
<< maksEncDec - mask of the decoder enc-dec attention, 
   (nHead, batchSize, 1, srcLen) if nHead > 1,
   (batchSize, 1, srcLen) else.
*/
XTensor NMTModel::MakeMTMaskDecInference(XTensor& paddingEnc)
{
    /* encoder-decoder mask that prevents the attention to paded words */
    XTensor maskEncDecTMP;
    maskEncDecTMP = Unsqueeze(paddingEnc, paddingEnc.order - 1, 1);

    if (config->model.encDecAttHeadNum > 1) {
        XTensor maskEncDec;
        Unsqueeze(maskEncDecTMP, maskEncDec, 0, config->model.encDecAttHeadNum);
        ScaleAndShiftMe(maskEncDec, 1e9F, -1e9F);
        return maskEncDec;
    }
    else {
        ScaleAndShiftMe(maskEncDecTMP, 1e9F, -1e9F);
        return maskEncDecTMP;
    }
}

/*
todo: used a fixed parameter order
collect all parameters
>> list - the list that keeps the parameters
*/
void NMTModel::GetParams(TensorList& list)
{
    list.Clear();

    if (config->model.useBigAtt) {

        /* encoder parameters */
        if (!config->model.decoderOnly) {
            if (encoder->useHistory) {
                for (int i = 0; i < encoder->nlayer + 1; i++)
                    list.Add(&encoder->history->weights[i]);
                for (int i = 0; i < encoder->nlayer; i++) {
                    list.Add(&encoder->history->layerNorms[i].weight);
                    list.Add(&encoder->history->layerNorms[i].bias);
                }
            }
            for (int i = 0; i < encoder->nlayer; i++) {
                list.Add(&encoder->selfAtts[i].weightQ);
                list.Add(&encoder->selfAtts[i].weightK);
                list.Add(&encoder->selfAtts[i].weightV);
                list.Add(&encoder->selfAtts[i].biasQ);
                list.Add(&encoder->selfAtts[i].biasK);
                list.Add(&encoder->selfAtts[i].biasV);
                if (encoder->selfAtts[i].useRPR)
                    list.Add(&encoder->selfAtts[i].RPEmbK);
                list.Add(&encoder->selfAtts[i].weightO);
                list.Add(&encoder->selfAtts[i].biasO);
                list.Add(&encoder->ffns[i].w1);
                list.Add(&encoder->ffns[i].b1);
                list.Add(&encoder->ffns[i].w2);
                list.Add(&encoder->ffns[i].b2);
                list.Add(&encoder->attLayerNorms[i].weight);
                list.Add(&encoder->attLayerNorms[i].bias);
                list.Add(&encoder->fnnLayerNorms[i].weight);
                list.Add(&encoder->fnnLayerNorms[i].bias);
            }
            if (encoder->finalNorm) {
                list.Add(&encoder->encoderLayerNorm->weight);
                list.Add(&encoder->encoderLayerNorm->bias);
            }
        }

        /* decoder parameters */
        if (decoder->useHistory) {
            for (int i = 0; i < decoder->nlayer + 1; i++)
                list.Add(&decoder->history->weights[i]);
            for (int i = 0; i < decoder->nlayer; i++) {
                list.Add(&decoder->history->layerNorms[i].weight);
                list.Add(&decoder->history->layerNorms[i].bias);
            }
        }

        for (int i = 0; i < decoder->nlayer; i++) {
            list.Add(&decoder->selfAtts[i].weightQ);
            list.Add(&decoder->selfAtts[i].weightK);
            list.Add(&decoder->selfAtts[i].weightV);
            list.Add(&decoder->selfAtts[i].biasQ);
            list.Add(&decoder->selfAtts[i].biasK);
            list.Add(&decoder->selfAtts[i].biasV);
            if (decoder->selfAtts[i].useRPR)
                list.Add(&decoder->selfAtts[i].RPEmbK);
            list.Add(&decoder->selfAtts[i].weightO);
            list.Add(&decoder->selfAtts[i].biasO);
            list.Add(&decoder->selfAttLayerNorms[i].weight);
            list.Add(&decoder->selfAttLayerNorms[i].bias);
            if (!config->model.decoderOnly) {
                list.Add(&decoder->enDeAtts[i].weightQ);
                list.Add(&decoder->enDeAtts[i].weightK);
                list.Add(&decoder->enDeAtts[i].weightV);
                list.Add(&decoder->enDeAtts[i].biasQ);
                list.Add(&decoder->enDeAtts[i].biasK);
                list.Add(&decoder->enDeAtts[i].biasV);
                list.Add(&decoder->enDeAtts[i].weightO);
                list.Add(&decoder->enDeAtts[i].biasO);
                list.Add(&decoder->enDeAttLayerNorms[i].weight);
                list.Add(&decoder->enDeAttLayerNorms[i].bias);
            }
            if (decoder->ffns != NULL) {
                list.Add(&decoder->ffns[i].w1);
                list.Add(&decoder->ffns[i].b1);
                list.Add(&decoder->ffns[i].w2);
                list.Add(&decoder->ffns[i].b2);
            }
            list.Add(&decoder->ffnLayerNorms[i].weight);
            list.Add(&decoder->ffnLayerNorms[i].bias);
        }
    }
    else {
        /* encoder parameters */
        if (!config->model.decoderOnly) {
            if (encoder->useHistory) {
                for (int i = 0; i < encoder->nlayer + 1; i++)
                    list.Add(&encoder->history->weights[i]);
                for (int i = 0; i < encoder->nlayer; i++) {
                    list.Add(&encoder->history->layerNorms[i].weight);
                    list.Add(&encoder->history->layerNorms[i].bias);
                }
            }
            for (int i = 0; i < encoder->nlayer; i++) {
                if (encoder->selfAtts[i].useRPR)
                    list.Add(&encoder->selfAtts[i].RPEmbK);
                list.Add(&encoder->selfAtts[i].weightK);
                list.Add(&encoder->selfAtts[i].biasK);
                list.Add(&encoder->selfAtts[i].weightV);
                list.Add(&encoder->selfAtts[i].biasV);
                list.Add(&encoder->selfAtts[i].weightQ);
                list.Add(&encoder->selfAtts[i].biasQ);
                list.Add(&encoder->selfAtts[i].weightO);
                list.Add(&encoder->selfAtts[i].biasO);
                list.Add(&encoder->attLayerNorms[i].weight);
                list.Add(&encoder->attLayerNorms[i].bias);
                list.Add(&encoder->ffns[i].w1);
                list.Add(&encoder->ffns[i].b1);
                list.Add(&encoder->ffns[i].w2);
                list.Add(&encoder->ffns[i].b2);
                list.Add(&encoder->fnnLayerNorms[i].weight);
                list.Add(&encoder->fnnLayerNorms[i].bias);
            }
            if (encoder->finalNorm) {
                list.Add(&encoder->encoderLayerNorm->weight);
                list.Add(&encoder->encoderLayerNorm->bias);
            }
        }

        /* decoder parameters */
        if (decoder->useHistory) {
            for (int i = 0; i < decoder->nlayer + 1; i++)
                list.Add(&decoder->history->weights[i]);
            for (int i = 0; i < decoder->nlayer; i++) {
                list.Add(&decoder->history->layerNorms[i].weight);
                list.Add(&decoder->history->layerNorms[i].bias);
            }
        }

        for (int i = 0; i < decoder->nlayer; i++) {
            if (decoder->selfAtts[i].useRPR)
                list.Add(&decoder->selfAtts[i].RPEmbK);
            list.Add(&decoder->selfAtts[i].weightK);
            list.Add(&decoder->selfAtts[i].biasK);
            list.Add(&decoder->selfAtts[i].weightV);
            list.Add(&decoder->selfAtts[i].biasV);
            list.Add(&decoder->selfAtts[i].weightQ);
            list.Add(&decoder->selfAtts[i].biasQ);
            list.Add(&decoder->selfAtts[i].weightO);
            list.Add(&decoder->selfAtts[i].biasO);
            list.Add(&decoder->selfAttLayerNorms[i].weight);
            list.Add(&decoder->selfAttLayerNorms[i].bias);
            if (!config->model.decoderOnly) {
                list.Add(&decoder->enDeAtts[i].weightK);
                list.Add(&decoder->enDeAtts[i].biasK);
                list.Add(&decoder->enDeAtts[i].weightV);
                list.Add(&decoder->enDeAtts[i].biasV);
                list.Add(&decoder->enDeAtts[i].weightQ);
                list.Add(&decoder->enDeAtts[i].biasQ);
                list.Add(&decoder->enDeAtts[i].weightO);
                list.Add(&decoder->enDeAtts[i].biasO);
                list.Add(&decoder->enDeAttLayerNorms[i].weight);
                list.Add(&decoder->enDeAttLayerNorms[i].bias);
            }
            if (decoder->ffns != NULL) {
                list.Add(&decoder->ffns[i].w1);
                list.Add(&decoder->ffns[i].b1);
                list.Add(&decoder->ffns[i].w2);
                list.Add(&decoder->ffns[i].b2);
            }
            list.Add(&decoder->ffnLayerNorms[i].weight);
            list.Add(&decoder->ffnLayerNorms[i].bias);
        }
    }

    if (decoder->finalNorm) {
        list.Add(&decoder->decoderLayerNorm->weight);
        list.Add(&decoder->decoderLayerNorm->bias);
    }

    if (!config->model.decoderOnly) {
        list.Add(encoder->embedder.w);
    }

    if (!config->model.shareEncDecEmb) {
        list.Add(decoder->embedder->w);
    }

    if (!config->model.shareDecInputOutputEmb) {
        list.Add(outputLayer->w);
    }
}

/*
dump the model to a file
>> fn - where to save the model
*/
void NMTModel::DumpToFile(const char* fn)
{
    double startT = GetClockSec();
    FILE* modelFile = fopen(fn, "wb");
    CheckNTErrors(modelFile, "Cannot open the model file");

    vector<int*> intConfig = GetIntConfigs();

    /* save the configurations */
    fwrite(&(config->model.encoderL1Norm), sizeof(bool), 1, modelFile);
    fwrite(&(config->model.decoderL1Norm), sizeof(bool), 1, modelFile);
    fwrite(&(config->model.useBigAtt), sizeof(bool), 1, modelFile);
    fwrite(&(config->model.encFinalNorm), sizeof(bool), 1, modelFile);
    fwrite(&(config->model.decFinalNorm), sizeof(bool), 1, modelFile);
    fwrite(&(config->model.encPreLN), sizeof(bool), 1, modelFile);
    fwrite(&(config->model.decPreLN), sizeof(bool), 1, modelFile);
    fwrite(&(config->model.useEncHistory), sizeof(bool), 1, modelFile);
    fwrite(&(config->model.useDecHistory), sizeof(bool), 1, modelFile);
    fwrite(&(config->model.shareEncDecEmb), sizeof(bool), 1, modelFile);
    fwrite(&(config->model.shareDecInputOutputEmb), sizeof(bool), 1, modelFile);
    for (auto c : intConfig) {
        fwrite(c, sizeof(int), 1, modelFile);
    }

    /* save the model parameters */
    TensorList params;
    GetParams(params);
    for (int i = 0; i < params.Size(); i++) {
        params[i]->BinaryDump(modelFile);
    }

    fclose(modelFile);
    double elapsed = GetClockSec() - startT;
    LOG("model saved (took %.1fs)", elapsed);
}

/* read the parameters */
void NMTModel::LoadFromFile(FILE* file)
{
    double startT = GetClockSec();

    LOG("loading parameters from the model file...");

    TensorList params;
    GetParams(params);

    int size = 0;
    for (int i = 0; i < params.Size(); i++) {
        size += params[i]->unitNum;
    }

    if (config->common.useFP16) {
        LOG("running with fp16");
    }
    else {
        LOG("running with fp32");
    }

    /* convert parameters to FP16 before reading files */
    if (config->common.useFP16) {
        for (int i = 0; i < params.Size(); i++) {
            XTensor* p = params[i];
            InitTensor(p, p->order, p->dimSize, X_FLOAT16, p->devID, p->enableGrad && X_ENABLE_GRAD);
        }

        XTensor& encEmb = encoder->embedder.posEmbeddingBase;
        encEmb = ConvertDataType(encEmb, X_FLOAT16);
        if (!config->model.shareEncDecEmb) {
            XTensor& decEmb = decoder->embedder->posEmbeddingBase;
            decEmb = ConvertDataType(decEmb, X_FLOAT16);
        }
    }

    for (int i = 0; i < params.Size(); i++)
        params[i]->BinaryRead(file);

    double elapsed = GetClockSec() - startT;
    LOG("model loaded (took %.1fs)", elapsed);

}

/* get the total number of parameters */
uint64_t NMTModel::GetParamNum()
{
    TensorList params;
    GetParams(params);
    uint64_t totalNum = 0;
    for (int i = 0; i < params.Size(); i++) {
        totalNum += uint64_t(params[i]->unitNum);
    }

    return totalNum;
}

/* set the training flags in all sub-models */
void NMTModel::SetTrainingFlag(bool isTraining)
{
    encoder->SetTrainingFlag(isTraining);
    decoder->SetTrainingFlag(isTraining);
    outputLayer->SetTrainingFlag(isTraining);
}

XModel* NMTModel::Clone(int devID)
{
    NMTModel* newModel = new NMTModel();
    newModel->InitModel(*config);
    return newModel;
}

bool NMTModel::RunSimple(XList* inputs, XList* outputs, XList* golds, XList* losses)
{
    CheckNTErrors(inputs != NULL && inputs->count == 2, "Wrong arguments!");
    CheckNTErrors(outputs != NULL && outputs->count == 1, "Wrong arguments!");
    CheckNTErrors(golds != NULL && golds->count == 3, "Wrong arguments!");
    CheckNTErrors(losses != NULL && losses->count == 1, "Wrong arguments!");

    XTensor* batchEnc = (XTensor*)inputs->GetItem(0);
    XTensor* paddingEnc = (XTensor*)inputs->GetItem(1);
    XTensor* batchDec = (XTensor*)golds->GetItem(0);
    XTensor* paddingDec = (XTensor*)golds->GetItem(1);
    XTensor* label = (XTensor*)golds->GetItem(2);

    XTensor* output = (XTensor*)outputs->GetItem(0);
    XTensor* loss = (XTensor*)losses->GetItem(0);

    /* place all input data on the correct device */
    batchEnc->FlushToDevice(devID);
    paddingEnc->FlushToDevice(devID);
    batchDec->FlushToDevice(devID);
    paddingDec->FlushToDevice(devID);
    label->FlushToDevice(devID);

    XNet net;

    /* make the network */
    *output = MakeMT(*batchEnc, *batchDec, *paddingEnc, *paddingDec);

    /* get loss and probabilities */
    XTensor labelOnehot;

    labelOnehot = IndexToOnehot(label, config->model.tgtVocabSize, config->training.labelSmoothingP);

    *loss = CrossEntropy(output, labelOnehot, paddingDec);

    float lossBatch = ReduceSumAllValue(*loss);

    bool doUpdate = (!IsNAN(lossBatch) && !IsINF(lossBatch) && lossBatch < 1e3F);

    if (doUpdate) {

        /* back-propagation */
        net.Backward(*loss);

        if (encoder->useHistory)
            encoder->history->ClearHistory(/*reset=*/false);
        if (decoder->useHistory)
            decoder->history->ClearHistory(/*reset=*/false);
    }

    return true;
}

} /* end of the nmt namespace */