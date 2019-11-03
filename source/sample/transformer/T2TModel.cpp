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


#include "T2TModel.h"
#include "T2TUtility.h"
#include "../../tensor/core/CHeader.h"
#include "../../tensor/XUtility.h"

namespace transformer
{

/* constructor */
T2TModel::T2TModel()
{
    devID = -1;
    isLM = false;
    isMT = false;
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
>> argc - number of arguments
>> argv - list of pointers to the arguments
*/
void T2TModel::InitModel(int argc, char ** argv)
{
    LoadParamInt(argc, argv, "dev", &devID, -1);
    LoadParamBool(argc, argv, "mt", &isMT, false);
    LoadParamBool(argc, argv, "lm", &isLM, !isMT);
    LoadParamInt(argc, argv, "nhead", &nhead, 8);

    encoder->InitModel(argc, argv, true, 0, devID);
    outputLayer->InitModel(argc, argv, devID);

    if(isMT)
        decoder->InitModel(argc, argv, true, 0, devID);

    TensorList params(10);
    GetParams(params);

    for(int i = 0; i < params.count; i++){
        XTensor * param = (XTensor*)params.Get(i);
        param->SetVarFlag();
    }
}

/* 
make the encoding network
>> input - input tensor
>> mask - the mask for positions that are/not involved in computation
>> isTraining - indicates whether we are training the model
<< return - encoding result
*/
XTensor T2TModel::MakeEncoder(XTensor &input, XTensor &mask, bool isTraining)
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
XTensor T2TModel::MakeDecoder(XTensor &inputDec, XTensor &outputEnc, XTensor &mask, XTensor &maskEncDec, bool isTraining)
{
    return decoder->Make(inputDec, outputEnc, mask, maskEncDec, isTraining);
}

/* 
make the network for language modeling (with the output softmax layer) 
>> input - input tensor
>> output - output tensor (distribution)
>> padding - padding of the sequences
>> isTraining - indicates whether the model is for training
*/
void T2TModel::MakeLM(XTensor &input, XTensor &output, XTensor &padding, bool isTraining)
{
    XTensor encoding;
    
    /* generate mask to see "previous" words only */
    //int len = input.GetDim(input.order - 2);
    //int * dims = new int[input.order + 1];
    //for(int i = 0; i < input.order; i++)
    //    dims[i + 1] = input.GetDim(i);
    //dims[0] = nhead;
    //dims[input.order] = len;
    //XTensor mask(input.order + 1, dims, X_FLOAT, 1.0F, input.devID, input.mem);

    int len = input.GetDim(input.order - 1);
    int * dims = new int[input.order + 2];
    for(int i = 0; i < input.order; i++)
        dims[i + 1] = input.GetDim(i);
    dims[0] = nhead;
    dims[input.order + 1] = len;
    XTensor mask;
    InitTensor(&mask, input.order + 2, dims, X_FLOAT, padding.devID);

    /* a upper triangular matrix where the cells of the upper triangular are set to -1e-9.
        this matrix can be used to prevent the attention to current or following words in
        a given sequence. */
    _SetDataLowTri(&mask, 1e9F, 0);
    _ScaleAndShiftMe(&mask, 1.0F, -1e9F);
        
    int * dimsPadding = new int[padding.order + 2];
    for(int i = 0; i < padding.order - 1; i++)
        dimsPadding[i] = padding.GetDim(i);
    dimsPadding[padding.order - 1] = padding.GetDim(-1);
    dimsPadding[padding.order] = padding.GetDim(-1);

    XTensor * padding2 = NewTensorBuf(padding.order + 1, dimsPadding, padding.dataType,
                                        padding.devID);

    for(int i = 0; i < padding2->order; i++)
        dimsPadding[i + 1] = padding2->GetDim(i);
    dimsPadding[0] = nhead;

    //XTensor * padding3 = NewTensorBuf(padding.order + 2, dimsPadding, padding.dataType,
    //                                    padding.devID);
    //    
    ///* mask of the padding */
    //_Unsqueeze(&padding, padding2, padding.order - 1, padding.GetDim(-1));
    //_Unsqueeze(padding2, padding3, 0, nhead);
    //    
    //_ScaleAndShiftMe(padding3, 1e9F, -1e9F);
    //    
    ////_Sum(&mask, padding3, &mask);

    encoding = MakeEncoder(input, mask, isTraining);
    outputLayer->Make(encoding, output);

    delete[] dims;
    delete[] dimsPadding;
        
    //DelTensorBuf(padding3);
    DelTensorBuf(padding2);
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
void T2TModel::MakeMT(XTensor &inputEnc, XTensor &inputDec, XTensor &output, XTensor &paddingEnc, XTensor &paddingDec, bool isTraining)
{
    XTensor encoding;
    XTensor decoding;
    XTensor maskEnc;
    XTensor maskDec;
    XTensor maskEncDec;

    /* encoder mask */
    MakeMTMaskEnc(inputEnc, paddingEnc, maskEnc);
    
    /* decoder mask */
    MakeMTMaskDec(inputEnc, inputDec, paddingEnc, paddingDec, maskDec, maskEncDec);

    encoding = MakeEncoder(inputEnc, maskEnc, isTraining);

    decoding = MakeDecoder(inputDec, encoding, maskDec, maskEncDec, isTraining);

    outputLayer->Make(decoding, output);
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
void T2TModel::MakeMTMask(XTensor &inputEnc,   XTensor &inputDec, 
                          XTensor &paddingEnc, XTensor &paddingDec, 
                          XTensor &maskEnc,    XTensor &maskDec,    XTensor &maskEncDec)
{
    int len = inputDec.GetDim(inputDec.order - 1);
    int * dims = new int[inputDec.order + 2];
    for(int i = 0; i < inputDec.order; i++)
        dims[i + 1] = inputDec.GetDim(i);
    dims[0] = nhead;
    dims[inputDec.order + 1] = len;
    InitTensor(&maskDec, inputDec.order + 2, dims, X_FLOAT, paddingDec.devID);
        
    /* an upper triangular matrix where the cells of the upper triangular are set to -1e-9.
       this matrix can be used to prevent the attention to current or following words in
       a given sequence. */
    _SetDataLowTri(&maskDec, 1e9F, 0);
    _ScaleAndShiftMe(&maskDec, 1.0F, -1e9F);

    /* encoder-decoder mask that prevents the attention to padding dummy words */
    dims[inputDec.order + 1] = inputEnc.GetDim(inputEnc.order - 1);
    InitTensor(&maskEncDec, inputDec.order + 2, dims, X_FLOAT, paddingEnc.devID);

    XTensor * maskEncDecTMPEnc = NewTensorBuf(paddingEnc.order + 1, dims + 1, paddingEnc.dataType,
                                                paddingEnc.devID);
    XTensor * maskEncDecTMPDec = NewTensorBuf(maskEncDecTMPEnc, paddingEnc.devID);

    _Unsqueeze(&paddingEnc, maskEncDecTMPEnc, paddingEnc.order - 1, paddingDec.GetDim(-1));
    _ScaleAndShiftMe(maskEncDecTMPEnc, 1e9F, -1e9F);
    _Unsqueeze(maskEncDecTMPEnc, &maskEncDec, 0, dims[0]);

    DelTensorBuf(maskEncDecTMPDec);
    DelTensorBuf(maskEncDecTMPEnc);

    /* padding on the source side */
    int * dimsPadding = new int[paddingEnc.order + 2];
    for (int i = 0; i < paddingEnc.order - 1; i++)
        dimsPadding[i] = paddingEnc.GetDim(i);
    dimsPadding[paddingEnc.order - 1] = paddingEnc.GetDim(-1);
    dimsPadding[paddingEnc.order] = paddingEnc.GetDim(-1);

    XTensor * padding2 = NewTensorBuf(paddingEnc.order + 1, dimsPadding, paddingEnc.dataType,
                                        paddingEnc.devID);

    for (int i = 0; i < padding2->order; i++)
        dimsPadding[i + 1] = padding2->GetDim(i);
    dimsPadding[0] = nhead;

    XTensor * padding3 = NewTensorBuf(paddingEnc.order + 2, dimsPadding, paddingEnc.dataType,
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
void T2TModel::MakeMTMaskEnc(XTensor &inputEnc, XTensor &paddingEnc, XTensor &maskEnc)
{
    /* padding on the source side */
    int * dimsPadding = new int[paddingEnc.order + 2];
    for (int i = 0; i < paddingEnc.order - 1; i++)
        dimsPadding[i] = paddingEnc.GetDim(i);
    dimsPadding[paddingEnc.order - 1] = paddingEnc.GetDim(-1);
    dimsPadding[paddingEnc.order] = paddingEnc.GetDim(-1);
    
    XTensor * padding2 = NewTensorBuf(paddingEnc.order + 1, dimsPadding, paddingEnc.dataType,
                                        paddingEnc.devID);
    
    for (int i = 0; i < padding2->order; i++)
        dimsPadding[i + 1] = padding2->GetDim(i);
    dimsPadding[0] = nhead;
    
    XTensor * padding3 = NewTensorBuf(paddingEnc.order + 2, dimsPadding, paddingEnc.dataType,
                                        paddingEnc.devID);
    
    /* mask of the padding */
    _Unsqueeze(&paddingEnc, padding2, paddingEnc.order - 1, paddingEnc.GetDim(-1));
    _Unsqueeze(padding2, padding3, 0, nhead);
    
    _ScaleAndShiftMe(padding3, 1e9F, -1e9F);
    
    InitTensor(&maskEnc, padding3);
    maskEnc.SetZeroAll();
    
    /* generate the mask on the source language side (for padding) */
    _Sum(&maskEnc, padding3, &maskEnc);
    
    DelTensorBuf(padding3);
    DelTensorBuf(padding2);
    delete[] dimsPadding;
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
void T2TModel::MakeMTMaskDec(XTensor &inputEnc, XTensor &inputDec,
                             XTensor &paddingEnc, XTensor &paddingDec,
                             XTensor &maskDec, XTensor &maskEncDec)
{
    int len = inputDec.GetDim(inputDec.order - 1);
    int * dims = new int[inputDec.order + 2];
    for(int i = 0; i < inputDec.order; i++)
        dims[i + 1] = inputDec.GetDim(i);
    dims[0] = nhead;
    dims[inputDec.order + 1] = len;
    InitTensor(&maskDec, inputDec.order + 2, dims, X_FLOAT, paddingDec.devID);
    
    /* An upper triangular matrix where the cells of the upper triangular are set to -1e-9.
       This matrix can be used to block the attention to current or following words in
       a given sequence. */
    _SetDataLowTri(&maskDec, 1e9F, 0);

    //maskDec.Dump(stderr, "mask: ");

    _ScaleAndShiftMe(&maskDec, 1.0F, -1e9F);
    
    //maskDec.Dump(stderr, "mask: ");

    /* encoder-decoder mask that prevents the attention to padding dummy words */
    dims[inputDec.order + 1] = inputEnc.GetDim(inputEnc.order - 1);
    InitTensor(&maskEncDec, inputDec.order + 2, dims, X_FLOAT, paddingEnc.devID);
    
    XTensor * maskEncDecTMPEnc = NewTensorBuf(paddingEnc.order + 1, dims + 1, paddingEnc.dataType,
                                                paddingEnc.devID);
    XTensor * maskEncDecTMPDec = NewTensorBuf(maskEncDecTMPEnc, paddingEnc.devID);
    
    _Unsqueeze(&paddingEnc, maskEncDecTMPEnc, paddingEnc.order - 1, paddingDec.GetDim(-1));

    //paddingEnc.Dump(stderr, "paddingenc:");
    //maskEncDecTMPEnc->Dump(stderr, "maskencdectmpenc:");

    _ScaleAndShiftMe(maskEncDecTMPEnc, 1e9F, -1e9F);

    //maskEncDecTMPEnc->Dump(stderr, "maskencdectmpenc:");

    _Unsqueeze(maskEncDecTMPEnc, &maskEncDec, 0, dims[0]);

    //maskEncDecTMPEnc->Dump(stderr, "maskencdectmpenc:");
    
    DelTensorBuf(maskEncDecTMPDec);
    DelTensorBuf(maskEncDecTMPEnc);
    delete[] dims;
}
/* 
get parameter matrics
>> list - the list that keeps the parameter matrics
*/
void T2TModel::GetParams(TensorList &list)
{
    list.Clear();
    list.Add(&outputLayer->w);
    
    for(int i = 0; i < encoder->nlayer; i++){
        list.Add(&encoder->fnns[i].w1);
        list.Add(&encoder->fnns[i].b1);
        list.Add(&encoder->fnns[i].w2);
        list.Add(&encoder->fnns[i].b2);
        //list.Add(&encoder->attentions[i].wk);
        //list.Add(&encoder->attentions[i].wq);
        //list.Add(&encoder->attentions[i].wv);
        list.Add(&encoder->attentions[i].wbig);
        list.Add(&encoder->attentions[i].wa);
        list.Add(&encoder->fnnLayerNorms[i].w);
        list.Add(&encoder->fnnLayerNorms[i].b);
        list.Add(&encoder->attLayerNorms[i].w);
        list.Add(&encoder->attLayerNorms[i].b);
    }
    
    list.Add(&encoder->embedder.w);

    if(isMT){
        for(int i = 0; i < decoder->nlayer; i++){
            list.Add(&decoder->fnns[i].w1);
            list.Add(&decoder->fnns[i].b1);
            list.Add(&decoder->fnns[i].w2);
            list.Add(&decoder->fnns[i].b2);
            list.Add(&decoder->attentionsEnde[i].wk);
            list.Add(&decoder->attentionsEnde[i].wq);
            list.Add(&decoder->attentionsEnde[i].wv);
            list.Add(&decoder->attentionsEnde[i].wa);
            list.Add(&decoder->attEndeLayerNorms[i].w);
            list.Add(&decoder->attEndeLayerNorms[i].b);
            //list.Add(&decoder->attentions[i].wk);
            //list.Add(&decoder->attentions[i].wq);
            //list.Add(&decoder->attentions[i].wv);
            list.Add(&decoder->attentions[i].wbig);
            list.Add(&decoder->attentions[i].wa);
            list.Add(&decoder->fnnLayerNorms[i].w);
            list.Add(&decoder->fnnLayerNorms[i].b);
            list.Add(&decoder->attLayerNorms[i].w);
            list.Add(&decoder->attLayerNorms[i].b);
        }
        
        list.Add(&decoder->embedder.w);
    }
}

/*
dump the parameters 
>> fn - where to keep the model
>> model - the model
*/
void T2TModel::Dump(const char * fn)
{
    double startT = GetClockSec();

    FILE * file = fopen(fn, "wb");
    CheckNTErrors(file, "Cannot open the model file");

    TensorList params(100);

    GetParams(params);

    for(int i = 0; i < params.count; i++){
        XTensor * p = (XTensor*)params.Get(i);
        p->Dump(file, "param:");
    }

    fclose(file);

    double elapsed = GetClockSec() - startT;

    XPRINT1(0, stderr, "[INFO] model saved (took %.1fs)\n", elapsed);
}

/* read the parameters */
void T2TModel::Read(const char * fn)
{
    double startT = GetClockSec();

    FILE * file = fopen(fn, "rb");
    CheckNTErrors(file, "Cannot open the model file");

    TensorList params(100);

    GetParams(params);

    for(int i = 0; i < params.count; i++){
        XTensor * p = (XTensor*)params.Get(i);
        p->Read(file, "param:");
    }

    fclose(file);

    double elapsed = GetClockSec() - startT;

    XPRINT1(0, stderr, "[INFO] model loaded (took %.1fs)\n", elapsed);
}

}
