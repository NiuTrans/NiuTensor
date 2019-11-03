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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-08-01
 */

#include <math.h>
#include "T2TEmbedding.h"
#include "T2TUtility.h"
#include "../../tensor/core/CHeader.h"

namespace transformer
{

/* constructor */
T2TEmbedder::T2TEmbedder()
{
    devID = -1;
    vSize = -1;
    maxLength = -1;
}

/* deconstructor */
T2TEmbedder::~T2TEmbedder()
{
}

/* 
initialize the model 
>> argc - number of arguments
>> argv - list of pointers to the arguments
>> myDevID - device id
*/
void T2TEmbedder::InitModel(int argc, char ** argv, int myDevID, bool isEnc)
{
    devID = myDevID;
    
    if(isEnc){
        LoadParamInt(argc, argv, "vsize", &vSize, -1);
    }
    else{
        LoadParamInt(argc, argv, "vsizetgt", &vSize, -1);
    }
    //LoadParamInt(argc, argv, "vsize", &vSize, -1);
    LoadParamInt(argc, argv, "maxlen", &maxLength, 512);
    LoadParamInt(argc, argv, "d", &eSize, DEFAULT_EMBEDDING_SIZE);
    LoadParamInt(argc, argv, "d", &d, DEFAULT_EMBEDDING_SIZE);

    InitTensor2D(&w, vSize, eSize, X_FLOAT, devID);

    DTYPE v = 1.0F/(float)sqrt((float)eSize);
    w.SetDataRandn(0, v);

    /* create the positional embedding matrix */
    MakePosEmbedding(eSize, d, maxLength);
}

/* 
make positional embeddings (of size eSize * length)
>> eSize - embedding size
>> d - dimension size of the hidden layers
>> length - length of the sequence
*/
void T2TEmbedder::MakePosEmbedding(int eSize, int d, int length)
{
    InitTensor2D(&posEmbeddingBase, length, eSize, X_FLOAT, devID);

    float * data = new float[posEmbeddingBase.unitNum];

    for(int pos = 0; pos < length; pos++){
        float * dp = data + pos * eSize;
        
        int channelSize = eSize / 2;
        int offset = 0;
        for(int i = 0; i < channelSize; i++){
            dp[offset++] = (float)sin(pos/pow(10000.0F, 2.0F*i/(d - 2)));
        }
        for(int i = 0; i < channelSize; i++){
            dp[offset++] = (float)cos(pos/pow(10000.0F, 2.0F*i/(d - 2)));
        }

        /*
        for(int k = 0; k < eSize; k++){
            if(k % 2 == 0){
                int i = k/2;
                dp[k] = (float)sin(pos/pow(10000.0F, 2.0F*i/d));
            }
            else{
                int i = (k - 1)/2;
                dp[k] = (float)cos(pos/pow(10000.0F, 2.0F*i/d));
            }
        }
        */
    }

    posEmbeddingBase.SetData(data, posEmbeddingBase.unitNum);

    delete[] data;
}

/* 
make the network 
*/
XTensor T2TEmbedder::Make(XTensor &input)
{
    //CheckNTErrors(input.GetDim(-1) == vSize, "Wrong vocabulary size!");
    CheckNTErrors(input.order > 1, "Wrong input tensor size!");
    CheckNTErrors(input.dimSize[input.order - 1] < maxLength, "The sequence is too long!");
    CheckNTErrors(vSize > 0, "set vocabulary size by \"-vsize\"");
    CheckNTErrors(eSize > 0, "set embedding size by \"-esize\"");

    int dims[MAX_TENSOR_DIM_NUM];
    memcpy(dims, input.dimSize, input.order * sizeof(int));
    dims[input.order] = eSize;

    XTensor wordEmbedding;
    XTensor posEmbedding;

    bool match = (posEmbedding.order == input.order);
    if(match){
        for(int i = 0; i < input.order; i++){
            if(dims[i] != posEmbedding.GetDim(i))
                match = false;
        }
    }

    /* we make positional embeddings first */
    //if(!match){
    if(true){
        InitTensor(&posEmbedding, input.order + 1, dims, X_FLOAT, devID);

        XTensor * posTMP = NewTensorBuf(2, dims + 1, X_FLOAT, devID);

        _CopyValues(&posEmbeddingBase, 0, posTMP->unitNum, posTMP, 0);
        _Unsqueeze(posTMP, &posEmbedding, 0, dims[0]);

        DelTensorBuf(posTMP);
    }

    /* then we make word embeddings */
    wordEmbedding = Gather(w, input);
    wordEmbedding = Linear(wordEmbedding, (float)sqrt((float)eSize));

    /* we sum over the two embeddings */
    return wordEmbedding + posEmbedding;
}

}
