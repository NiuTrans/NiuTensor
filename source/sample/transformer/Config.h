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
 * We define all options of the NMT system here, including
 * the configuration of model, training, translation, and
 * other common options.
 * 
 * $Created by: HU Chi (huchinlp@gmail.com) 2021-06
 */

#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <chrono>
#include <vector>
#include <string>
#include "../../tensor/XConfig.h"

using namespace std;
using namespace nts;

/* the nmt namespace */
namespace nmt
{

#define MAX_PATH_LEN 1024

/* training configuration */
class TrainingConfig : public XConfig
{
public:
    /* path to the training file */
    char trainFN[MAX_PATH_LEN];

    /* path to the validation file */
    char validFN[MAX_PATH_LEN];

    /* train the model or not */
    bool isTraining;

    /* incremental train the model or not */
    bool incremental;

    /* learning rate */
    float lrate;

    /* the initial learning rate for warm-up */
    float warmupInitLR;

    /* the minimum learning rate for training */
    float minLR;

    /* the parameter that controls the maximum learning rate in training */
    float lrbias;

    /* the maximum number of training epochs */
    int nepoch;

    /* the maximum number of training steps */
    int nstep;

    /* update parameters every N steps */
    int updateFreq;

    /* save a checkpoint every N steps */
    int saveFreq;

    /* the maximum number of checkpoints to keep */
    int ncheckpoint;

    /* indicates whether we use Adam */
    bool useAdam;

    /* hyper parameters of Adam */
    float adamBeta1;
    float adamBeta2;
    float adamDelta;

    /* the weight decay factor */
    float weightDecay;

    /* step number of warm-up for training */
    int nwarmup;

    /* the factor of label smoothing */
    float labelSmoothingP;

public:
    /* load configuration from the command */
    void Load(int argsNum, const char** args);
};

/* translation configuration */
class TranslationConfig : public XConfig 
{
public:
    /* path to the input file (for inference) */
    char inputFN[MAX_PATH_LEN];

    /* path to the output file (for inference) */
    char outputFN[MAX_PATH_LEN];

    /* beam size */
    int beamSize;

    /* the alpha parameter controls the length preference */
    float lenAlpha;

    /* scalar of the input sequence (for max number of search steps) */
    float maxLenAlpha;

    /* max length of the generated sequence */
    int maxLen;

public:
    /* load configuration from the command */
    void Load(int argsNum, const char** args);
};

/* model configuration */
class ModelConfig : public XConfig 
{
public:
    /* indicates whether the encoder uses L1-Norm */
    bool encoderL1Norm;

    /* indicates whether the decoder uses L1-Norm */
    bool decoderL1Norm;

    /* indicates whether qkv weights are continuous */
    bool useBigAtt;

    /* indicates whether the model contains a encoder */
    bool decoderOnly;

    /* indicates whether use layer history for the encoder */
    bool useEncHistory;

    /* indicates whether use layer history for the decoder */
    bool useDecHistory;

    /* the vocab size of source language */
    int srcVocabSize;

    /* the maximum length of a source sentence */
    int maxSrcLen;

    /* the number of heads in the encoder self-attention */
    int encSelfAttHeadNum;

    /* the dimension of the encoder embedding */
    int encEmbDim;

    /* the dimension of the encoder FFN hidden layer */
    int encFFNHiddenDim;

    /* the number of encoder layers */
    int encLayerNum;

    /* if set, use Pre-LN for the encoder, else use Post-LN */
    bool encPreLN;

    /* the vocab size of the target language */
    int tgtVocabSize;

    /* the maximum length of a target sentence */
    int maxTgtLen;

    /* the number of heads in the decoder self-attention */
    int decSelfAttHeadNum;

    /* the number of heads in the encoder-decoder-attention */
    int encDecAttHeadNum;

    /* the dimension of the decoder embedding */
    int decEmbDim;

    /* the dimension of the decoder FFN hidden layer */
    int decFFNHiddenDim;

    /* the number of decoder layers */
    int decLayerNum;

    /* if set, use Pre-LN for the decoder, else use Post-LN */
    bool decPreLN;

    /* the maximum relative position in RPR attentions */
    int maxRelativeLength;

    /* the padding id */
    int pad;

    /* the unk id */
    int unk;

    /* start symbol */
    int sos;

    /* end symbol */
    int eos;

    /* if set, add layer norm to the encoder output */
    bool encFinalNorm;

    /* if set, add layer norm to the decoder output */
    bool decFinalNorm;

    /* indicates whether share encoder and decoder embeddings */
    bool shareEncDecEmb;

    /* indicates whether share decoder embeddings and output weights */
    bool shareDecInputOutputEmb;

    /* dropout rate of the model */
    float dropout;

    /* dropout rate of FFN hidden layers */
    float ffnDropout;

    /* dropout rate of attentions */
    float attDropout;

public:
    /* load configuration from the command */
    void Load(int argsNum, const char** args);
};

/* common configuration */
class CommonConfig : public XConfig 
{
public:
    /* random seed */
    int seed;

    /* device id, >=0 for the GPU, -1 for the CPU */
    int devID;

    /* path to the model */
    char modelFN[MAX_PATH_LEN];

    /* path to the source vocabulary file */
    char srcVocabFN[MAX_PATH_LEN];

    /* path to the target vocabulary file */
    char tgtVocabFN[MAX_PATH_LEN];

    /* interval step for logging */
    int logInterval;

    /* word batch size */
    int wBatchSize;

    /* sentence batch size */
    int sBatchSize;

    /* buffer size for batches */
    int bufSize;

    /* bucket size for batches */
    int bucketSize;

    /* indicates whether the model is running with FP16 data type */
    bool useFP16;

public:
    /* load configuration from the command */
    void Load(int argsNum, const char** args);
};

/* configuration of the nmt project  */
class NMTConfig {
public:
    /* model configuration */
    ModelConfig model;

    /* common configuration */
    CommonConfig common;
    
    /* training configuration */
    TrainingConfig training;
    
    /* translation configuration */
    TranslationConfig translation;

public:
    /* load configuration from the command */
    NMTConfig(int argc, const char** argv);

    /* load configuration from a file */
    int LoadFromFile(const char* configFN, char** args);
};

/* split string by a delimiter */
vector<string> SplitString(const string& s, const string& delimiter, int maxNum);

} /* end of the nmt namespace */

#endif /* __CONFIG_H__ */