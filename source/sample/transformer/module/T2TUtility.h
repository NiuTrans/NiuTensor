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
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-06
 */

#ifndef __T2TUTILITY_H__
#define __T2TUTILITY_H__

#include <string>
#include <cstdio>

#include "../../../tensor/XList.h"

using namespace std;
using namespace nts;

namespace transformer
{

#define MAX_PARAM_NUM 100

/* load arguments */
void LoadParamInt(int argc, char** argv, const char* name, int* p, int defaultP);
void LoadParamBool(int argc, char** argv, const char* name, bool* p, bool defaultP);
void LoadParamFloat(int argc, char** argv, const char* name, float* p, float defaultP);
void LoadParamString(int argc, char** argv, const char* name, char* p, const char* defaultP);

/* show arguments */
void ShowParams(int argc, char** argv);

/* split string */
IntList SplitInt(const string& s, const string& delimiter);
FloatList SplitFloat(const string& s, const string& delimiter);
UInt64List SplitToPos(const string& s, const string& delimiter);

/* configurations for t2t */
class T2TConfig {
public:
    /* path to the model */
    char modelFN[1024];

    /* path to the source vocab */
    char srcVocabFN[1024];

    /* path to the target vocab */
    char tgtVocabFN[1024];

    /* path to the input file (for inference) */
    char testFN[1024];

    /* path to the output file (for inference) */
    char outputFN[1024];

    /* path to the training file */
    char trainFN[1024];

    /* path to the validation file */
    char validFN[1024];

    /* device id */
    int devID;

    /* beam size */
    int beamSize;

    /* word batch size */
    int wBatchSize;

    /* sentence batch size */
    int sBatchSize;

    /* number of heads in attention */
    int nhead;

    /* number of encoder layers */
    int nEncLayer;

    /* number of decoder layers */
    int nDecLayer;

    /* the maximum relative position in RPR attentions */
    int maxRP;

    /* the dimension of embeddings */
    int embSize;

    /* the dimension of hidden layer */
    int modelSize;

    /* the maximum length in positional embedding */
    int maxPosLen;

    /* the dimension of fnn hidden layer */
    int fnnHiddenSize;

    /* the vocab size of source sequence */
    int srcVocabSize;

    /* the vocab size of target sequence */
    int tgtVocabSize;

    /* the padding id */
    int padID;

    /* start symbol */
    int startID;

    /* end symbol */
    int endID;

    /* indicates whether the model uses pre-norm */
    bool preNorm;

    /* indicates whether the model is running for machine translation */
    bool isMT;

    /* indicates whether the model is running with FP16 data type */
    bool useFP16;

    /* indicates whether we use the RPR attention */
    bool useRPR;

    /* indicates whether we train the model */
    bool isTraining;

    /* dropout rate for the model */
    float dropout;

    /* dropout rate for fnn layers */
    float fnnDropout;

    /* dropout rate for attention layers */
    float attDropout;

    /* the alpha parameter controls the length preference */
    float lenAlpha;

    /* scalar of the input sequence (for max number of search steps) */
    float maxLenAlpha;

    /* learning rate */
    float lrate;

    /* the parameter that controls the maximum learning rate in training */
    float lrbias;

    /* training epoch number */
    int nepoch;

    /* traing step number */
    int nstep;

    /* indicates whether we use Adam */
    bool useAdam;

    /* hyper parameters of Adam */
    float adamBeta1;
    float adamBeta2;
    float adamDelta;

    /* step number of warm-up for training */
    int nwarmup;

    /* indicates whether the data file is shuffled for training */
    bool isShuffled;

    /* the factor of label smoothing */
    float labelSmoothingP;

    /* number of steps after which we make a checkpoint */
    int nStepCheckpoint;

    /* indicates whether we make a checkpoint after each training epoch */
    bool useEpochCheckpoint;

    /* number of batches on which we do model update */
    int updateStep;

    /* indicates whether we intend to debug the net */
    bool isDebugged;

    /* indicates whether the sequence is sorted by length */
    bool isLenSorted;

    /* buffer size */
    int bufSize;

    /* indicates whether we double the </s> symbol for the output of LM */
    bool isDoubledEnd;

    /* indicates whether we use batchsize = max * sc
       rather rather than batchsize = word-number, where max is the maximum
       length and sc is the sentence number */
    bool isSmallBatch;

    /* counterpart of "isSmallBatch" */
    bool isBigBatch;

    /* randomize batches */
    bool isRandomBatch;

    /* bucket size */
    int bucketSize;

public:

    /* load configurations from the command */
    T2TConfig(int argc, const char** argv);

    /* load configurations from a file */
    int LoadFromFile(const char* configFN, char** args);
};

}

#endif
