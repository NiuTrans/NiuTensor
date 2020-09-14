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
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04, 2020-06
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>

#include "T2TUtility.h"
#include "../../../tensor/XGlobal.h"

using namespace nts;
using namespace std;

namespace transformer
{

/*
load configurations from the command
>> argc - number of arguments
>> argv - the list of arguments
*/
T2TConfig::T2TConfig(int argc, const char** argv)
{
    char** args = new char* [MAX_PARAM_NUM];
    for (int i = 0; i < argc; i++) {
        args[i] = new char[strlen(argv[i]) + 1];
        strcpy(args[i], argv[i]);
    }

    char* configFN = new char[1024];
    LoadParamString(argc, args, "config", configFN, "");

    int argsNum = argc;

    /* load configurations from a file */
    if (strcmp(configFN, "") != 0)
        argsNum = LoadFromFile(configFN, args);

    ShowParams(argsNum, args);

    /* options for the model */
    LoadParamInt(argsNum, args, "nhead", &nhead, 8);
    LoadParamInt(argsNum, args, "enclayer", &nEncLayer, 1);
    LoadParamInt(argsNum, args, "declayer", &nDecLayer, 1);
    LoadParamInt(argsNum, args, "maxrp", &maxRP, 8);
    LoadParamInt(argsNum, args, "embsize", &embSize, 256);
    LoadParamInt(argsNum, args, "modelsize", &modelSize, 256);
    LoadParamInt(argsNum, args, "maxpos", &maxPosLen, 1024);
    LoadParamInt(argsNum, args, "fnnhidden", &fnnHiddenSize, modelSize * 4);
    LoadParamInt(argsNum, args, "vsize", &srcVocabSize, 10000);
    LoadParamInt(argsNum, args, "vsizetgt", &tgtVocabSize, 10000);
    LoadParamInt(argsNum, args, "padid", &padID, 1);
    LoadParamInt(argsNum, args, "startid", &startID, 2);
    LoadParamInt(argsNum, args, "endid", &endID, 2);
    LoadParamBool(argsNum, args, "rpr", &useRPR, false);
    LoadParamBool(argsNum, args, "prenorm", &preNorm, false);
    LoadParamString(argsNum, args, "model", modelFN, "model.bin");
    LoadParamString(argsNum, args, "srcvocab", srcVocabFN, "vocab.src");
    LoadParamString(argsNum, args, "tgtvocab", tgtVocabFN, "vocab.tgt");

    /* options for training */
    LoadParamString(argsNum, args, "train", trainFN, "");
    LoadParamString(argsNum, args, "valid", validFN, "");
    LoadParamInt(argsNum, args, "dev", &devID, 0);
    LoadParamInt(argsNum, args, "wbatch", &wBatchSize, 2048);
    LoadParamInt(argsNum, args, "sbatch", &sBatchSize, 1);
    isTraining = (strcmp(trainFN, "") == 0) ? false : true;
    LoadParamBool(argsNum, args, "mt", &isMT, true);
    LoadParamFloat(argsNum, args, "dropout", &dropout, 0.1);
    LoadParamFloat(argsNum, args, "fnndrop", &fnnDropout, 0.0);
    LoadParamFloat(argsNum, args, "attdrop", &attDropout, 0.0);

    LoadParamFloat(argc, args, "lrate", &lrate, 1.0F);
    LoadParamFloat(argc, args, "lrbias", &lrbias, 0);
    LoadParamInt(argc, args, "nepoch", &nepoch, 20);
    LoadParamInt(argc, args, "nstep", &nstep, 100000);
    LoadParamInt(argc, args, "nwarmup", &nwarmup, 3000);
    LoadParamBool(argc, args, "adam", &useAdam, true);
    LoadParamFloat(argc, args, "adambeta1", &adamBeta1, 0.9F);
    LoadParamFloat(argc, args, "adambeta2", &adamBeta2, 0.98F);
    LoadParamFloat(argc, args, "adamdelta", &adamDelta, 1e-9F);
    LoadParamBool(argc, args, "shuffled", &isShuffled, true);
    LoadParamFloat(argc, args, "labelsmoothing", &labelSmoothingP, 0.1);
    LoadParamInt(argc, args, "nstepcheckpoint", &nStepCheckpoint, -1);
    LoadParamBool(argc, args, "epochcheckpoint", &useEpochCheckpoint, false);
    LoadParamInt(argc, args, "updatestep", &updateStep, 1);
    LoadParamBool(argc, args, "debug", &isDebugged, false);
    LoadParamBool(argc, args, "sorted", &isLenSorted, false);

    LoadParamInt(argc, args, "bufsize", &bufSize, 50000);
    LoadParamBool(argc, args, "doubledend", &isDoubledEnd, false);
    LoadParamBool(argc, args, "smallbatch", &isSmallBatch, true);
    LoadParamBool(argc, args, "bigbatch", &isBigBatch, false);
    LoadParamBool(argc, args, "randbatch", &isRandomBatch, false);
    LoadParamInt(argc, args, "bucketsize", &bucketSize, 0);

    /* options for translating */
    LoadParamString(argsNum, args, "test", testFN, "");
    LoadParamString(argsNum, args, "output", outputFN, "");
    LoadParamInt(argsNum, args, "beamsize", &beamSize, 1);
    LoadParamBool(argsNum, args, "fp16", &useFP16, false);
    LoadParamFloat(argsNum, args, "lenalpha", &lenAlpha, 0.6);
    LoadParamFloat(argsNum, args, "maxlenalpha", &maxLenAlpha, 2.0);

    for (int i = 0; i < argc; i++)
        delete[] args[i];
    delete[] args;
    delete[] configFN;
}

/*
load configurations from a file
>> configFN - path to the configuration file
>> args - the list to store the configurations
format: one option per line, separated by a blank or a tab
*/
int T2TConfig::LoadFromFile(const char* configFN, char** args) {
    ifstream f(configFN, ios::in);
    CheckNTErrors(f.is_open(), "unable to open the config file");

    int argsNum = 0;

    /* parse arguments */
    string key, value;
    while (f >> key >> value) {
        key += '-';
        strcpy(args[argsNum++], key.c_str());
        strcpy(args[argsNum++], value.c_str());
    }

    /* record the number of arguments */
    return argsNum;
}

void LoadParamString(int argc, char** argv, const char* name, char* p, const char* defaultP)
{
    char vname[128];
    vname[0] = '-';
    strcpy(vname + 1, name);
    bool hit = false;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], vname) && i + 1 < argc) {
            strcpy(p, argv[i + 1]);
            hit = true;
            break;
        }
    }
    if (!hit)
        strcpy(p, defaultP);
}

void LoadParamInt(int argc, char** argv, const char* name, int* p, int defaultP)
{
    char vname[128];
    vname[0] = '-';
    strcpy(vname + 1, name);
    bool hit = false;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], vname) && i + 1 < argc) {
            *(int*)p = atoi(argv[i + 1]);
            hit = true;
            break;
        }
    }
    if (!hit)
        *p = defaultP;
}

void LoadParamBool(int argc, char** argv, const char* name, bool* p, bool defaultP)
{
    char vname[128];
    vname[0] = '-';
    strcpy(vname + 1, name);
    bool hit = false;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], vname)) {
            *(bool*)p = true;
            hit = true;
            break;
        }
    }
    if (!hit)
        *p = defaultP;
}

void LoadParamFloat(int argc, char** argv, const char* name, float* p, float defaultP)
{
    char vname[128];
    vname[0] = '-';
    strcpy(vname + 1, name);
    bool hit = false;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], vname) && i + 1 < argc) {
            *p = (float)atof(argv[i + 1]);
            hit = true;
            break;
        }
    }
    if (!hit)
        *p = defaultP;
}

void ShowParams(int argc, char** argv)
{
    fprintf(stderr, "args:\n");
    for (int i = 0; i < argc; i++) {
        if (argv[i][1] == 0)
            continue;
        if (argv[i][0] == '-' && (argv[i][1] < '1' || argv[i][1] > '9')) {
            if (i + 1 < argc && argv[i + 1][0] != '-')
                fprintf(stderr, " %s=%s\n", argv[i], argv[i + 1]);
            else
                fprintf(stderr, " %s=yes\n", argv[i]);
        }
    }
    fprintf(stderr, "\n");
}

#define MAX_WORD_NUM 120

/*
split string by delimiter, this will return indices of all sub-strings
>> s - the original string
>> delimiter - as it is
<< indices - indices of all sub-strings
*/
UInt64List SplitToPos(const string& s, const string& delimiter)
{
    UInt64List indices;
    if (delimiter.length() == 0) {
        indices.Add(0);
    }
    size_t pos = 0;
    uint64_t start = 0;
    while ((pos = s.find(delimiter, start)) != string::npos) {
        if (pos != start) {
            indices.Add(start);
        }
        start = pos + delimiter.length();
    }
    if (start != s.length()) {
        indices.Add(start);
    }
    return indices;
}

/* split a string to a int64_t list */
IntList SplitInt(const string& s, const string& delimiter)
{
    IntList values;
    auto indices = SplitToPos(s, delimiter);
    for (int i = 0; i < indices.Size(); i++) {
        values.Add(strtol(s.data() + indices[i], nullptr, 10));
    }
    return values;
}

/* split a string to a float list */
FloatList SplitFloat(const string& s, const string& delimiter)
{
    FloatList values;
    auto indices = SplitToPos(s, delimiter);
    for (int i = 0; i < indices.Size(); i++) {
        values.Add(strtof(s.data() + indices[i], nullptr));
    }
    return values;
}

}