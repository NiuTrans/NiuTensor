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
 * $Created by: HU Chi (huchinlp@gmail.com) 2021-06
 */

#include <fstream>
#include "Config.h"

using namespace nts;
using namespace std;

/* the nmt namespace */
namespace nmt
{

/*
load configurations from the command
>> argc - number of arguments
>> argv - the list of arguments
*/
NMTConfig::NMTConfig(int argc, const char** argv)
{
    char** args = new char* [MAX_PARAM_NUM];
    for (int i = 0; i < argc; i++) {
        args[i] = new char[strlen(argv[i]) + 1];
        strcpy(args[i], argv[i]);
    }
    for (int i = argc; i < MAX_PARAM_NUM; i++) {
        args[i] = NULL;
    }

    char* configFN = new char[1024];
    LoadParamString(argc, args, "config", configFN, "");

    int argsNum = argc;

    /* override the configuration according to the file content */
    if (strcmp(configFN, "") != 0)
        argsNum = LoadFromFile(configFN, args);

    ShowParams(argsNum, args);

    /* parse configuration in args */
    model.Load(argsNum, (const char **)args);
    common.Load(argsNum, (const char **)args);
    training.Load(argsNum, (const char **)args);
    translation.Load(argsNum, (const char **)args);

    for (int i = 0; i < MAX(argc, argsNum); i++)
        delete[] args[i];
    delete[] args;
    delete[] configFN;
}

/*
load configurations from a file
>> configFN - path to the configuration file
>> args - the list to store the configurations
<< argsNum - the number of arguments
format: one option per line, separated by a blank or a tab
*/
int NMTConfig::LoadFromFile(const char* configFN, char** args) 
{
    ifstream f(configFN, ios::in);
    CheckNTErrors(f.is_open(), "Failed to open the config file");

    int argsNum = 0;

    /* parse arguments from the file */
    string key, value;
    while (f >> key >> value && argsNum < (MAX_PARAM_NUM - 1)) {
        if (args[argsNum] != NULL) {
            delete[] args[argsNum];
        }
        if (args[argsNum + 1] != NULL) {
            delete[] args[argsNum + 1];
        }
        args[argsNum] = new char[1024];
        args[argsNum + 1] = new char[1024];
        strcpy(args[argsNum++], key.c_str());
        strcpy(args[argsNum++], value.c_str());
    }

    /* record the number of arguments */
    return argsNum;
}

/* load model configuration from the command */
void ModelConfig::Load(int argsNum, const char** args)
{
    Create(argsNum, args);

    LoadBool("bigatt", &useBigAtt, false);
    LoadBool("encprenorm", &encPreLN, true);
    LoadBool("decprenorm", &decPreLN, true);
    LoadBool("encl1norm", &encoderL1Norm, false);
    LoadBool("decl1norm", &decoderL1Norm, false);
    LoadBool("decoderonly", &decoderOnly, false);
    LoadBool("enchistory", &useEncHistory, false);
    LoadBool("dechistory", &useDecHistory, false);
    LoadBool("encfinalnorm", &encFinalNorm, true);
    LoadBool("decfinalnorm", &decFinalNorm, true);
    LoadBool("shareencdec", &shareEncDecEmb, false);
    LoadBool("sharedec", &shareDecInputOutputEmb, false);

    LoadInt("pad", &pad, -1);
    LoadInt("sos", &sos, -1);
    LoadInt("eos", &eos, -1);
    LoadInt("unk", &unk, -1);
    LoadInt("encemb", &encEmbDim, 512);
    LoadInt("decemb", &decEmbDim, 512);
    LoadInt("maxsrc", &maxSrcLen, 200);
    LoadInt("maxtgt", &maxTgtLen, 200);
    LoadInt("enclayer", &encLayerNum, 6);
    LoadInt("declayer", &decLayerNum, 6);
    LoadInt("maxrp", &maxRelativeLength, -1);
    LoadInt("encffn", &encFFNHiddenDim, 1024);
    LoadInt("decffn", &decFFNHiddenDim, 1024);
    LoadInt("srcvocabsize", &srcVocabSize, -1);
    LoadInt("tgtvocabsize", &tgtVocabSize, -1);
    LoadInt("encheads", &encSelfAttHeadNum, 4);
    LoadInt("decheads", &decSelfAttHeadNum, 4);
    LoadInt("encdecheads", &encDecAttHeadNum, 4);

    LoadFloat("dropout", &dropout, 0.3F);
    LoadFloat("ffndropout", &ffnDropout, 0.1F);
    LoadFloat("attdropout", &attDropout, 0.1F);
}

/* load training configuration from the command */
void TrainingConfig::Load(int argsNum, const char **args)
{
    Create(argsNum, args);

    LoadString("train", trainFN, "");
    LoadString("valid", validFN, "");

    LoadBool("adam", &useAdam, true);

    LoadInt("nepoch", &nepoch, 50);
    LoadInt("nstep", &nstep, 100000);
    LoadInt("savefreq", &saveFreq, 10000);
    LoadInt("nwarmup", &nwarmup, 8000);
    LoadInt("updatefreq", &updateFreq, 1);
    LoadInt("ncheckpoint", &ncheckpoint, 10);
    
    LoadFloat("lrbias", &lrbias, 0);
    LoadFloat("lrate", &lrate, 0.0015F);
    LoadFloat("minlr", &minLR, 1e-9F);
    LoadFloat("warmupinitlr", &warmupInitLR, 1e-7F);
    LoadFloat("adambeta1", &adamBeta1, 0.9F);
    LoadFloat("adambeta2", &adamBeta2, 0.98F);
    LoadFloat("adamdelta", &adamDelta, 1e-9F);
    LoadFloat("labelsmoothing", &labelSmoothingP, 0.1F);
    LoadFloat("weightdecay", &weightDecay, 0.0F);
    isTraining = (strcmp(trainFN, "") == 0) ? false : true;
    incremental = false;
}

/* load training configuration from the command */
void TranslationConfig::Load(int argsNum, const char** args)
{
    Create(argsNum, args);
    LoadString("input", inputFN, "");
    LoadString("output", outputFN, "");
    LoadInt("beam", &beamSize, 1);
    LoadInt("maxlen", &maxLen, 200);
    LoadFloat("lenalpha", &lenAlpha, 0.6F);
    LoadFloat("maxlenalpha", &maxLenAlpha, 1.25F);
}

/* load training configuration from the command */
void CommonConfig::Load(int argsNum, const char** args)
{
    Create(argsNum, args);
    LoadString("model", modelFN, "model.bin");
    LoadString("srcvocab", srcVocabFN, "");
    LoadString("tgtvocab", tgtVocabFN, "");
    LoadInt("seed", &seed, 1);
    LoadInt("dev", &devID, -1);
    LoadInt("sbatch", &sBatchSize, 32);
    LoadInt("wbatch", &wBatchSize, 4096);
    LoadInt("bufsize", &bufSize, 2000000);
    LoadInt("bucketsize", &bucketSize, wBatchSize);
    LoadInt("loginterval", &logInterval, 100);
    LoadBool("fp16", &useFP16, false);
}

/* 
split string into sub-strings by a delimiter
>> s - the original string
>> delimiter - as it is
>> maxNum - the maximum number of sub-strings
<< substrings - all sub-strings
*/
vector<string> SplitString(const string& s, const string& delimiter, int maxNum)
{
    CheckNTErrors(delimiter.length() > 0, "Invalid delimiter");
    
    vector<string> substrings;
    size_t pos = 0;
    size_t start = 0;
    while ((pos = s.find(delimiter, start)) != string::npos) {
        if (pos != start) {
            substrings.emplace_back(s.substr(start, pos - start));
        }
        start = pos + delimiter.length();
        if (substrings.size() == maxNum)
            break;
    }
    if (start != s.length() && substrings.size() < maxNum) {
        substrings.emplace_back(s.substr(start, s.length() - start));
    }
    return substrings;
}

} /* end of the nmt namespace */