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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2019-03-27
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04, 2020-06
 */

#ifndef __SEARCHER_H__
#define __SEARCHER_H__

#include "../Model.h"
#include "Predictor.h"

using namespace std;

/* the nmt namespace */
namespace nmt
{

/* The class organizes the search process. It calls "predictors" to generate
   distributions of the predictions and prunes the search space by beam pruning.
   This makes a graph where each path represents a translation hypotheses.
   The output can be the path with the highest model score. */
class BeamSearch
{
private:
    /* the alpha parameter controls the length preference */
    float alpha;

    /* predictor */
    Predictor predictor;

    /* max length of the generated sequence */
    int maxLen;

    /* beam size */
    int beamSize;

    /* batch size */
    int batchSize;

    /* we keep the final hypotheses in a heap for each sentence in the batch. */
    XHeap<MIN_HEAP, float>* fullHypos;

    /* array of the end symbols */
    int* endSymbols;

    /* number of the end symbols */
    int endSymbolNum;

    /* start symbol */
    int startSymbol;

    /* scalar of the input sequence (for max number of search steps) */
    float scalarMaxLength;

    /* indicate whether the early stop strategy is used */
    bool isEarlyStop;

    /* pids for alive states */
    IntList aliveStatePids;

    /* alive sentences */
    IntList aliveSentList;

    /* whether we need to reorder the states */
    bool needReorder;

public:
    /* constructor */
    BeamSearch();

    /* de-constructor */
    ~BeamSearch();

    /* initialize the model */
    void Init(NMTConfig& config);

    /* search for the most promising states */
    void Search(NMTModel* model, XTensor& input, XTensor& padding, IntList** output, XTensor& score);

    /* preparation */
    void Prepare(int myBatchSize, int myBeamSize);

    /* compute the model score for each hypotheses */
    void Score(StateBundle* prev, StateBundle* beam);

    /* generate token indices via beam pruning */
    void Generate(StateBundle* prev, StateBundle* beam);

    /* expand the search graph */
    void Expand(StateBundle* prev, StateBundle* beam, XTensor& reorderState);

    /* collect hypotheses with ending symbol */
    void Collect(StateBundle* beam);

    /* fill the hypotheses heap with incomplete hypotheses */
    void FillHeap(StateBundle* beam);

    /* save the output sequences and score */
    void Dump(IntList** output, XTensor* score);

    /* check if the token is an end symbol */
    bool IsEnd(int token);

    /* check whether all hypotheses are completed */
    bool IsAllCompleted(StateBundle* beam);

    /* update the beam by pruning finished states */
    void RemoveFinishedStates(StateBundle* beam, XTensor& aliveEncoding,
        XTensor& aliveInput, XTensor& alivePadding, XTensor& aliveIdx);

    /* set end symbols for search */
    void SetEnd(const int* tokens, const int tokenNum);

    /* make a mask to prevent duplicated entries in beam expansion for the first position */
    XTensor MakeFirstMask(StateBundle* beam);
};

struct Timer 
{
    std::chrono::system_clock::time_point time;
    std::chrono::duration<double> duration;

    Timer() {
        duration = std::chrono::system_clock::now() - std::chrono::system_clock::now();
    }

    void ResetTime() {
        time = std::chrono::system_clock::now();
    }

    void Update() {
        duration += (std::chrono::system_clock::now() - time);
        ResetTime();

    }

    void Report(const string& prefix) {
        LOG("Duration of %s: %f", prefix.c_str(), duration.count());
    }
};

class GreedySearch
{
private:
    /* max length of the generated sequence */
    int maxLen;

    /* batch size */
    int batchSize;

    /* array of the end symbols */
    int* endSymbols;

    /* number of the end symbols */
    int endSymbolNum;

    /* start symbol */
    int startSymbol;

    /* scalar of the input sequence (for max number of search steps) */
    float scalarMaxLength;

public:
    /* timer for the encoder */
    Timer encoderTimer;

    /* timer for the decoder */
    Timer decoderTimer;

    /* constructor */
    GreedySearch();

    /* de-constructor */
    ~GreedySearch();

    /* initialize the model */
    void Init(NMTConfig& config);

    /* search for the most promising states */
    void Search(NMTModel* model, XTensor& input, XTensor& padding, IntList** outputs);

    /* preparation */
    void Prepare(int myBatchSize);

    /* check if the token is an end symbol */
    bool IsEnd(int token);

    /* set end symbols for search */
    void SetEnd(const int* tokens, const int tokenNum);
};

} /* end of the nmt namespace */

#endif /* __SEARCHER_H__ */