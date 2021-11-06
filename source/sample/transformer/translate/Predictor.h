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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2019-03-13
 * This is the first source file I create in 2019 - new start!
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04
 */

#ifndef __PREDICTOR_H__
#define __PREDICTOR_H__

#include "../Model.h"
#include "LengthPenalty.h"

using namespace std;

/* the nmt namespace */
namespace nmt
{

#define _PID_EMPTY -1

/* state for search. It keeps the path (back-pointer), prediction distribution,
   and etc. It can be regarded as a hypotheses in translation. */
class State
{
public:
    /* we assume that the prediction is an integer */
    int prediction;

    /* id of the problem. One can regard it as the sentence id when we
       translate a number of sentences in the batched manner. The hypotheses
       is empty if id = -1 */
    int pid;

    /* indicates whether the state is an end */
    bool isEnd;

    /* indicates whether the state is the start */
    bool isStart;

    /* indicates whether the state is completed */
    bool isCompleted;

    /* probability of every prediction (last state of the path) */
    float prob;

    /* probability of every path */
    float probPath;

    /* model score of every path. A model score = path probability + some other stuff */
    float modelScore;

    /* number of steps we go over so far */
    int nstep;

    /* pointer to the previous state */
    State* last;
};

/* a bundle of states */
class StateBundle
{
public:
    /* predictions */
    XTensor prediction;

    /* id of the previous state that generates the current one  */
    XTensor preID;

    /* mark that indicates whether each hypotheses is completed */
    XTensor endMark;

    /* probability of every prediction (last state of the path) */
    XTensor prob;

    /* probability of every path */
    XTensor probPath;

    /* model score of every path */
    XTensor modelScore;

    /* step number of each hypotheses */
    float nstep;

    /* list of states */
    State* states;

    /* number of states */
    int stateNum;

    /* indicates whether it is the first state */
    bool isStart;

public:
    /* constructor */
    StateBundle();

    /* de-constructor */
    ~StateBundle();

    /* create states */
    void MakeStates(int num);
};

/* The predictor reads the current state and then predicts the next.
   It is exactly the same procedure of MT inference -
   we get the state of previous words and then generate the next word.
   Here, a state can be regarded as the representation of words (word
   indices, hidden states, embeddings and etc.).  */
class Predictor
{
private:
    /* pointer to the transformer model */
    NMTModel* m;

    /* current state */
    StateBundle* s;

    /* start symbol */
    int startSymbol;

    /* end symbol */
    int endSymbol;

public:
    /* constructor */
    Predictor();

    /* de-constructor */
    ~Predictor();

    /* create an initial state */
    void Create(NMTModel* model, XTensor* top, const XTensor* input, int beamSize, StateBundle* state);

    /* set the start symbol */
    void SetStartSymbol(int symbol);

    /* read a state */
    void Read(NMTModel* model, StateBundle* state);

    /* predict the next state */
    void Predict(StateBundle* next, XTensor& aliveIndices, XTensor& encoding,
        XTensor& inputEnc, XTensor& paddingEnc, int rawBatchSize,
        bool isStart, XTensor& reorderState, bool needReorder, int nstep);

    /* generate paths up to the states of the current step */
    XTensor GeneratePaths(StateBundle* state);

    /* get the predictions of the previous step */
    XTensor GetLastPrediction(StateBundle* state, int devID);
};

} /* end of the nmt namespace */

#endif /* __PREDICTOR_H__ */
