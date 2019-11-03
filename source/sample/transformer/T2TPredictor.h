/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2019, Natural Language Processing Lab, Northestern University.
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
 */

#ifndef __T2TPREDICTOR_H__
#define __T2TPREDICTOR_H__

#include "T2TModel.h"
#include "T2TLengthPenalty.h"

namespace transformer
{

#define T2T_PID_EMPTY -1

/* state for search. It keeps the path (back-pointer), prediction distribution,
   and etc. It can be regarded as a hypothsis in translation. */
class T2TState
{
public:
    /* we assume that the prediction is an integer */
    int prediction;

    /* id of the problem. One can regard it as the sentence id when we 
       translate a number of sentences in the batched manner. The hypothesis 
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

    /* nubmer of steps we go over so far */
    int nstep;

    /* pointer to the previous state */
    T2TState * last;
};

/* a bundle of states */
class T2TStateBundle
{
public:
    /* predictions */
    XTensor prediction;
    
    /* id of the previous state that generates the current one  */
    XTensor preID;

    /* mark that indicates whether each hypothesis is completed */
    XTensor endMark;

    /* probability of every prediction (last state of the path) */
    XTensor prob;

    /* probability of every path */
    XTensor probPath;

    /* model score of every path */
    XTensor modelScore;

    /* step number of each hypothesis */
    XTensor nstep;

    /* layers on the encoder side. We actually use the encoder output instead
       of all hidden layers. */
    TensorList layersEnc;

    /* layers on the decoder side */
    TensorList layersDec;

    /* list of states */
    T2TState * states;

    /* number of states */
    int stateNum;

    /* indicates whether it is the first state */
    bool isStart;

public:
    /* constructor */
    T2TStateBundle();

    /* de-constructor */
    ~T2TStateBundle();

    /* create states */
    void MakeStates(int num);
};

/* The predictor reads the current state and then predicts the next. 
   It is exactly the same procedure of MT inference -
   we get the state of previous words and then generate the next word.
   Here, a state can be regared as the representation of words (word 
   indices, hidden states, embeddings and etc.).  */
class T2TPredictor
{
private:
    /* pointer to the transformer model */
    T2TModel * m;

    /* current state */
    T2TStateBundle * s;

    /* start symbol */
    int startSymbol;

public:
    /* constructor */
    T2TPredictor();

    /* de-constructor */
    ~T2TPredictor();

    /* create an initial state */
    void Create(T2TModel * model, XTensor * top, const XTensor * input, int beamSize, T2TStateBundle * state);

    /* set the start symbol */
    void SetStartSymbol(int symbol);

    /* read a state */
    void Read(T2TModel * model, T2TStateBundle * state);

    /* predict the next state */
    void Predict(T2TStateBundle * next, XTensor * encoding, XTensor * inputEnc, XTensor * paddingEnc);

    /* generate paths up to the states of the current step */
    XTensor GeneratePaths(T2TStateBundle * state);
};

}

#endif
