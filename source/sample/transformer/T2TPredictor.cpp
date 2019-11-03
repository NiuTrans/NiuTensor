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
 */

#include "T2TPredictor.h"
#include "../../tensor/core/CHeader.h"

using namespace nts;

namespace transformer
{

/* constructor */
T2TStateBundle::T2TStateBundle()
{
    states = NULL;
    isStart = false;
}

/* de-constructor */
T2TStateBundle::~T2TStateBundle()
{
    if(states != NULL)
        delete[] states;
}

/* 
create states 
>> num - number of states
*/
void T2TStateBundle::MakeStates(int num)
{
    CheckNTErrors(num > 0, "invalid number");

    if(states != NULL)
        delete[] states;

    states = new T2TState[num];

    for(int i = 0; i < num; i++){
        states[i].prediction = -1;
        states[i].pid = T2T_PID_EMPTY;
        states[i].isEnd = false;
        states[i].isStart = false;
        states[i].isCompleted = false;
        states[i].prob = 0;
        states[i].probPath = 0;
        states[i].modelScore = 0;
        states[i].nstep = 0;
        states[i].last = NULL;
    }

    stateNum = num;
}

/* constructor */
T2TPredictor::T2TPredictor()
{
    startSymbol = -1;
}

/* de-constructor */
T2TPredictor::~T2TPredictor()
{
}

/* 
create an initial state 
>> model - the t2t model
>> top - the top-most layer of the network
>> input - input of the network
>> beamSize - beam size
>> state - the state to be initialized
*/
void T2TPredictor::Create(T2TModel * model, XTensor * top, const XTensor * input, int beamSize, T2TStateBundle * state)
{
    state->layersEnc.Clear();
    state->layersDec.Clear();

    XTensor * encoding = XLink::SearchNode(top, ENCODING_NAME);
    CheckNTErrors(encoding != NULL, "No encoding layers found!");

    state->layersEnc.Add(encoding);
    state->layersDec.Add(NULL);

    int dims[MAX_TENSOR_DIM_NUM];
    for (int i = 0; i < input->order - 1; i++)
        dims[i] = input->GetDim(i);
    dims[input->order - 1] = beamSize;

    InitTensor(&state->probPath, input->order, dims, X_FLOAT, input->devID);
    InitTensor(&state->nstep, input->order, dims, X_FLOAT, input->devID);
    InitTensor(&state->endMark, input->order, dims, X_INT, input->devID);

    state->probPath.SetZeroAll();
    state->nstep.SetZeroAll();
    state->endMark.SetZeroAll();

    state->stateNum = 0;
}

/*
set start symbol
>> symbol - the symbol (in integer)
*/
void T2TPredictor::SetStartSymbol(int symbol)
{
    startSymbol = symbol;
}

/* 
read a state 
>> model - the t2t model that keeps the network created so far
>> state - a set of states. It keeps
             1) hypotheses (states)
             2) probablities of hypotheses
             3) parts of the network for expanding toward the next state
*/
void T2TPredictor::Read(T2TModel * model, T2TStateBundle * state)
{
    m = model;
    s = state;
}

/*
predict the next state
>> next - next states (assuming that the current state has been read)
>> encoding - encoder output
>> inputEnc - input of the encoder
>> paddingEnc - padding of the encoder
*/
void T2TPredictor::Predict(T2TStateBundle * next, XTensor * encoding,
                           XTensor * inputEnc, XTensor * paddingEnc)
{
    int dims[MAX_TENSOR_DIM_NUM];

    next->layersEnc.Clear();
    next->layersDec.Clear();
    
    AttDecoder &decoder = *m->decoder;
    
    /* word indices of previous positions */
    XTensor * inputLast = (XTensor*)s->layersDec.GetItem(0);

    /* word indices of positions up to next state */
    XTensor inputDec;

    /* the first token */
    XTensor first;
    
    CheckNTErrors(inputEnc->order >= 2, "Wrong order of the tensor!");
    for(int i = 0; i < inputEnc->order - 1; i++)
        dims[i] = inputEnc->GetDim(i);
    dims[inputEnc->order - 1] = 1;

    InitTensor(&first, inputEnc->order, dims, X_INT, inputEnc->devID);
    _SetDataFixedInt(&first, startSymbol);

    /* add a new word into the input sequence of the decoder side */
    if (inputLast == NULL) {
        inputDec = Identity(first);
    }
    else{
        inputDec = GeneratePaths(s);
        inputDec.SetDevice(inputEnc->devID);

        inputDec = Concatenate(first, inputDec, inputDec.order - 1);
    }

    /* prediction probabilities */
    XTensor &output = next->prob;
    XTensor decoding;
    XTensor decodingStep;
    
    for(int i = 0; i < inputDec.order - 1; i++)
        dims[i] = inputDec.GetDim(i);
    dims[inputDec.order - 1] = inputDec.GetDim(-1);
    
    XTensor paddingDec;
    InitTensor(&paddingDec, inputDec.order, dims, X_INT, paddingEnc->devID);
    SetDataFixedInt(paddingDec, 1);
    
    XTensor maskDec;
    XTensor maskEncDec;
    
    /* decoder mask */
    m->MakeMTMaskDec(*inputEnc, inputDec, *paddingEnc, paddingDec, maskDec, maskEncDec);

    /* make the decoding network */
    decoding = decoder.Make(inputDec, *encoding, maskDec, maskEncDec, false);

    XTensor selectSrc;
    XTensor selectTgt;

    CheckNTErrors(decoding.order >= 2, "The tensor must be of order 2 or larger!");

    int stride = decoding.GetDim(decoding.order - 2);

    InitTensor1D(&selectSrc, 1, X_INT);
    InitTensor1D(&selectTgt, 1, X_INT);

    selectSrc.SetInt(stride - 1, 0);
    selectTgt.SetInt(0, 0);

    selectSrc.SetDevice(decoding.devID);
    selectTgt.SetDevice(decoding.devID);
    
    /* the decoder output of the last position */
    decodingStep = CopyIndexed(decoding, decoding.order - 2, selectSrc, selectTgt);

    /* generate the output probabilities */
    m->outputLayer->Make(decodingStep, output);
    
    next->layersEnc.AddList(&s->layersEnc);
    next->layersDec.Add(&inputDec);
    next->layersDec.Add(&output);
}

/* 
generate paths up to the states of the current step 
>> state - state bundle of the current step
*/
XTensor T2TPredictor::GeneratePaths(T2TStateBundle * state)
{
    CheckNTErrors(state->stateNum >= 0, "Illegal state!");

    int distance = -1;
    
    for(int i = 0; i < state->stateNum; i++){
        T2TState * cur = state->states + i;
        int nsteps = 0;

        while(cur != NULL){
            nsteps++;
            cur = cur->last;
        }

        if(nsteps > distance)
            distance = nsteps;
    }

    XTensor path;
    InitTensor2D(&path, state->stateNum, distance, X_INT);
    path.SetZeroAll();

    for(int i = 0; i < state->stateNum; i++){
        T2TState * cur = state->states + i;
        int nsteps = 0;

        while(cur != NULL){
            nsteps++;
            path.Set2DInt(cur->prediction, i, distance - nsteps);
            cur = cur->last;
        }
    }

    return path;
}

}

