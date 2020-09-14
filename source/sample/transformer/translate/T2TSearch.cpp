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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2019-03-27
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04, 2020-06
 */

#include "T2TSearch.h"
#include "../module/T2TUtility.h"
#include "../../../tensor/core/CHeader.h"

using namespace nts;

namespace transformer
{
/* constructor */
BeamSearch::BeamSearch()
{
    alpha = 0;
    maxLength = 0;
    beamSize = 0;
    batchSize = 0;
    endSymbolNum = 0;
    fullHypos = NULL;
    endSymbols = new int[32];
    startSymbol = -1;
}

/* de-constructor */
BeamSearch::~BeamSearch()
{
    if (fullHypos != NULL)
        delete[] fullHypos;
    if (endSymbols != NULL)
        delete[] endSymbols;
}

/*
initialize the model
>> argc - number of arguments
>> argv - list of pointers to the arguments
*/
void BeamSearch::Init(T2TConfig& config)
{
    beamSize = config.beamSize;
    batchSize = config.sBatchSize;
    alpha = config.lenAlpha;
    endSymbols[0] = config.endID;
    startSymbol = config.startID;
    scalarMaxLength = config.maxLenAlpha;

    if (endSymbols[0] >= 0)
        endSymbolNum = 1;
}

/*
prepare for search
>> batchSize - size of the batch
>> beamSize - size of the beam
*/
void BeamSearch::Prepare(int myBatchSize, int myBeamSize)
{
    batchSize = myBatchSize;
    beamSize = myBeamSize;
    needReorder = false;

    /* prepare for the heap of hypotheses */
    if (fullHypos != NULL)
        delete[] fullHypos;

    fullHypos = new XHeap<MIN_HEAP, float>[batchSize];

    for (int i = 0; i < batchSize; i++)
        fullHypos[i].Init(beamSize);

    /* prepare for the indices of alive states */
    aliveStatePids.Clear();
    aliveSentList.Clear();
    for (int i = 0; i < batchSize; i++) {
        aliveStatePids.Add(i);
        aliveSentList.Add(i);
    }
}

/*
search for the most promising states
>> model - the transformer model
>> input - input of the model
>> padding - padding of the input
>> output - output that represents the sequences as rows
>> score - score of the sequences
*/
void BeamSearch::Search(T2TModel* model, XTensor& input, XTensor& padding, 
                        IntList* output, XTensor& score)
{
    T2TPredictor predictor;
    XTensor maskEnc;
    XTensor encoding;
    XTensor encodingBeam;
    XTensor inputBeam;
    XTensor paddingBeam;

    CheckNTErrors(endSymbolNum > 0, "The search class is not initialized!");
    CheckNTErrors(startSymbol >= 0, "The search class is not initialized!");

    Prepare(input.unitNum / input.dimSize[input.order - 1], beamSize);

    /* encoder mask */
    model->MakeMTMaskEnc(padding, maskEnc);

    /* make the encoding network */
    encoding = model->MakeEncoder(input, &maskEnc, false);

    encodingBeam = Unsqueeze(encoding, encoding.order - 2, beamSize);
    inputBeam = Unsqueeze(input, input.order - 1, beamSize);
    paddingBeam = Unsqueeze(padding, padding.order - 1, beamSize);

    encodingBeam.ReshapeMerged(encodingBeam.order - 4);
    inputBeam.ReshapeMerged(inputBeam.order - 3);
    paddingBeam.ReshapeMerged(paddingBeam.order - 3);

    /* max output-length = scalar * source-length */
    int lengthLimit = (int)(input.dimSize[input.order - 1] * scalarMaxLength);

    CheckNTErrors(lengthLimit > 0, "no max length specified!");
    maxLength = lengthLimit;

    T2TStateBundle* states = new T2TStateBundle[lengthLimit + 1];
    T2TStateBundle* first = states;
    T2TStateBundle* cur = NULL;
    T2TStateBundle* next = NULL;

    /* create the first state */
    predictor.Create(model, &encodingBeam, &input, beamSize, first);
    predictor.SetStartSymbol(startSymbol);

    first->isStart = true;

    XTensor aliveState;
    InitTensor1D(&aliveState, batchSize * beamSize, X_INT, input.devID);
    SetAscendingOrder(aliveState, 0);

    XTensor reorderState;
    InitTensor1D(&reorderState, batchSize * beamSize, X_INT, input.devID);
    SetAscendingOrder(reorderState, 0);

    /* generate the sequence from left to right */
    for (int l = 0; l < lengthLimit; l++) {
        if (beamSize > 1) {
            inputBeam = AutoGather(inputBeam, reorderState);
            paddingBeam = AutoGather(paddingBeam, reorderState);
            encodingBeam = AutoGather(encodingBeam, reorderState);
        }

        cur = states + l;
        next = states + l + 1;

        /* read the current state */
        predictor.Read(model, cur);

        /* predict the next state */
        predictor.Predict(next, aliveState, encodingBeam, inputBeam,
            paddingBeam, batchSize * beamSize, l == 0, reorderState, needReorder, l);

        /* compute the model score (given the prediction probability) */
        Score(cur, next);

        /* beam pruning */
        Generate(cur, next);

        /* expand the search graph */
        Expand(cur, next, reorderState);

        /* push complete hypotheses into the heap */
        Collect(next);

        /* stop searching when all hypotheses are completed */
        if (IsAllCompleted(next)) {
            maxLength = l + 1;
            break;
        }

        /* remove finished sentences */

        //RemoveFinishedStates(next, encodingBeam, inputBeam, paddingBeam, aliveState);
    }

    /* fill the heap with incomplete hypotheses if necessary */
    FillHeap(next);

    Dump(output, &score);

    delete[] states;
}

/*
compute the model score for each hypotheses
>> prev - the beam of the previous state
>> beam - the beam that keeps a number of states
*/
void BeamSearch::Score(T2TStateBundle* prev, T2TStateBundle* beam)
{
    XTensor& score = beam->modelScore;
    XTensor& prob = beam->prob;
    XTensor& probPath = beam->probPath;
    XTensor& probPathPrev = prev->probPath;
    XTensor mask;

    int order = prob.order;
    int outputSize = prob.dimSize[prob.order - 1];
    int dims[MAX_TENSOR_DIM_NUM];
    for (int i = 0; i < order; i++)
        dims[i] = prob.dimSize[i];

    if (prob.dataType == X_FLOAT16)
        prob = ConvertDataType(prob, X_FLOAT);

    InitTensor(&score, &prob);
    InitTensor(&probPath, &prob);

    prob.Reshape(prob.unitNum / outputSize, outputSize);
    score.Reshape(score.unitNum / outputSize, outputSize);
    probPath.Reshape(score.unitNum / outputSize, outputSize);
    probPathPrev.Reshape(probPathPrev.unitNum);

    /* the log-scale probability of the entire sequence */
    SumDim(prob, probPathPrev, probPath, 0);

    beam->nstep = prev->nstep + 1.0F;

    /* the GNMT-like length penalty */
    float lp = T2TLengthPenalizer::GNMT(beam->nstep, alpha);

    /* score = log-prob/lp */
    score = probPath / lp;

    if (prev->isStart) {
        XTensor firstMask = MakeFirstMask(beam);
        firstMask.Reshape(firstMask.unitNum);

        /* mask the hypotheses in the beam except the first one */
        SumDim(score, firstMask, score, 0);
    }

    InitTensor(&mask,
        prev->endMark.order, prev->endMark.dimSize, X_FLOAT,
        prev->endMark.devID);
    mask.SetZeroAll();
    _SetDataFixedCond(&mask, &prev->endMark, -1e9F);

    mask.Reshape(mask.unitNum);

    /* mask the completed hypotheses so that they cannot
       be involved in further sorting and beam search. */
    SumDim(score, mask, score, 0);

    prob.Reshape(order, dims);
    score.Reshape(order, dims);
    probPath.Reshape(order, dims);
}

/*
generate tokens for the next state via beam pruning
>> prev - the last beam
>> beam - the beam that keeps a number of states
*/
void BeamSearch::Generate(T2TStateBundle* prev, T2TStateBundle* beam)
{
    int dims[MAX_TENSOR_DIM_NUM];
    int dimsBeam[MAX_TENSOR_DIM_NUM];
    int dimsTopK[MAX_TENSOR_DIM_NUM];

    XTensor scoreTopK;
    XTensor indexCPU;
    XTensor& score = beam->modelScore;
    XTensor& index = beam->prediction;
    XTensor& preID = beam->preID;
    XTensor& probPath = beam->probPath;
    XTensor& prob = beam->prob;

    int order = score.order;

    for (int i = 0; i < order; i++) {
        dims[i] = score.dimSize[i];
        dimsBeam[i] = score.dimSize[i];
        dimsTopK[i] = score.dimSize[i];
    }

    CheckNTErrors(order >= 3, "The tensor must be of order 2 or larger.");
    CheckNTErrors(dimsBeam[order - 3] % beamSize == 0, "Wrong dimension size!");

    int sizeVocab = score.dimSize[score.order - 1];
    int stride = score.dimSize[score.order - 1];

    dimsBeam[order - 3] /= beamSize;
    dimsBeam[order - 1] *= beamSize;
    dimsTopK[order - 3] = dimsBeam[order - 3];
    dimsTopK[order - 1] = beamSize;

    InitTensor(&scoreTopK, order, dimsTopK, score.dataType, score.devID);
    InitTensor(&index, order, dimsTopK, X_INT, score.devID);
    InitTensor(&preID, order, dimsTopK, X_INT, -1);
    InitTensor(&indexCPU, order, dimsTopK, X_INT, -1);

    score.Reshape(order, dimsBeam);
    prob.Reshape(order, dimsBeam);

    /* keep the most promising candidates in the beam */
    TopK(score, scoreTopK, index, -1, beamSize, true);

    float lp = T2TLengthPenalizer::GNMT(beam->nstep, alpha);

    CopyValues(index, indexCPU);
    CopyValues(index, preID);

    /* "preID" represents the id (or the offset) of the previous state used to make the current
       hypotheses. Note that we reshape the "score" tensor into a matrix where each
       row means a previous state. The column number is size-of-beam \times vocab-size. We,
       therefore, divide entries of the top-k index by vocab-size to compute the id of the
       previous state for each hypotheses in the top-k list. */
    DescaleMe(preID, sizeVocab);

    /* Then, we do something similar to "preID". For the top-k predictions, we need
       to know their indices in the vocabulary. We compute the offset of each prediction
       in the vocabulary by dividing it with vocab-size and computing the remainder. */
    ModMe(index, sizeVocab);

    /* we keep the top-k scores */
    score = CopyValues(scoreTopK);

    for (int i = 0; i < indexCPU.unitNum; i += beamSize) {
        for (int j = 0; j < beamSize; j++) {
            indexCPU.SetInt(i * stride + indexCPU.GetInt(i + j), i + j);
        }
    }

    /* sequence probability of top-k candidates */
    for (int i = 0; i < probPath.order; i++) {
        dims[i] = probPath.dimSize[i];
        dimsTopK[i] = scoreTopK.dimSize[i];
    }

    order = probPath.order;

    prob.Reshape(prob.unitNum, 1);
    probPath.Reshape(probPath.unitNum, 1);
    indexCPU.Reshape(indexCPU.dimSize[0], indexCPU.dimSize[indexCPU.order - 1]);

    indexCPU.SetDevice(prob.devID);
    prob = Gather(prob, indexCPU);
    probPath = Gather(probPath, indexCPU);

    prob.Reshape(order, dimsTopK);
    probPath.Reshape(order, dimsTopK);
}

/*
expand the search graph
>> prev - the last beam
>> beam - the beam that keeps a number of states
>> reorderState - the new order of states
*/
void BeamSearch::Expand(T2TStateBundle* prev, T2TStateBundle* beam, XTensor& reorderState)
{
    CheckNTErrors(beam->prediction.unitNum == beam->preID.unitNum, 
                  "A problem occurs in the beam!");

    beam->MakeStates(beam->prediction.unitNum);

    T2TState* states = beam->states;
    XTensor& idRef = beam->preID;
    XTensor& modelScoreRef = beam->modelScore;
    XTensor& probRef = beam->prob;
    XTensor& probPathRef = beam->probPath;
    XTensor& predictionRef = beam->prediction;
    XTensor& endMark = beam->endMark;
    XTensor   id;
    XTensor   modelScore;
    XTensor   prob;
    XTensor   probPath;
    XTensor   prediction;
    XTensor   endMarkCPU;
    XTensor reorderStateCPU;

    InitTensorOnCPU(&id, &idRef);
    InitTensorOnCPU(&modelScore, &modelScoreRef);
    InitTensorOnCPU(&prob, &probRef);
    InitTensorOnCPU(&probPath, &probPathRef);
    InitTensorOnCPU(&prediction, &predictionRef);
    InitTensorOnCPU(&endMarkCPU, &predictionRef);
    InitTensor(&endMark, &predictionRef);
    InitTensorOnCPU(&reorderStateCPU, &reorderState);

    /* we copy the data to CPU because the frequent access to GPU is slow
       and we can speed-up the process by doing the job on CPU. */
    CopyValues(idRef, id);
    CopyValues(modelScoreRef, modelScore);
    CopyValues(probRef, prob);
    CopyValues(probPathRef, probPath);
    CopyValues(predictionRef, prediction);

    CheckNTErrors(beam->stateNum == id.unitNum, "Errors occur in counting!");

    /* Related variables are kept on the states of the graph. All these are
       maintained on CPUs to ease the implementation of frequent access and
       modification of the states. An alternative is to do this on GPUs but
       it needs much more coding work and the speed-up is not obvious. */

    for (int i = 0; i < beam->stateNum; i += beamSize) {
        for (int j = 0; j < beamSize; j++) {
            int k = i + j;
            T2TState& state = states[k];

            int offset = id.GetInt(k);
            int pid = i / beamSize;
            reorderStateCPU.SetInt(i + offset, i + j);
            if (offset != j)
                needReorder = true;

            T2TState* last = prev->states + pid * beamSize + offset;

            CheckNTErrors(offset >= 0, "Wrong state index!");

            /* pointer to the previous state */
            if (prev->isStart) {
                state.last = NULL;
                state.pid = pid;
                state.nstep = 0;
                state.isCompleted = false;
            }
            else {
                state.last = last;
                state.pid = state.last->pid;
                state.nstep = last->nstep + 1;
                state.isCompleted = last->isCompleted;
                CheckNTErrors(offset < prev->stateNum, "Wrong state index!");
            }
            /*if(aliveStatePids.size() < batchSize)
                state.pid = aliveStatePids[i/beamSize];*/

                /* scores */
            state.modelScore = modelScore.Get(k);
            state.prob = prob.Get(k);
            state.probPath = probPath.Get(k);

            /* prediction */
            state.prediction = prediction.GetInt(k);

            CheckNTErrors(state.prediction >= 0, "Illegal prediction!");

            /* check if it is the end of the sequence */
            state.isEnd = IsEnd(state.prediction);
            state.isCompleted = (state.isCompleted || state.isEnd);

            /* set the ending mark */
            endMarkCPU.SetInt(state.isEnd, k);
        }
    }

    /* copy the ending mark from CPU to the target device */
    CopyValues(endMarkCPU, endMark);
    CopyValues(reorderStateCPU, reorderState);
}

/*
collect hypotheses with ending symbols. Given a beam of hypotheses,
we remove the finished hypotheses and keep them in a heap.
>> beam  - the beam that keeps a number of states
*/
void BeamSearch::Collect(T2TStateBundle* beam)
{
    T2TState* states = beam->states;

    for (int i = 0; i < beam->stateNum; i++) {
        T2TState& state = states[i];

        CheckNTErrors(state.pid >= 0 && state.pid < batchSize,
            "Invalid sample id!");

        /* check if this is the first end symbol. It is false
           if there have been end symbols in previously generated words. */
        bool isCompleted = state.isCompleted && 
             (state.last == NULL || !state.last->isCompleted);

        /* we push the hypothesis into the heap when it is completed */
        if ((state.isEnd || state.isCompleted)) {
            fullHypos[state.pid].Push(HeapNode<float>(&state, state.modelScore));
        }
    }
}

/*
fill the hypothesis heap with incomplete hypotheses
>> beam  - the beam that keeps a number of states (final)
*/
void BeamSearch::FillHeap(T2TStateBundle* beam)
{
    T2TState* states = beam->states;

    for (int i = 0; i < beam->stateNum / beamSize; i++) {
        for (int j = 0; j < beamSize; j++) {
            T2TState& state = states[i * beamSize + j];

            /* we push the incomplete hypothesis into the heap */
            if (fullHypos[state.pid].Count() == 0 && state.isEnd && state.isCompleted) {
                fullHypos[state.pid].Push(HeapNode<float>(&state, state.modelScore));
            }
            else {
                auto node = fullHypos[state.pid].Top();
                float score = node.value;
                if (score < state.modelScore)
                    fullHypos[state.pid].Push(HeapNode<float>(&state, state.modelScore));
            }
        }
    }
}

/*
save the output sequences in a tensor
>> output - output sequences (for return)
>> score - score of thes sequences
*/
void BeamSearch::Dump(IntList* output, XTensor* score)
{
    int dims[3] = { batchSize, 1, maxLength };

    InitTensor(score, 2, dims, X_FLOAT);
    score->SetZeroAll();

    /* heap for an input sentence in the batch */
    for (int h = 0; h < batchSize; h++) {
        XHeap<MIN_HEAP, float>& heap = fullHypos[h];
        int c = heap.Count();

        float bestScore = -1e9F;
        T2TState* state = NULL;
        for (int i = 0; i < c; i++) {
            auto node = heap.Pop();
            T2TState* s = (T2TState*)node.index;
            if (i == 0 || bestScore < node.value) {
                state = s;
                bestScore = node.value;
            }
        }

        int count = 0;
        bool isCompleted = true;

        /* we track the state from the end to the beginning */
        while (state != NULL) {
            if (!state->isCompleted)
                isCompleted = false;
            if (isCompleted) {
                output[h].Add(2);
            }
            else {
                output[h].Add(state->prediction);
            }
            state = state->last;
        }
        output[h].Reverse();

        score->Set2D(bestScore, h, 0);
    }
}

/*
check if the token is an end symbol
>> token - token to be checked
*/
bool BeamSearch::IsEnd(int token)
{
    CheckNTErrors(endSymbolNum > 0, "No end symbol?");

    for (int i = 0; i < endSymbolNum; i++) {
        if (endSymbols[i] == token)
            return true;
    }

    return false;
}

/*
set end symbols for search
>> tokens - end symbols
>> tokenNum - number of the end symbols
*/
void BeamSearch::SetEnd(const int* tokens, const int tokenNum)
{
    if (endSymbols != NULL)
        delete[] endSymbols;

    if (tokenNum <= 0)
        return;

    /* we may have multiple end symbols */
    tokens = new int[tokenNum];
    for (int i = 0; i < tokenNum; i++)
        endSymbols[i] = tokens[i];
    endSymbolNum = tokenNum;
}

/*
check whether all hypotheses are completed
>> beam - the beam that keeps the searching states
*/
bool BeamSearch::IsAllCompleted(T2TStateBundle* beam)
{
    T2TState* states = beam->states;

    for (int i = 0; i < beam->stateNum; i++) {
        T2TState& state = states[i];
        if (!state.isCompleted)
            return false;
    }

    return true;
}

/*
update the beam by removing finished hypotheses
>> beam - the beam that keeps the searching states
>> aliveEncoding - new input embeddings for the encoder, (B, L, E)
>> aliveInput - new input tokens of the encoder, (B, L)
>> alivePadding - new paddings for the inputs, (B, L)
<< aliveIdx - the indices of alive states
*/
void BeamSearch::RemoveFinishedStates(T2TStateBundle* beam, XTensor& aliveEncoding,
                                      XTensor& aliveInput, XTensor& alivePadding, 
                                      XTensor& aliveState)
{
    T2TState* states = beam->states;

    /* get the indices of uncompleted sentences and states */
    aliveSentList.Clear();
    IntList aliveStateList;
    int count = 0;

    /* the number of completed sentences */
    for (int i = 0; i < beam->stateNum; i += beamSize) {
        int endState = 0;
        for (int j = 0; j < beamSize; j++) {
            if (states[i + j].isEnd) {
                endState++;
            }
        }
        bool isSentCompleted = (endState == beamSize);

        int sent = i / beamSize;
        if (!isSentCompleted) {
            aliveSentList.Add(sent);
            for (int j = 0; j < beamSize; j++) {
                aliveStateList.Add(i + j);
            }
        }
        else {
            aliveStatePids.Remove(sent - count);
            count++;
        }
    }

    InitTensor1D(&aliveState, aliveStateList.Size(), X_INT, aliveEncoding.devID);
    aliveState.SetData(aliveStateList.items, aliveStateList.Size());

    XTensor aliveSent;
    InitTensor1D(&aliveSent, aliveSentList.Size(), X_INT, aliveEncoding.devID);
    aliveSent.SetData(aliveSentList.items, aliveSentList.Size());

    if (aliveStateList.Size() < aliveEncoding.dimSize[0] && aliveStateList.Size() > 0) {
        aliveInput = AutoGather(aliveInput, aliveState);
        alivePadding = AutoGather(alivePadding, aliveState);
        aliveEncoding = AutoGather(aliveEncoding, aliveState);
        beam->prob = AutoGather(beam->prob, aliveSent);
        beam->endMark = AutoGather(beam->endMark, aliveSent);
        beam->probPath = AutoGather(beam->probPath, aliveSent);
        beam->modelScore = AutoGather(beam->modelScore, aliveSent);
        beam->prediction = AutoGather(beam->prediction, aliveSent);
    }
}

/*
make a mask to prevent duplicated entries in beam expansion for the first position
>> beam - the beam that keeps the searching states
*/
XTensor BeamSearch::MakeFirstMask(T2TStateBundle* beam)
{
    XTensor& prob = beam->prob;
    XTensor mask;

    int order = prob.order;
    int dims[MAX_TENSOR_DIM_NUM];
    for (int i = 0; i < order - 1; i++)
        dims[i] = prob.dimSize[i];

    InitTensor(&mask, order - 1, dims, X_FLOAT);
    mask.SetZeroAll();

    for (int i = 0; i < mask.unitNum; i++) {
        if (i % beamSize != 0)
            mask.Set(-1e9, i);
    }

    mask.SetDevice(prob.devID);

    return mask;
}

/* constructor */
GreedySearch::GreedySearch()
{
    maxLength = 0;
    batchSize = 0;
    endSymbolNum = 0;
    endSymbols = new int[32];
    startSymbol = -1;
}

/* de-constructor */
GreedySearch::~GreedySearch()
{
    if (endSymbols != NULL)
        delete[] endSymbols;
}

/*
initialize the model
>> argc - number of arguments
>> argv - list of pointers to the arguments
*/
void GreedySearch::Init(T2TConfig& config)
{
    batchSize = config.wBatchSize;
    endSymbols[0] = config.endID;
    startSymbol = config.startID;
    scalarMaxLength = config.maxLenAlpha;

    if (endSymbols[0] >= 0)
        endSymbolNum = 1;
}

/*
prepare for search
>> batchSize - size of the batch
*/
void GreedySearch::Prepare(int myBatchSize)
{
    batchSize = myBatchSize;
}

/* check if the token is an end symbol */
bool GreedySearch::IsEnd(int token)
{
    CheckNTErrors(endSymbolNum > 0, "No end symbol?");

    for (int i = 0; i < endSymbolNum; i++) {
        if (endSymbols[i] == token)
            return true;
    }

    return false;
}

/* set end symbols for search */
void GreedySearch::SetEnd(const int* tokens, const int tokenNum)
{
    if (endSymbols != NULL)
        delete[] endSymbols;

    if (tokenNum <= 0)
        return;

    /* we may have multiple end symbols */
    tokens = new int[tokenNum];
    for (int i = 0; i < tokenNum; i++)
        endSymbols[i] = tokens[i];
    endSymbolNum = tokenNum;
}

/*
search for the most promising states
>> model - the transformer model
>> input - input of the model
>> padding - padding of the input
>> output - output that represents the sequences as rows
*/
void GreedySearch::Search(T2TModel* model, XTensor& input, 
                          XTensor& padding, IntList* output)
{
    XTensor maskEnc;
    XTensor encoding;

    /* dynamic batch size */
    Prepare(input.unitNum / input.dimSize[input.order - 1]);

    /* encoder mask */
    model->MakeMTMaskEnc(padding, maskEnc);

    /* make the encoding network */
    encoding = model->encoder->Make(input, &maskEnc, false);

    /* max output-length = scalar * source-length */
    maxLength = (int)(input.dimSize[input.order - 1] * scalarMaxLength);

    /* the first token */
    XTensor inputDec;
    InitTensor2D(&inputDec, batchSize, 1, X_INT, input.devID);
    inputDec.SetDataFixed(startSymbol);

    /* initialize the finished flags */
    int* finishedFlags = new int[batchSize];
    for (int i = 0; i < batchSize; i++)
        finishedFlags[i] = 0;

    /* generate the sequence from left to right */
    int l = 0;
    for (; l < maxLength; l++) {
        XTensor prob;
        XTensor maskDec;
        XTensor maskEncDec;
        XTensor paddingDec;
        XTensor decoding;
        XTensor indexCPU;
        XTensor bestScore;

        InitTensor(&paddingDec, inputDec.order, inputDec.dimSize, X_INT, padding.devID);
        paddingDec.SetDataFixed(1);

        /* decoder mask */
        model->MakeMTMaskDec(padding, paddingDec, maskDec, maskEncDec);

        /* make the decoding network */
        decoding = model->decoder->Make(inputDec, encoding, NULL, &maskEncDec, l, false);

        /* generate the output probabilities */
        model->outputLayer->Make(decoding, prob, false, false);

        /* get the most promising prediction */
        prob.Reshape(prob.dimSize[0], prob.dimSize[prob.order - 1]);
        InitTensor2D(&bestScore, prob.dimSize[0], 1, prob.dataType, prob.devID);
        TopK(prob, bestScore, inputDec, -1, 1);

        /* save the prediction */
        InitTensorOnCPU(&indexCPU, &inputDec);
        CopyValues(inputDec, indexCPU);

        for (int i = 0; i < batchSize; i++) {
            output[i].Add(indexCPU.GetInt(i));
            if (IsEnd(indexCPU.GetInt(i)))
                finishedFlags[i] = 1;
        }

        int finished = 0;
        for (int i = 0; i < batchSize; i++)
            finished += finishedFlags[i];
        if (finished == batchSize)
            break;
    }

    delete[] finishedFlags;
}

}