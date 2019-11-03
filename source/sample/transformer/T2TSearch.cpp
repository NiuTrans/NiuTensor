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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2019-03-27
 */

#include "T2TSearch.h"
#include "T2TUtility.h"
#include "../../tensor/core/CHeader.h"

using namespace nts;

namespace transformer
{
    
/* constructor */
T2TSearch::T2TSearch()
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
T2TSearch::~T2TSearch()
{
    if(fullHypos != NULL)
        delete[] fullHypos;
    if(endSymbols != NULL)
        delete[] endSymbols;
}

/*
initialize the model
>> argc - number of arguments
>> argv - list of pointers to the arguments
*/
void T2TSearch::Init(int argc, char ** argv)
{
    LoadParamInt(argc, argv, "beamsize", &beamSize, 1);
    LoadParamInt(argc, argv, "batchsize", &batchSize, 1);
    LoadParamFloat(argc, argv, "lenalpha", &alpha, 0.2F);
    LoadParamInt(argc, argv, "endid", endSymbols, -1);
    LoadParamInt(argc, argv, "startid", &startSymbol, -1);

    if(endSymbols[0] >= 0)
        endSymbolNum = 1;
}

/* 
search for the most promising states 
>> model - the transformer model
>> input - input of the model
>> padding - padding of the input
>> output - output that represents the sequences as rows
*/
void T2TSearch::Search(T2TModel * model, XTensor * input, XTensor * padding, XTensor * output)
{
    T2TPredictor predictor;
    XTensor maskEnc;
    XTensor encoding;
    XTensor encodingBeam;
    XTensor inputBeam;
    XTensor paddingBeam;

    CheckNTErrors(endSymbolNum > 0, "The search class is not initialized!");
    CheckNTErrors(startSymbol >= 0, "The search class is not initialized!");

    Prepare(input->unitNum/input->GetDim(-1), beamSize);

    /* encoder mask */
    model->MakeMTMaskEnc(*input, *padding, maskEnc);

    //input->Dump(stderr, "input:");
    //maskEnc.Dump(stderr, "maskenc:");
    
    /* make the encoding network */
    encoding = model->MakeEncoder(*input, maskEnc, false);
    encoding.SetName(ENCODING_NAME);

    encodingBeam = Unsqueeze(encoding, encoding.order - 2, beamSize);
    inputBeam = Unsqueeze(*input, input->order - 1, beamSize);
    paddingBeam = Unsqueeze(*padding, padding->order - 1, beamSize);

    encodingBeam.ReshapeMerged(encodingBeam.order - 4);
    inputBeam.ReshapeMerged(inputBeam.order - 3);
    paddingBeam.ReshapeMerged(paddingBeam.order - 3);
    
    /* max output-length = 2 * source-length */
    maxLength = input->GetDim(-1) * 2;
    CheckNTErrors(maxLength > 0, "no max length specified!");
    
    T2TStateBundle * states = new T2TStateBundle[maxLength + 1];
    T2TStateBundle * first = states;
    
    /* create the first state */
    predictor.Create(model, &encodingBeam, input, beamSize, first);
    predictor.SetStartSymbol(startSymbol);

    first->isStart = true;

    /* generate the sequence from left to right */
    for(int i = 0 ; i < maxLength; i++){
        T2TStateBundle * cur = states + i;
        T2TStateBundle * next = states + i + 1;

        /* read the current state */
        predictor.Read(model, cur);

        /* predict the next state */
        predictor.Predict(next, &encodingBeam, &inputBeam, &paddingBeam);

        /* compute the model score (given the prediction probability) */
        Score(cur, next);

        /* beam pruning */
        Generate(next);

        /* expand the search graph */
        Expand(cur, next);

        /* push complete hypotheses into the heap */
        Collect(next);
    }

    /* fill the heap with imcomplete hypotheses if neccesary */
    FillHeap(&states[maxLength]);
    
    Dump(output);

    delete[] states;
}

/* 
prepare for search
>> batchSize - size of the batch
>> beamSize - size of the beam
*/
void T2TSearch::Prepare(int myBatchSize, int myBeamSize)
{
    batchSize = myBatchSize;
    beamSize = myBeamSize;

    if (fullHypos != NULL)
        delete[] fullHypos;

    fullHypos = new XHeap<MIN_HEAP, float>[batchSize];

    for (int i = 0; i < batchSize; i++)
        fullHypos[i].Init(beamSize);
}

/* 
compute the model score for each hypothesis 
>> prev - the beam of the previous state
>> beam - the beam that keeps a number of states
*/
void T2TSearch::Score(T2TStateBundle * prev, T2TStateBundle * beam)
{
    XTensor &score = beam->modelScore;
    XTensor &prob = beam->prob;
    XTensor &probPath = beam->probPath;
    XTensor &probPathPrev = prev->probPath;
    XTensor &lenPrev = prev->nstep;
    XTensor &len = beam->nstep;
    XTensor lp;
    XTensor mask;

    int order = prob.order;
    int outputSize = prob.GetDim(-1);
    int dims[MAX_TENSOR_DIM_NUM];
    for(int i = 0; i < order; i++)
        dims[i] = prob.GetDim(i);
    
    InitTensor(&score, &prob);
    InitTensor(&probPath, &prob);

    prob.Reshape(prob.unitNum/outputSize, outputSize);
    score.Reshape(score.unitNum/outputSize, outputSize);
    probPath.Reshape(score.unitNum / outputSize, outputSize);
    probPathPrev.Reshape(probPathPrev.unitNum);

    /* the log-scale probability of the entire sequence */
    _SumDim(&prob, &probPathPrev, &probPath, 0);


    InitTensor(&len, &lenPrev);
    InitTensor(&lp, &lenPrev);

    _ScaleAndShift(&lenPrev, &len, 1.0F, 1.0F);

    /* the GNMT-like length penalty */
    lp = T2TLengthPenalizer::GNMT(len, alpha);

    lp.Reshape(lp.unitNum);

    /* score = log-prob/lp */
    _DivDim(&probPath, &lp, &score, 0);

    if (prev->isStart) {
        XTensor firstMask = MakeFirstMask(beam);
        firstMask.Reshape(firstMask.unitNum);

        /* mask the hypotheses in the beam expect the first one */
        _SumDim(&score, &firstMask, &score, 0);
    }

    InitTensor(&mask, 
               prev->endMark.order, prev->endMark.dimSize, X_FLOAT, 
               prev->endMark.devID);
    _SetDataFixedCond(&mask, &prev->endMark, -1e9F);
    
    mask.Reshape(mask.unitNum);

    /* mask the completed hypotheses so that they cannot 
       be involved in further sorting and beam search. */
    _SumDim(&score, &mask, &score, 0);
    
    prob.Reshape(order, dims);
    score.Reshape(order, dims);
    probPath.Reshape(order, dims);
    probPathPrev.Reshape(order - 1, dims);
    lp.Reshape(order - 1, dims);
    mask.Reshape(order -1 , dims);
}

/* 
generate tokens for the next state via beam pruning
>> beam - the beam that keeps a number of states
*/
void T2TSearch::Generate(T2TStateBundle * beam)
{
    int dims[MAX_TENSOR_DIM_NUM];
    int dimsBeam[MAX_TENSOR_DIM_NUM];
    int dimsTopK[MAX_TENSOR_DIM_NUM];
    
    XTensor scoreTopK;
    XTensor &score = beam->modelScore;
    XTensor &index = beam->prediction;
    XTensor &preID = beam->preID;
    XTensor &probPath = beam->probPath;
    XTensor &prob = beam->prob;
    int order = score.order;

    CheckNTErrors(order >= 3, "The tensor must be of order 2 or larger.");
    CheckNTErrors(dimsBeam[order - 3] % beamSize == 0, "Wrong dimension size!");
    
    for (int i = 0; i < order; i++) {
        dims[i] = score.GetDim(i);
        dimsBeam[i] = score.GetDim(i);
        dimsTopK[i] = score.GetDim(i);
    }

    int sizeVocab = score.GetDim(-1);
    int stride = score.GetDim(-1);

    dimsBeam[order - 3] /= beamSize;
    dimsBeam[order - 1] *= beamSize;
    dimsTopK[order - 3] = dimsBeam[order - 3];
    dimsTopK[order - 1] = beamSize;
    
    InitTensor(&scoreTopK, order, dimsTopK, score.dataType,
                 score.devID);
    InitTensor(&index, order, dimsTopK, X_INT,
                 score.devID);
    InitTensor(&preID, order, dimsTopK, X_INT, -1);
    
    score.Reshape(order, dimsBeam);
    
    /* keep the most promissing candidates in the beam */
    TopK(score, scoreTopK, index, -1, beamSize);
    
    CopyValues(index, preID);
    
    /* "preID" represents the id (or the offset) of the previous state used to make the current
       hypothesis. Note that we reshape the "score" tensor into a matrix where each
       row means a previous state. The column number is size-of-beam \times vocab-size. We,
       therefore, divide entries of the top-k index by vocab-size to compute the id of the
       previous state for each hypothesis in the top-k list. */
    DescaleMe(preID, sizeVocab);
    
    /* Then, we do something similar to "preID". For the top-k predictions, we need 
       to know their indices in the vocabulary. We compute the offset of each prediction
       in the vocabulary by dividing it with vocab-size and computing the remainder. */
    ModMe(index, sizeVocab);

    score.Reshape(order, dims);

    /* we keep the top-k scores */
    InitTensor(&score, &scoreTopK);
    CopyValues(scoreTopK, score);

    /*  CPU data (TODO: remove GPU->CPU data copy!!!) */
    XTensor indexGPU;
    indexGPU = CopyValues(index);
    //InitTensorV2(&indexCPU, index.order, index.dimSize, index.dataType, index.denseRatio, -1);
    //CopyValues(index, indexCPU);

    for (int i = 0; i < indexGPU.unitNum; i++)
        indexGPU.SetInt(i * stride + indexGPU.GetInt(i), i);

    CheckNTErrors(IsSameShaped(prob, probPath), "Wrong tensor shape!");

    /* sequence probability of top-k candidates */
    XTensor probPathTopK;
    InitTensor(&probPathTopK, &scoreTopK);
    XTensor probTopK;
    InitTensor(&probTopK, &scoreTopK);

    for (int i = 0; i < probPath.order; i++) {
        dims[i] = probPath.GetDim(i);
        dimsTopK[i] = probPathTopK.GetDim(i);
    }

    order = probPath.order;
    probPath.Reshape(1, probPath.unitNum);
    probPathTopK.Reshape(1, probPathTopK.unitNum);
    prob.Reshape(1, prob.unitNum);
    probTopK.Reshape(1, probTopK.unitNum);

    _CopyIndexed(&probPath, &probPathTopK, probPathTopK.order - 1, &indexGPU);
    _CopyIndexed(&prob, &probTopK, probTopK.order - 1, &indexGPU);

    probPath.Reshape(order, dims);
    probPathTopK.Reshape(order, dimsTopK);

    prob.Reshape(order, dims);
    probTopK.Reshape(order, dimsTopK);

    probPath = probPathTopK;
    prob = probTopK;
}

/* 
expand the search graph 
>> beam - the beam that keeps a number of states
*/
void T2TSearch::Expand(T2TStateBundle * prev, T2TStateBundle * beam)
{
    CheckNTErrors(beam->prediction.unitNum == beam->preID.unitNum, "A problem occurs in the beam!");
    
    beam->MakeStates(beam->prediction.unitNum);

    T2TState * states = beam->states;
    XTensor & idRef = beam->preID;
    XTensor & modelScoreRef = beam->modelScore;
    XTensor & probRef = beam->prob;
    XTensor & probPathRef = beam->probPath;
    XTensor & predictionRef = beam->prediction;
    XTensor & endMark = beam->endMark;
    XTensor   id;
    XTensor   modelScore;
    XTensor   prob;
    XTensor   probPath;
    XTensor   prediction;
    XTensor   endMarkCPU;
    
    InitTensorOnCPU(&id, &idRef);
    InitTensorOnCPU(&modelScore, &modelScoreRef);
    InitTensorOnCPU(&prob, &probRef);
    InitTensorOnCPU(&probPath, &probPathRef);
    InitTensorOnCPU(&prediction, &predictionRef);
    InitTensorOnCPU(&endMarkCPU, &predictionRef);
    InitTensor(&endMark, &predictionRef);
    
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
    for(int i = 0; i < beam->stateNum; i += beamSize){
        for (int j = 0; j < beamSize; j++) {
            int k = i + j;
            T2TState & state = states[k];

            int offset = id.GetInt(k);
            int pid = i / beamSize;
            T2TState * last = prev->states + pid * beamSize + offset;

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
}

/* 
collect hypotheses with ending symbols. Given a beam of hypotheses,
we remove the finished hypotheses and keep them in a heap.
>> beam  - the beam that keeps a number of states
*/
void T2TSearch::Collect(T2TStateBundle * beam)
{
    T2TState * states = beam->states;

    for (int i = 0; i < beam->stateNum; i++) {
        T2TState & state = states[i];
        
        CheckNTErrors(state.pid >= 0 && state.pid < batchSize, 
                      "Invalid sample id!");

        /* we push the hypothesis into the heap when it is completed */
        if(state.isEnd != 0)
            fullHypos[state.pid].Push(HeapNode<float>(&state, state.modelScore));
    }
}

/* 
fill the hypotheis heap with incomplete hypotheses 
>> beam  - the beam that keeps a number of states (final)
*/
void T2TSearch::FillHeap(T2TStateBundle * beam)
{
    bool * emptyFlags = new bool[batchSize];
    for (int i = 0; i < batchSize; i++)
        emptyFlags[i] = (fullHypos[i].Count() == 0);

    T2TState * states = beam->states;

    for (int i = 0; i < beam->stateNum; i++) {
        T2TState & state = states[i];

        CheckNTErrors(state.pid >= 0 && state.pid < batchSize,
                      "Invalid sample id!");

        /* we push the imcomplete hypothesis into the heap */
        if (emptyFlags[state.pid] && state.isEnd == 0)
            fullHypos[state.pid].Push(HeapNode<float>(&state, state.modelScore));
    }

    delete[] emptyFlags;
}

/* 
save the output sequences in a tensor 
>> output - output sequences (for return)
*/
void T2TSearch::Dump(XTensor * output)
{
    int dims[3] = {batchSize, beamSize, maxLength};
    int * words = new int[maxLength];

    InitTensor(output, 3, dims, X_INT);
    SetDataFixedInt(*output, -1);

    /* heap for an input sentence in the batch */
    for(int h = 0; h < batchSize; h++){

        XHeap<MIN_HEAP, float> &heap = fullHypos[h];

        /* for each output in the beam */
        for(int i = 0; i < beamSize && heap.Count() > 0; i++){
            T2TState * state = (T2TState *)heap.Pop().index;
            
            int count = 0;
            bool isCompleted = true;

            /* we track the state from the end to the beginning */
            while(state != NULL){
                if (!state->isCompleted)
                    isCompleted = false;
                if (isCompleted)
                    words[count++] = -1;
                else
                    words[count++] = state->prediction;
                state = state->last;
            }

            /* dump the sentence to the output tensor */
            for(int w = 0; w < count; w++)
                output->Set3DInt(words[count - w - 1], h, beamSize - i - 1, w);
        }
    }

    delete[] words;
}

/* 
check if the token is an end symbol 
>> token - token to be checked
*/
bool T2TSearch::IsEnd(int token)
{
    CheckNTErrors(endSymbolNum > 0, "No end symbol?");

    for(int i = 0; i < endSymbolNum; i++){
        if(endSymbols[i] == token)
            return true;
    }

    return false;
}

/* 
set end symbols for search
>> tokens - end symbols
>> tokenNum - number of the end symbols
*/
void T2TSearch::SetEnd(const int * tokens, const int tokenNum)
{
    if(endSymbols != NULL)
        delete[] endSymbols;

    if(tokenNum <= 0)
        return;

    /* we may have multiple end symbols */
    tokens = new int[tokenNum];
    for(int i = 0; i < tokenNum; i++)
        endSymbols[i] = tokens[i];
    endSymbolNum = tokenNum;
}

/*
make a mask to prevent duplicated entries in beam expansion for the first position
>> beam - the beam that keeps the searching states
*/
XTensor T2TSearch::MakeFirstMask(T2TStateBundle * beam)
{
    XTensor &prob = beam->prob;
    XTensor mask;

    int order = prob.order;
    int dims[MAX_TENSOR_DIM_NUM];
    for (int i = 0; i < order - 1; i++)
        dims[i] = prob.GetDim(i);

    InitTensor(&mask, order - 1, dims, X_FLOAT);
    mask.SetZeroAll();

    for (int i = 0; i < mask.unitNum; i++) {
        if (i % beamSize != 0)
            mask.Set(-1e9, i);
    }

    mask.SetDevice(prob.devID);

    return mask;
}

}
