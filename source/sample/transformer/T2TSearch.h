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

#ifndef __T2TSEARCH_H__
#define __T2TSEARCH_H__

#include "T2TModel.h"
#include "T2TPredictor.h"

namespace transformer
{

/* The class orgnizes the search process. It calls "predictors" to generate
   distributions of the predictions and prunes the search space by beam pruning.
   This makes a graph where each path respresents a translation hypothsis.
   The output can be the path with the highest model score. */
class T2TSearch
{
private:
    /* the alpha parameter controls the length preference */
    float alpha;

    /* predictor */
    T2TPredictor predictor;
    
    /* max length of the generated sequence */
    int maxLength;
    
    /* beam size */
    int beamSize;

    /* batch size */
    int batchSize;

    /* we keep the final hypotheses in a heap for each sentence in the batch. */
    XHeap<MIN_HEAP, float> * fullHypos;

    /* array of the end symbols */
    int * endSymbols;

    /* number of the end symbols */
    int endSymbolNum;

    /* start symbol */
    int startSymbol;

public:
    /* constructor */
    T2TSearch();

    /* de-constructor */
    ~T2TSearch();
    
    /* initialize the model */
    void Init(int argc, char ** argv);

    /* search for the most promising states */
    void Search(T2TModel * model, XTensor * input, XTensor * padding, XTensor * output);

    /* preparation */
    void Prepare(int myBatchSize,int myBeamSize);

    /* compute the model score for each hypothesis */
    void Score(T2TStateBundle * prev, T2TStateBundle * beam);

    /* generate token indices via beam pruning */
    void Generate(T2TStateBundle * beam);

    /* expand the search graph */
    void Expand(T2TStateBundle * prev, T2TStateBundle * beam);

    /* collect hypotheses with ending symbol */
    void Collect(T2TStateBundle * beam);

    /* fill the hypotheis heap with incomplete hypothses */
    void FillHeap(T2TStateBundle * beam);

    /* save the output sequences in a tensor */
    void Dump(XTensor * output);

    /* check if the token is an end symbol */
    bool IsEnd(int token);

    /* set end symbols for search */
    void SetEnd(const int * tokens, const int tokenNum);

    /* make a mask to prevent duplicated entries in beam expansion for the first position */
    XTensor MakeFirstMask(T2TStateBundle * beam);
};

}

#endif
