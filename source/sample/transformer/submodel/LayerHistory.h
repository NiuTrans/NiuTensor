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
 * $Created by: Bei Li (libei_neu@outlook.com) 2020-02-03
 */

#ifndef __LAYERHISTORY_H__
#define __LAYERHISTORY_H__

#include "LayerNorm.h"
#include "LayerHistory.h"
#include "../../../tensor/function/FHeader.h"

using namespace nts;
using namespace std;

/* the nmt namespace */
namespace nmt
{

#define MAX_LAYER_NUM 50

/* 
the class of history list
*/
class History {
public:
    /* number of elements in the list */
    int count;

    /* the history list */
    XTensor list[MAX_LAYER_NUM];

public: 

    /* contructor */
    History();

    /* de-contructor */
    ~History();

    /* append a layer to the list */
    void Add(XTensor& layer);
};

/* 
the class of layer history
it generates the weighted sum of previous layers
the result for the i-th layer is:
res = sum(layers[0...i] * weight[i][0...i])
*/
class LayerHistory
{
public:
    /* indicates whether train the model */
    bool isTraining;

    /* device id */
    int devID;

    /* the triangle weight matrices for dlcl */
    XTensor* weights;

    /* hidden size */
    int d;

    /* layer number */
    int nlayer;

    /* current layer number */
    int count;

    /* a history to store the value of intimidate layers */
    History* history;

    /* layer normalization for each intimidate layer */
    LayerNorm* layerNorms;

public:
    /* set the training flag */
    void SetTrainingFlag(bool myIsTraining);

    /* constructor */
    LayerHistory();

    /* de-constructor */
    ~LayerHistory();

    /* initialize the model */
    void InitModel(NMTConfig& config, bool isEnc);

    /* add the layer output to the history */
    void Add(XTensor& tensor);

    /* compute the layer input for the current layer, 
       the weight sum of all previous layer output after normed in the history */
    XTensor Pop();

    /* clean the history*/
    void ClearHistory(bool reset=true);
};

} /* end of the nmt namespace */

#endif /* __LAYERHISTORY_H__ */
