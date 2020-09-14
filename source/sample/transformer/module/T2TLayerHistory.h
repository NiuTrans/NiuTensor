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
 * $Created by: Bei Li (libei_neu@outlook.com) 2020-02-03
 */

#ifndef __LAYERHISTORY_H__
#define __LAYERHISTORY_H__

#include "T2TLayerNormal.h"
#include "T2TLayerHistory.h"

#include "../../../tensor/function/FHeader.h"

using namespace nts;

namespace transformer
{

/*
multi-head attention
y(Q, K, V) = cat(head_1, head_2, ..., head_n)
where head_i = Attention(Q * w_i^Q, K * w_i^K, V * w_i^V)
      attention(Q, K, V) = softmax(Q * K^T/d_k^0.5) V
      d_k = dimension size of K
*/
class LayerHistory
{
public:
    /* device id */
    int devID;

    /* the triangle weight matrix for dlcl */
    XTensor weight;

    /* hidden size */
    int d;

    /* layer number */
    int nlayer;

    /* current layer number */
    int count;

    /* a history to store the value of intimidate layers */
    TensorList history;

    /* layer normalization for each intimidate layer */
    T2TLN* layerNorms;

public:
    /* constructor */
    LayerHistory();

    /* de-constructor */
    ~LayerHistory();

    /* initialize the model */
    void InitModel(T2TConfig& config);

    /* add the layer output to the history */
    void Add(XTensor& tensor);

    /* compute the layer input for the current layer, the weight sum of all previous layer output after normed in the history */
    XTensor Pop();

    /* clean the history*/
    void ClearHistory();
};

}

#endif
