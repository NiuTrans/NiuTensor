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

#include <cmath>

#include "T2TUtility.h"
#include "T2TEmbedding.h"
#include "T2TLayerNormal.h"
#include "T2TLayerHistory.h"

#include "../../../tensor/core/CHeader.h"

#define SAFE_DELETE(x) do{ if((x) != NULL){delete (x); (x) = NULL;} } while(false)
#define SAFE_DELETE_ARRAY(x) do{ if((x) != NULL) {delete [] (x); (x)=NULL;} } while(false)

namespace transformer
{

/* constructor */
LayerHistory::LayerHistory()
{
    d = -1;
    count = -1;
    weight = NULL;
    layerNorms = NULL;
}

/* de-constructor */
LayerHistory::~LayerHistory()
{
    history.Clear();
    delete[] layerNorms;
}

/*
initialize the model
>> config - configurations of the model
*/
void LayerHistory::InitModel(T2TConfig& config)
{
    devID = config.devID;
    d = config.modelSize;
    nlayer = config.nEncLayer;

    InitTensor2D(&weight, nlayer + 1, nlayer + 1, X_FLOAT, devID);

    layerNorms = new T2TLN[nlayer];

    /* initialize the layer normalization of each layer */
    for (int i = 0; i < nlayer; i++) {
        layerNorms[i].InitModel(config);
    }
}

/*
the Add operation
>> tensor - the previous layer output. It might be of size B * L * H
            where B = batch size, L = sequence length,
            and H = vector size of each position
*/
void LayerHistory::Add(XTensor& tensor)
{
    /* the embedding is not normed */
    count += 1;
    if (history.Size() == 0) {

        //sample_ = tensor;
        history.Add(&tensor);
        return;
    }
    XTensor ln = layerNorms[count - 2].Make(tensor);
    history.Add(&ln);
}

/*
generate the weight sum vector of all previous layer output in the history as the layer input
*/
XTensor LayerHistory::Pop()
{
    /* the number of layer output in the history */
    size_t size = history.Size();

    TensorList historyList;
    for (size_t i = 0; i < size; i++)
        historyList.Add(history[i]);

    /* we need stack the tensor along the first dim*/
    XTensor stackTensor = Stack(historyList, 0);

    XTensor interWeight;

    InitTensor2D(&interWeight, 1, weight.dimSize[1], DEFAULT_DTYPE, devID);
    XTensor layerWeight;
    InitTensor1D(&layerWeight, size, DEFAULT_DTYPE, devID);

    _SelectRange(&weight, &interWeight, 0, size - 1, size);
    interWeight.Reshape(interWeight.unitNum);
    _SelectRange(&interWeight, &layerWeight, 0, 0, size);
    MultiplyDimMe(stackTensor, layerWeight, 0);

    XTensor result;
    ReduceSum(stackTensor, result, 0);

    return result;
}

void LayerHistory::ClearHistory()
{
    history.Clear();
}

}