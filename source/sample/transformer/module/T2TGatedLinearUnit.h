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


#ifndef __GLU_H__
#define __GLU_H__

#include "T2TLayerNormal.h"
#include "T2TGatedLinearUnit.h"

using namespace nts;

namespace transformer
{

/* a fnn: y = max(0, x * w1 + b1) * w2 + b2 */
class GLU
{
public:
    /* device id */
    int devID;

    /* size of input vector */
    int inSize;

    /* size of output vector */
    int outSize;

    /* size of hidden layers */
    int hSize;

    /* matrix of transformation 1 */
    XTensor w1;

    /* bias of transformation 1 */
    XTensor b1;

    /* matrix of transformation 2 */
    XTensor w2;

    /* bias of transformation 2 */
    XTensor b2;

public:

    /* constructor */
    GLU();

    /* de-constructor */
    ~GLU();

    /* initialize the model */
    void InitModel(T2TConfig& config);

    /* make the network */
    XTensor Make(XTensor& input);
};

}

#endif