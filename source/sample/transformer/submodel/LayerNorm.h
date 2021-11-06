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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-31
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04
 */

#ifndef __LAYERNORMAL_H__
#define __LAYERNORMAL_H__

#include "../Config.h"
#include "../../../network/XNet.h"

using namespace nts;

/* the nmt namespace */
namespace nmt
{

/* layer normalization: y = norm(x) * w + b
   where norm(x) = (x - mean)/standardDeviation */
class LayerNorm
{
public:
    /* indicates whether use L1-Norm for normalization */
    bool isL1Normed;

    /* indicates whether train the model */
    bool isTraining;

    /* device id */
    int devID;

    /* dimension size of the model */
    int d;

    /* the transformation matrix w */
    XTensor weight;

    /* the bias term b */
    XTensor bias;

public:

    /* constructor */
    LayerNorm();

    /* de-constructor */
    ~LayerNorm();

    /* run layernorm (wrapper) */
    XTensor Run(XTensor& input);

    /* run layernorm with L2-Norm */
    XTensor RunL2Norm(XTensor& input);

    /* run layernorm with L1-Norm */
    XTensor RunL1Norm(XTensor& input);

    /* set the training flag */
    void SetTrainingFlag(bool myIsTraining);

    /* initialize the model */
    void InitModel(NMTConfig& config, int myDevID, int hiddenSize, bool myL1Normed);
};

} /* end of the nmt namespace */

#endif /* __LAYERNORMAL_H__ */