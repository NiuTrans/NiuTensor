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

#ifndef __OUTPUT_H__
#define __OUTPUT_H__

#include <memory>
#include "../Config.h"
#include "../../../tensor/function/FHeader.h"

using namespace nts;

/* the nmt namespace */
namespace nmt
{

/* output layer */
class OutputLayer
{
public:
    /* indicates whether share decoder embeddings and output weights */
    bool shareDecInputOutputEmb;

    /* indicates whether train the model */
    bool isTraining;

    /* device id */
    int devID;

    /* vocabulary size */
    int vSize;

    /* vector size of the linear transformation */
    int hSize;

    /* transformation matrix */
    XTensor* w;

public:
    /* set the training flag */
    void SetTrainingFlag(bool myIsTraining);

    /* constructor */
    OutputLayer();

    /* de-constructor */
    ~OutputLayer();

    /* initialize the model */
    void InitModel(NMTConfig& config);

    /* make the network */
    XTensor Make(XTensor& input, bool normalized);
};

} /* end of the nmt namespace */

#endif /* __OUTPUT_H__ */