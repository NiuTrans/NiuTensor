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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-31
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04
 */

#ifndef __T2TOUTPUT_H__
#define __T2TOUTPUT_H__

#include "T2TUtility.h"
#include "../../../tensor/function/FHeader.h"

using namespace nts;

namespace transformer
{

/* output layer */
class T2TOutput
{
public:
    /* device id */
    int devID;

    /* vocabulary size */
    int vSize;

    /* vector size of the linear transformation */
    int hSize;

    /* transformation matrix */
    XTensor w;

public:
    /* constructor */
    T2TOutput();

    /* de-constructor */
    ~T2TOutput();

    /* initialize the model */
    void InitModel(T2TConfig& config);

    /* make the network (redefined output tensor) */
    void Make(XTensor& input, XTensor& output, bool isTraining, bool normalized);
};

}

#endif