/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2018, Natural Language Processing Lab, Northestern University. 
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
 */

#ifndef __T2TFNN_H__
#define __T2TFNN_H__

#include "../../tensor/XTensor.h"

using namespace nts;

namespace transformer
{

/* a fnn: y = max(0, x * w1 + b1) * w2 + b2 */
class T2TFNN
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
    
    /* dropout probability */
    DTYPE dropoutP;

public:

    /* constructor */
    T2TFNN();

    /* deconstructor */
    ~T2TFNN();

    /* initialize the model */
    void InitModel(int argc, char ** argv, int myDevID = -1);

    /* make the network */
    XTensor Make(XTensor &input, bool isTraining);

};

}

#endif
