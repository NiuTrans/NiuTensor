/*
* NiuTrans.Tensor - an open-source tensor library
* Copyright (C) 2016-2021
* Natural Language Processing Lab, Northeastern University
* and
* NiuTrans Research
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
* We define a class that keeps a tensor (could be either a parameter or 
* gradient).
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-03-25
* I will take the first shot of the COVID-19 vaccine this afternoon.
*/

#ifndef __XTENSORKEEPER_H__
#define __XTENSORKEEPER_H__

#include "../network/XNet.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
parameter state
1) not ready
2) ready
3) the parameter has been collected from other models
4) the updated parameter
*/
enum PARAM_STATE {
    PARAM_STATE_NOT_READY,
    PARAM_STATE_READY,
    PARAM_STATE_COLLECTED,
    PARAM_STATE_UPDATED
};

/* tensor keeper */
class XTensorKeeper
{
public:
    /* the parameter */
    XTensor * tensor;

    /* the gradient */
    XTensor * grad;

    /* the parameter state */
    PARAM_STATE flag;

    /* the state of the entire training process
    (choosing from PARAM_STATE_NOT_READY and
    PARAM_STATE_UPDATED */
    PARAM_STATE trainFlag;

    /* a mutex for locking and unlocking the parameter */
    MUTEX_HANDLE accessLock;

    /* a mutex of the overall training */
    MUTEX_HANDLE trainLock;

public:
    /* constructor */
    XTensorKeeper();

    /* constructor */
    ~XTensorKeeper();
};
}

#endif