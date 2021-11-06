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
*/

#include "XTensorKeeper.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* constructor */
XTensorKeeper::XTensorKeeper()
{
    tensor = NULL;
    grad = NULL;
    flag = PARAM_STATE_NOT_READY;
    trainFlag = PARAM_STATE_NOT_READY;
    MUTEX_INIT(accessLock);
    MUTEX_INIT(trainLock);
}

/* constructor */
XTensorKeeper::~XTensorKeeper()
{
    MUTEX_DELE(accessLock);
    MUTEX_DELE(trainLock);
}

}