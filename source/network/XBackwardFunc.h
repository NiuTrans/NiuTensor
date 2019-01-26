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
 * backward computation for activation function
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-18
 * Dingdang won 5 games in the GO training yesterday, hahaha ...
 */

#include "../tensor/XTensor.h"
#include "../tensor/function/FHeader.h"

#ifndef __XBACKWARDFUNC_H__
#define __XBACKWARDFUNC_H__

namespace nts{

/* this class computes the gradient for activation functions given a node */
class XFuncGrad
{
public:
    /* compute dE/dx of a node */
    static
    void MakeGrad(XTensor * node, bool isEfficient);

    /* indicates whether the node is for an activation function */
    static
    bool IsFunc(XTensor * node);
};

}

#endif