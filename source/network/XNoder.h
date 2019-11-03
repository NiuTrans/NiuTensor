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
 * low-level utilities
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-18
 */

#include "../tensor/core/CHeader.h"

#ifndef __XNODER_H__
#define __XNODER_H__

namespace nts{

#define NODE_UNFINISHED 0
#define NODE_DOING      1
#define NODE_FINISHED   2

/* node management */
class XNoder
{
public:
    /* make gradient tensor for a node */
    static
    void MakeGrad(XTensor * node);

    /* the node is a leaf node (intput) or not */
    static
    bool IsLeaf(XTensor * node);

    /* the node is a root node (output) or not */
    static
    bool IsRoot(XTensor * node);

    /* the node keeps the gradinent or not */
    static
    bool IsGrad(XTensor * node);
};

}

#endif