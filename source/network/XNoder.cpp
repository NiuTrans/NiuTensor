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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-18
 */

#include "XNoder.h"

namespace nts{

/* make gradient tensor for a node */
void XNoder::MakeGrad(XTensor * node)
{
    if(node == NULL)
        return;

    if(!_IsSameShaped(node, node->grad)){
        delete node->grad;
        node->grad = NewTensor(node);
        node->grad->SetZeroAll();
    }
}

/* the node is a leaf node (intput) or not */
bool XNoder::IsLeaf(XTensor * node)
{
    if(node == NULL)
        return false;

    if(node->income.tailNum == 0)
        return true;
    else
        return false;
}

/* the node is a root node (output) or not */
bool XNoder::IsRoot(XTensor * node)
{
    if(node == NULL)
        return false;

    if(node->outgo.tailNum == 0)
        return true;
    else
        return false;
}

/* the node keeps the gradinent or not */
bool XNoder::IsGrad(XTensor * node)
{
    if(node == NULL)
        return false;
    
    if(node->isGrad)
        return true;
    else
        return false;
}

}