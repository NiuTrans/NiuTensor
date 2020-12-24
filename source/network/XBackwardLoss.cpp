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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-17
 */

#include "XBackwardLoss.h"
#include "XNoder.h"
#include "../tensor/XName.h"
#include "../tensor/function/FHeader.h"
#include "../tensor/core/getandset/SetData.h"
#include "../tensor/function/HardTanH.h"
#include "../tensor/function/Identity.h"
#include "../tensor/function/LogSoftmax.h"
#include "../tensor/function/Rectify.h"
#include "../tensor/function/Sigmoid.h"
#include "../tensor/function/Softmax.h"

namespace nts{

/* compute dE/dx of a node */
void XLossGrad::MakeGrad(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    int operID = income.typeID;

    CheckNTErrors(income.tailNum >= 1, "Wrong number of tensors for loss computation!");

    XTensor * output = income.tails[0];
    XTensor * gold = NULL;
    XTensor * weight = NULL;
    XTensor * padding = NULL;
    int leadingDim;

    bool isRoot = XNoder::IsRoot(node);

    if (!isEfficient || output->isGrad) {
        XNoder::MakeGrad(output);
        XTensor * dedy = output->grad;

        if (income.tailNum == 1) {
            dedy->SetDataFixed(1);
            return;
        }

        gold = income.tails[1];

        XTensor* tmp;
        if (!isRoot) {
            tmp = NewTensor(output);
            tmp->SetZeroAll();
        }
        else{
            tmp = dedy;
        }

        if (operID == LOSS_CROSSENTROPY) {
            if (income.tailNum == 3)
                padding = income.tails[2];
            leadingDim = income.GetParamInt(0);
            CheckNTErrors(leadingDim >= 0 && leadingDim < output->order, "wrong leading dimension in logsoftmax!");
            _CrossEntropyBackward(tmp, output, gold, weight, padding, leadingDim);
            if (isRoot)
                gold->DestroyData();
            else
                _SumMe(dedy, tmp);
        }
        else {
            ShowNTErrors("Unsupported backward computation! TODO!");
        }
        
        if (!isRoot)
            DelTensor(tmp);
    }

    node->visitMark = NODE_FINISHED;
}

/* indicates whether the node is for a loss computation */
bool XLossGrad::IsLossOP(XTensor * node)
{
    XLink &income = node->income;
    return (income.typeID & LOSS_BASE) != 0;
}

}