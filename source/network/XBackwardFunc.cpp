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

#include "XNoder.h"
#include "XBackwardFunc.h"
#include "../tensor/XName.h"
#include "../tensor/function/FHeader.h"

namespace nts{

/* compute dE/dx of a node */
void XFuncGrad::MakeGrad(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    int operID = income.typeID;

    CheckNTErrors(node->grad != NULL, "No gradient found!");
    CheckNTErrors(income.tailNum == 1, "Too many input tensors for the function!");

    XTensor * input = income.tails[0];
    XTensor * output = node;

    XNoder::MakeGrad(input);

    if(operID == FUNC_HARDTANH)
        _HardTanHBackward(output, input, output->grad, input->grad);
    else if(operID == FUNC_IDENTITY)
        _IdentityBackward(output, input, output->grad, input->grad);
    else if(operID == FUNC_LOGSOFTMAX){
        int leadDim = income.GetParamInt(0);
        CheckNTErrors(leadDim >= 0 && leadDim < input->order, "wrong leading dimension in logsoftmax!");
        _LogSoftmaxBackward(NULL, output, input, output->grad, input->grad, NULL, leadDim, NOLOSS);
    }
    else if(operID == FUNC_RECTIFY)
        _RectifyBackward(output, input, output->grad, input->grad);
    else if(operID == FUNC_SIGMOID)
        _SigmoidBackward(output, input, output->grad, input->grad);
    else if(operID == FUNC_SOFTMAX){
        int leadDim = income.GetParamInt(0);
        CheckNTErrors(leadDim >= 0 && leadDim < input->order, "wrong leading dimension in softmax!");
        _SoftmaxBackward(NULL, output, input, output->grad, input->grad, NULL, leadDim, NOLOSS);
    }
    else{
        ShowNTErrors("Wrong activation function type!");
    }

    node->visitMark = NODE_FINISHED;
}

/* indicates whether the node is for an activation function */
bool XFuncGrad::IsFunc(XTensor * node)
{
    XLink &income = node->income;
    return (income.typeID & FUNCTION_BASE) != 0;
}

}
