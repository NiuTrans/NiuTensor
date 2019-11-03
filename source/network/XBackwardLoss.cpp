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

    XNoder::MakeGrad(output);
    XTensor * dedy = output->grad;

    if (income.tailNum == 1) {
        if(dedy->dataType == X_FLOAT)
            _SetDataFixedFloat(dedy, 1.0F);
        else if(dedy->dataType == X_DOUBLE)
            _SetDataFixedDouble(dedy, 1.0);
        else if(dedy->dataType == X_INT)
            _SetDataFixedInt(dedy, 1);
        else
            ShowNTErrors("TODO");

        return;
    }

    gold = income.tails[1];

    if(operID == LOSS_CROSSENTROPY) {
        if (income.tailNum == 3) 
            padding = income.tails[2];
        leadingDim = income.GetParamInt(0);
        CheckNTErrors(leadingDim >= 0 && leadingDim < output->order, "wrong leading dimension in logsoftmax!");
        _CrossEntropyBackward(dedy, output, gold, weight, padding, leadingDim);
    }
    else{
        ShowNTErrors("Wrong activation function type!");
    }

    node->visitMark = NODE_FINISHED;
}

/* indicates whether the node is for a loss computation */
bool XLossGrad::IsLossOP(XTensor * node)
{
    XLink &income = node->income;
    return (income.typeID & LOSS_BASE) != 0;
}

/* 
compute dE/dx for a given function y = f(x) 
>> gold - gold standard to measure error (or loss)
>> y - output of the function
>> x - input of the function
>> dedy - dE/dy
>> dedx - dE/dx
>> funcID - id of the function f
>> params - parameters of the function
>> lossName - name of the loss, e.g., cross entropy
*/
//void XLossGrad::Compute(XTensor * gold, XTensor * y, XTensor * x, 
//                        XTensor * dedy, XTensor * dedx, XTensor * padding,
//                        int funcID, void * params,
//                        LOSS_FUNCTION_NAME lossName)
//{
//    CheckNTErrors(gold && y && x, "Empty input tensors!");
//    CheckNTErrors(dedx, "Empty gradient tensors!");
//    CheckNTErrors((funcID & FUNCTION_BASE) != 0, "Illegal function id");
//
//    if(funcID == FUNC_HARDTANH){
//        _HardTanHBackward(gold, y, x, dedy, dedx, lossName);
//    }
//    else if(funcID == FUNC_IDENTITY){
//        _IdentityBackward(gold, y, x, dedy, dedx, lossName);
//    }
//    else if(funcID == FUNC_LOGSOFTMAX){
//        int leadDim = *(int*)params;
//        _LogSoftmaxBackward(gold, y, x, dedy, dedx, padding, leadDim, lossName);
//    }
//    else if(funcID == FUNC_RECTIFY){
//        _RectifyBackward(gold, y, x, dedy, dedx, lossName);
//    }
//    else if(funcID == FUNC_SIGMOID){
//        _SigmoidBackward(gold, y, x, dedy, dedx, lossName);
//    }else if(funcID == FUNC_SOFTMAX){
//        int leadDim = *(int*)params;
//        _SoftmaxBackward(gold, y, x, dedy, dedx, padding, leadDim, lossName);
//    }
//    else{
//        ShowNTErrors("wrong function found when call the backward process!");
//    }
//
//}

/* 
compute dE/dy for variable y and error(loss) function E
>> gold - gold standard to measure error (or loss)
>> y - output of the function
>> dedy - dE/dy
>> lossName - name of the loss, e.g., cross entropy
*/
//void XLossGrad::Compute(XTensor * gold, XTensor * y, 
//                        XTensor * dedy, XTensor * padding,
//                        LOSS_FUNCTION_NAME lossName)
//{
//    if(gold == NULL){
//        if(dedy->dataType == X_FLOAT)
//            _SetDataFixedFloat(dedy, 1.0F);
//        else if(dedy->dataType == X_DOUBLE)
//            _SetDataFixedDouble(dedy, 1.0);
//        else if(dedy->dataType == X_INT)
//            _SetDataFixedInt(dedy, 1);
//        else{
//            ShowNTErrors("TODO");
//        }
//        return;
//    }
//
//    //_LossBackward(dedy, gold, y, lossName);
//    if(lossName == CROSSENTROPY)
//        _CrossEntropyBackward(dedy, y, gold, NULL, padding);
//
//}

}