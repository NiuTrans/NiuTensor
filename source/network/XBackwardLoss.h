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
 * My students worked all night to prepare a submission to CWMT. Good luck
 * to them!
 */

#include "../tensor/XTensor.h"
#include "../tensor/function/FHeader.h"
#include "../tensor/loss/LHeader.h"

#ifndef __XBACKWARDLOSS_H__
#define __XBACKWARDLOSS_H__

namespace nts{

/* this class computes the gradient (of a output node) 
   with respect to the loss */
class XLossGrad
{
public:
    /* compute dE/dx of a node */
    static
    void MakeGrad(XTensor * node, bool isEfficient);

    /* indicates whether the node is for a Loss computation */
    static
    bool IsLossOP(XTensor * node);

    ///* compute dE/dx for a given function y = f(x) */
    //void Compute(XTensor * gold, XTensor * y, XTensor * x, 
    //             XTensor * dedy, XTensor * dedx, XTensor * padding,
    //             int funcID, void * params,
    //             LOSS_FUNCTION_NAME lossName);

    /* compute dE/dy for variable y and error(loss) function E */
    void Compute(XTensor * gold, XTensor * y, 
                 XTensor * dedy, XTensor * padding,
                 LOSS_FUNCTION_NAME lossName);
};

}

#endif