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
* An implementation of the Adam optimizer.
* 
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-03-15
* A foggy day. But all my students come back for work after the holiday
* - full of happiness to see a new start.
*/

#ifndef __ADAM_H__
#define __ADAM_H__

#include "../XOptimizer.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* an implementation of the Adam optimizer */
class Adam : public XOptimizer
{
protected:
    /* list of the moment of the parameter matrices */
    TensorList moments;

    /* list of the 2nd order moment of the parameter matrices */
    TensorList moments2nd;

    /* hyper parameters of Adam */
    float adamBeta1;
    float adamBeta2;
    float adamDelta;
    float adamBeta1T;
    float adamBeta2T;

public:
    /* constructor */
    Adam();

    /* de-constructor */
    ~Adam();

    /* initialize the optimizer */
    void Init(XConfig &config);

    /* clear the optimizer */
    void Clear();

    /* reset the optimizer (re-start) */
    void Reset();
    
    /* clone the optimizer (with the data in it) */
    XOptimizer * Clone(int devID);
    
    /* copy data */
    void Copy(XOptimizer * source, XOptimizer * target, int devID);

    /* show settings */
    void ShowSettings();

    /* record the update */
    void Note(XModel * model);

    /* update a parameter matrix */
    void UpdateParam(XTensor * param, XTensor * grad, int pid);

};

}

#endif
