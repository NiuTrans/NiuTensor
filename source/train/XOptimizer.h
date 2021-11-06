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
* This class define the template of the update rule in gradient based methods
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-03-01
* March came finally but there was a snow last night.
*/

#ifndef __XOPTIMIZER_H__
#define __XOPTIMIZER_H__

#include "XModel.h"
#include "../tensor/XConfig.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* this class defines a template of the optimizer and 
   implement the simple delta-rule in SGD. */
class XOptimizer
{
public:
    /* update step number */
    int nstep;

    /* training epoch number */
    int nepoch;

    /* learning rate */
    float lrate;

public:
    /* constructor */
    XOptimizer();

    /* de-constructor */
    ~XOptimizer();

    /* initialize the optimizer */
    virtual
    void Init(XConfig &config);

    /* clear the optimizer */
    virtual
    void Clear();

    /* reset the optimizer (re-start) */
    virtual
    void Reset();
    
    /* clone the optimizer (with the data in it) */
    virtual
    XOptimizer * Clone(int devID);
    
    /* copy data */
    virtual
    void Copy(XOptimizer * source, XOptimizer * target, int devID);
    
    /* show settings */
    virtual
    void ShowSettings();

    /* record the update */
    virtual
    void Note(XModel * model);

    /* update a parameter matrix */
    virtual
    void UpdateParam(XTensor * param, XTensor * grad);

    /* get learning rate */
    float GetLearningRate();

    /* set learning rate */
    void SetLearningRate(float myLRate);
};

}

#endif
