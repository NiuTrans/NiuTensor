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
*/

#include "XOptimizer.h"
#include "../tensor/core/CHeader.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* constructor */
XOptimizer::XOptimizer()
{
    Clear();
}

/* de-constructor */
XOptimizer::~XOptimizer()
{
    Clear();
}

/* 
initialize the optimizer 
>> config - the configuration
*/
void XOptimizer::Init(XConfig &config)
{
    nstep = config.GetInt("nstep", 100000);
    nepoch = config.GetInt("nepoch", 50);
    lrate = config.GetFloat("lrate", 0.1F);
}

/* clear the optimizer */
void XOptimizer::Clear()
{
    nstep = 0;
    nepoch = 0;
    lrate = 0;
}

/* reset the optimizer (re-start) */
void XOptimizer::Reset()
{
}
    
/* clone the optimizer (with the data in it) */
XOptimizer * XOptimizer::Clone(int devID)
{
    XOptimizer * opt = new XOptimizer();
    
    Copy(this, opt, devID);
    
    return opt;
}
    
/*
copy data
>> source - where we copy the data from
>> target - where we copy the data to
>> devID - the device where place the new data
*/
void XOptimizer::Copy(XOptimizer * source, XOptimizer * target, int devID)
{
    CheckNTErrors(source != NULL, "No input source optimizer");
    CheckNTErrors(target != NULL, "No input source optimizer");
    
    target->nstep = source->nstep;
    target->nepoch = source->nepoch;
    target->lrate = source->lrate;
}

void XOptimizer::ShowSettings()
{
    XPRINT(1, stderr, "[INFO] Optimizer Setup:\n");
    XPRINT2(1, stderr, "%25s = %d\n", "nstep", nstep);
    XPRINT2(1, stderr, "%25s = %d\n", "nepoch", nepoch);
    XPRINT2(1, stderr, "%25s = %.3f\n", "lrate", lrate);
}

/* 
record the update 
>> model - the model that we want to update
*/
void XOptimizer::Note(XModel * model)
{
    nstep++;
}

/* 
update a parameter matrix
>> param - the parameter matrix
>> gard - the gradient
*/
void XOptimizer::UpdateParam(XTensor * param, XTensor * grad)
{
    /* the delta rule
       \theta_new = \theta_old - \grad * \lrate */
    _Sum(param, grad, param, -lrate);
}

/* get learning rate */
float XOptimizer::GetLearningRate()
{
    return lrate;
}

/* 
set learning rate 
>> myLRate - the learning rate that we want to use
*/
void XOptimizer::SetLearningRate(float myLRate)
{
    lrate = myLRate;
}

}
