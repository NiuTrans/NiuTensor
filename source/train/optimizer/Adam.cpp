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
*/

#include "Adam.h"
#include "../../tensor/core/CHeader.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* constructor */
Adam::Adam() : XOptimizer()
{
    Clear();
}

/* de-constructor */
Adam::~Adam()
{
    Clear();
}

/*
initialize the optimizer
>> config - the configuration
*/
void Adam::Init(XConfig &config)
{
    XOptimizer::Init(config);

    adamBeta1 = config.GetFloat("adambeta1", 0.9F);
    adamBeta2 = config.GetFloat("adambeta2", 0.98F);
    adamDelta = config.GetFloat("adamdelta", 1e-9F);
}

/* clear the optimizer */
void Adam::Clear()
{
    XOptimizer::Clear();

    for (int i = 0; i < moments.count; i++) {
        XTensor * m = moments[i];
        delete m;
    }
    moments.Clear();

    for (int i = 0; i < moments2nd.count; i++) {
        XTensor * m2nd = moments2nd[i];
        delete m2nd;
    }
    moments2nd.Clear();

    adamBeta1T = 1.0F;
    adamBeta2T = 1.0F;
}

/* reset the optimizer (re-start) */
void Adam::Reset()
{
    for (int i = 0; i < moments.count; i++) {
        XTensor * m = moments[i];
        m->SetZeroAll();
    }

    for (int i = 0; i < moments2nd.count; i++) {
        XTensor * m2nd = moments2nd[i];
        m2nd->SetZeroAll();
    }

    adamBeta1T = 1.0F;
    adamBeta2T = 1.0F;
}
    
/*
clone the optimizer (with the data in it)
>> devID - device where we place the data
*/
XOptimizer * Adam::Clone(int devID)
{
    Adam * opt = new Adam();
    
    Copy(this, opt, devID);
    
    return (XOptimizer*)opt;
}
    
/* copy data */
void Adam::Copy(XOptimizer * source, XOptimizer * target, int devID)
{
    XOptimizer::Copy(source, target, devID);
    
    Adam * s = (Adam*)source;
    Adam * t = (Adam*)target;
    
    t->adamBeta1 = s->adamBeta1;
    t->adamBeta2 = s->adamBeta2;
    t->adamDelta = s->adamDelta;
    t->adamBeta1T = s->adamBeta1T;
    t->adamBeta2T = s->adamBeta2T;
    
    t->moments.Clear();
    for(int i = 0; i < s->moments.count; i++){
        XTensor * st = s->moments[i];
        XTensor * stNew = new XTensor();
        InitTensorV2(stNew, st->order, st->dimSize, st->dataType, st->denseRatio, devID);
        _CopyValues(st, stNew);
        t->moments.Add(stNew);
    }
    
    t->moments2nd.Clear();
    for(int i = 0; i < s->moments2nd.count; i++){
        XTensor * st = s->moments2nd[i];
        XTensor * stNew = new XTensor();
        InitTensorV2(stNew, st->order, st->dimSize, st->dataType, st->denseRatio, devID);
        _CopyValues(st, stNew);
        t->moments2nd.Add(stNew);
    }
}

/* show settings */
void Adam::ShowSettings()
{
    XPRINT(1, stderr, "[INFO] Optimizer = Adam\n");
    XOptimizer::ShowSettings();
    XPRINT2(1, stderr, "%25s = %f\n", "adambeta1", adamBeta1);
    XPRINT2(1, stderr, "%25s = %f\n", "adambeta2", adamBeta2);
    XPRINT2(1, stderr, "%25s = %f\n", "adamdelta", adamDelta);
}

/* record the update */
void Adam::Note(XModel * model)
{
    nstep++;
}

/* 
update a parameter matrix using Adam
>> param - the parameter to update
>> grad - the gradient of the parameter
>> pid - index of the parameter
*/
void Adam::UpdateParam(XTensor * param, XTensor * grad, int pid)
{
    adamBeta1T *= adamBeta1;
    adamBeta2T *= adamBeta2;
    float e = lrate * (float)sqrt(1 - adamBeta2T) / (1 - adamBeta1T);
    float d = adamDelta * (float)sqrt(1 - adamBeta2T);

    /* m = beta_1 * m + (1-beta_1) * grad */
    XTensor * m = moments[pid];
    _ScaleAndShiftMe(m, adamBeta1, 0);
    _Sum(m, grad, m, (1.0F - adamBeta1));

    /* v = beta_2 * v + (1-beta_2) * grad * grad*/
    XTensor * v = moments2nd[pid];
    _Multiply(grad, grad, v, adamBeta2 / (1.0F - adamBeta2));
    _ScaleAndShiftMe(v, (1.0F - adamBeta2), 0);

    /* allocate a piece of buffer memory */
    GMems.GetMem(v->devID)->LockBuf();
    XTensor* v2 = NewTensorBuf(v, v->devID);

    /* v2 = m / (sqrt(v) + delta) */
    _Power(v, v2, 0.5F);
    _ScaleAndShiftMe(v2, 1.0F, d);
    _Div(m, v2, v2);

    /* the delta rule */
    _Sum(param, v2, param, -e);

    /* release a piece of buffer memory */
    DelTensorBuf(v2);
    GMems.GetMem(v->devID)->UnlockBuf();
}

}
