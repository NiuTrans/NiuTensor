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
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-03-16
* I wore my coat again after the rain yesterday.
*/

#include "XLearningRate.h"
#include <math.h>

namespace nts { // namespace nts(NiuTrans.Tensor)

/* constructor */
XLearningRate::XLearningRate()
{
}

/* de-constructor */
XLearningRate::~XLearningRate()
{
}

/* a Transformer-style scheduler. For more details, see
"Attention is all need" by Vaswani at al. 
>> lrate - the learning rate
>> nstep - the update step number 
>> nwarmup - the warmup step number 
*/
float XLearningRate::MakeLRTransformer(const float lrate, const int nstep, const int nwarmup, const float warmupInitLR)
{
    float lr = 0.0F;
    float warmupEndLR = lrate;
    float lrStep = (warmupEndLR - warmupInitLR) / nwarmup;
    float decayFactor = warmupEndLR * pow(float(nwarmup), 0.5F);

    /* learning rate, scheduled by inverse square root */
    if (nstep < nwarmup)
        lr = warmupInitLR + nstep * lrStep;
    else
        lr = decayFactor * pow((float)nstep, -0.5F);

    return lr;
}

}