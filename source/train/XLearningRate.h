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
* This a learning rate generator. E.g., one can adjust learning rate as
* the training process proceeds.
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-03-16
* I wore my coat again after the rain yesterday.
*/

#ifndef __XLEARNINGRATE_H__
#define __XLEARNINGRATE_H__

namespace nts { // namespace nts(NiuTrans.Tensor)

/* Learning rate scheduler */
class XLearningRate
{
public:
    /* constructor */
    XLearningRate();

    /* de-constructor */
    ~XLearningRate();

    /* a Transformer-style scheduler */
    float MakeLRTransformer(const float lrate, const int nstep, const int nwarmup, const float warmupInitLR);
};

}

#endif