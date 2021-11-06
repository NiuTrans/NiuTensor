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
* We define various template classes here. They will be overloaded and used 
* in applications.
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-02-25
* The meeting at 3:00pm today was canceled. More time for coding.
*/

#ifndef __XNETTEMPLATE_H__
#define __XNETTEMPLATE_H__

#include "../tensor/XTensor.h"
#include "../tensor/XThread.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* 
data distributor template. It distributes batches of data to workers.

The use of data distributor follows:
Start() -> GetBatch() -> ... -> GetBatch() -> End()

In addition, GetBatch() should be thread-safe, and thus could be 
called by different threads simultaneously.
*/
class DataDistributeBase
{
protected:
    /* mutex of batch loading */
    MUTEX_HANDLE loadMutex;

public:
    /* constructor */
    DataDistributeBase();

    /* de-constructor */
    ~DataDistributeBase();

    /* start the job (e.g., open the file).
       NOTE THAT before calling Start() one should initialize
       the distributor if neccessary */
    virtual
    bool Start();

    /* end the job (e.g., close the file) */
    virtual
    bool End();

    /* get a batch of samples */
    virtual
    bool GetBatchSimple(XList * inputs, XList * golds);
    

public:
    /* get a batch of samples */
    bool GetBatch(XList * args);

    /* get a batch of samples (for multi-threading) */
    bool GetBatchSafe(XList * args);
};

}

#endif // __XNETTEMPLATE_H__

