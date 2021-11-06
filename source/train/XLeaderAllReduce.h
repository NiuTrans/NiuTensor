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
* A server of (ring-based) all reduce (AR) training. Unlike the parameter server (as
* in XLeaderPS), the AR training enjoys less the data transmission time across devices by 
* the ring-based all reduce algorithm used in high-performance computation. In this way,
* collecting gradient from workers is much more efficient. Since all workers share the
* same gradient after all-reduce, the update can be done on the worker side. This further
* saves time because we do not need to broadcast the latest parameter from the server to 
* the workers.
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-03-30
* I have more time for coding today because the meeting was cancelled :) Why there are so many
* meetings?
*/

#ifndef __XLEADERALLREDUCE_H__
#define __XLEADERALLREDUCE_H__

#include "XLeader.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* parameter server */
class XLeaderAllReduce : public XLeader
{
protected:
    /* optimizer for each model */
    XList optimizers;
    
public:
	/* constructor */
	XLeaderAllReduce();

	/* deconstructor */
	~XLeaderAllReduce();
    
    /* clear */
    void Clear();

    /* create workers and other stuff used in training */
    void MakeAll(XConfig * config, XModel * model, XOptimizer * optimizer, const int * devIDs, const int jobWorkerNum);

    /* wait for finished states (i.e., all workers finish their jobs) */
    void WaitForFinishing(const int * activeJobWorkers, const int isToUpdate);

    /* run the model and update it (for one time) */
    bool Run(XConfig* config, DataDistributeBase* dataDistributor, XOptimizer* optimizer);

    /* run the model */
    int RunModel(XConfig* config, DataDistributeBase* dataDistributor, int* active);

    /* update the model */
    void RunUpdate(XConfig* config, XOptimizer* optimizer, const int* active);
};

}

#endif
