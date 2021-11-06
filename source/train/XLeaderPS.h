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
* The parameter server mode for distributed training. The server (i.e., XLeader)
* collect gradient from each worker. After the update of the parameters, it
* broadcast the lastest parameters to all the workers. NOTE that the training time
* would incease significantly if there are a large model and a large number of
* workers.
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-03-30
* Sandstorm has finally passed. It is sunny today :)
*/

#ifndef __XLEADERPS_H__
#define __XLEADERPS_H__

#include "XLeader.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* parameter server */
class XLeaderPS : public XLeader
{
public:
	/* constructor */
	XLeaderPS();

	/* deconstructor */
	~XLeaderPS();

    /* create workers and other stuff used in training */
    void MakeAll(XConfig * config, XModel * model, const int * devIDs, const int jobWorkerNum);

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