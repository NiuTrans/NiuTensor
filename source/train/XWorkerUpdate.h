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
* The worker that updates the model.
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-03-01
*/

#ifndef __XWORKERUPDATE_H__
#define __XWORKERUPDATE_H__

#include "XWorker.h"
#include "XOptimizer.h"
#include "XWorkerBroadcast.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#define SLEEP_TIME_IN_MODEL_UPDATE 5

/* The class defines the model-update worker */
class XWorkerUpdate : public XWorker
{
protected:
    /* the optimizer */
    XOptimizer * optimizer;

public:
    /* constructor */
    XWorkerUpdate();

    /* de-constructor */
    ~XWorkerUpdate();

    /* set the optimizer */
    void SetOptimizer(XOptimizer * myOptimizer);

    /* get the optimizer */
    XOptimizer * GetOptimizer();

    /* update the parameter */
    void UpdateParameter(XTensorKeeper * paramKeeper, XOptimizer * optimizer);

    /* wrapper of UpdateParameter */
    static
    void Update(XList * args);

    /* add a new job of parameter update */
    bool AddJobUpdate(XQueue * jobQueue, XTensorKeeper * paramKeeper, XOptimizer * optimizer);

    /* update a number of parameters simultaneously */
    void UpdateParameterBatch(XList * updaters, XList * paramKeepers, XList * optimizers);

    /* wrapper of UpdateParameterBatch */
    static
    void UpdateBatch(XList * args);

    /* add a new job of parameter update (for a batch)*/
    bool AddJobUpdateBatch(XQueue * jobQueue, XList * updaters, XList * paramKeepers, XList * optimizers);
};

}

#endif
