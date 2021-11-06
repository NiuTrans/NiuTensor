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
* The worker that collects data from workers.
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-03-02
* minus 10 degrees centigrade comes again!
*/

#ifndef __XWORKERCOLLECT_H__
#define __XWORKERCOLLECT_H__

#include "XWorker.h"
#include "XModel.h"
#include "XWorkerJob.h"
#include "XWorkerUpdate.h"
#include "XWorkerBroadcast.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#define SLEEP_TIME_IN_COLLECTING 5
#define SLEEP_TIME_IN_COLLECTING_OTHER 5

/*
data collection method
1) point-to-point
2) reduce sum
3) all-reduce
*/
enum DATA_COLLECT_TYPE { DATA_COLLECT_P2P, DATA_COLLECT_REDUCESUM};

/* The class defines the collecting-data worker. It collect (gradient) data
   from workers for the leader (server). */
class XWorkerCollect : public XWorker
{
protected:
    DATA_COLLECT_TYPE collectMode;

public:
    /* constructor */
    XWorkerCollect();

    /* de-constructor */
    ~XWorkerCollect();

    /* set the collection type */
    void SetCollectMode(DATA_COLLECT_TYPE myMode);

    /* P2P data collection */
    void CollectP2P(XTensor * source, XTensor * target);
    
    /* P2P data collection */
    void CollectP2P(XTensorKeeper * source, XTensorKeeper * target, const bool isGrad);

    /* wrapper of CollectP2P */
    static
    void CollectDataP2P(XList * args);
    
    /* add a new job of collecting data via p2p data transmission */
    bool AddJobCollectDataP2P(XQueue * jobQueue, XTensorKeeper * source, XTensorKeeper * target, const bool isGrad);
    
    /* add a new job of collecting gradient via p2p data transmission */
    bool AddJobCollectGradP2P(XQueue * jobQueue, XTensorKeeper * source, XTensorKeeper * target);
    
    /* add a new job of collecting data in standard tensors via p2p data transmission */
    bool AddJobCollectTensorP2P(XQueue * jobQueue, XTensorKeeper * source, XTensorKeeper * target);
    
    /* all-reduce data collection via all-reduce */
    void CollectAllReduce(XList * all, const bool isGrad);
    
    /* wrapper of CollectAllReduce via all-reduce */
    static
    void CollectDataAllReduce(XList * args);
    
    /* add a new job of collecting data via all-reduce */
    bool AddJobCollectDataAllReduce(XQueue * jobQueue, XList * all, const bool isGrad);
    
    /* add a new job of collecting gradient via all-reduce */
    bool AddJobCollectGradAllReduce(XQueue * jobQueue, XList * all);
    
    /* add a new job of collecting data in standard tensors via all-reduce */
    bool AddJobCollectTensorAllReduce(XQueue * jobQueue, XList * all);

};

}

#endif
