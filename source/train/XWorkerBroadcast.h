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
* The worker that boradcast the lastest parameters from the server to
* the workers.
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-03-03
* Several visiters will come today, so i have less time for coding.
*/

#ifndef __XWORKERBROADCAST_H__
#define __XWORKERBROADCAST_H__

#include "XWorker.h"
#include "XModel.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#define SLEEP_TIME_IN_BROADCASTING 5

/*
data broadcasting method
1) point-to-point
*/
enum DATA_BROADCAST_TYPE { DATA_BROADCAST_P2P };

/* This class defines a broadcaster that transmits parameters from
   a server to workers. */
class XWorkerBroadcast : public XWorker
{
protected:
    DATA_BROADCAST_TYPE broadcastMode;

public:
    /* constructor */
    XWorkerBroadcast();

    /* de-constructor */
    ~XWorkerBroadcast();

    /* set the broadcasting type */
    void SetBroadcastMode(DATA_BROADCAST_TYPE myMode);

    /* broadcast data for a parameter */
    void BroadcastData(XTensorKeeper * source, XList * targetList);

    /* wrapper of BroadcastDataSingle */
    static
    void Broadcast(XList * args);

    /* P2P data broadcasting */
    void BroadcastP2P(XTensor * source, XTensor * target);

    /* add a new job of broadcasting data (for a parameter) */
    bool AddJobBroadcast(XQueue * jobQueue, XTensorKeeper * source, XList * targetList);
};

}

#endif