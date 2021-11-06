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
*/


#include "XWorkerBroadcast.h"
#include "../tensor/core/CHeader.h"

namespace nts { // namespace nts(NiuTrans.Tensor)


/* constructor */
XWorkerBroadcast::XWorkerBroadcast()
{
    type = XWORKER_TYPE_BROADCAST;
}

/* de-constructor */
XWorkerBroadcast::~XWorkerBroadcast()
{
}

/* set the broadcasting type */
void XWorkerBroadcast::SetBroadcastMode(DATA_BROADCAST_TYPE myMode)
{
    broadcastMode = myMode;
}

/* 
broadcast data for a parameter 
>> source - the data (as a model) that we want to broadcast
>> targetList - the target places where we recieve the data
*/
void XWorkerBroadcast::BroadcastData(XTensorKeeper * source, XList * targetList)
{
    CheckNTErrors(source->flag == PARAM_STATE_UPDATED,
                  "The parameter is not ready for broadcasting");

    for (int i = 0; i < targetList->count; i++) {
        XTensorKeeper * target = (XTensorKeeper*)targetList->GetItem(i);

        /* data transmit */
        BroadcastP2P(source->tensor, target->tensor);

        /* update the flag */
        target->flag = PARAM_STATE_UPDATED;
    }
}

/* 
wrapper of BroadcastDataSingle 
>> args - the list of arguments
*/
void XWorkerBroadcast::Broadcast(XList * args)
{
    int paramCount = 0;

    XWorkerBroadcast * broadcaster = (XWorkerBroadcast*)args->GetItem(paramCount++);
    XTensorKeeper * source = (XTensorKeeper*)args->GetItem(paramCount++);

    /* target models */
    int targetNum = args->GetItemInt(paramCount++);
    XList target;
    for (int i = 0; i < targetNum; i++) {
        XTensorKeeper * model = (XTensorKeeper*)args->GetItem(paramCount++);
        target.Add(model);
    }

    broadcaster->BroadcastData(source, &target);
}

/* 
P2P data broadcasting 
>> source - the source data
>> target - the target data
*/
void XWorkerBroadcast::BroadcastP2P(XTensor * source, XTensor * target)
{
    CheckNTErrors(source != NULL, "The source tensor should not be NULL!");
    CheckNTErrors(target != NULL, "The target tensor should not be NULL!");
    CheckNTErrors(IsSameShaped(*source, *target), "The two tensors should be of the same shape!");

    if(source != target)
        CopyValues(*source, *target);
}

/* 
add a new job of broadcasting data (for a parameter)
>> jobQueue - the queue where we push jobs
>> source - the data that we want to broadcast
>> targetList - the target places where we recieve the data
*/
bool XWorkerBroadcast::AddJobBroadcast(XQueue * jobQueue, XTensorKeeper * source, XList * targetList)
{
    CheckNTErrors(source != NULL, "no input source tensor!");
    CheckNTErrors(targetList != NULL, "no input target tensor list!");

    XList args;
    args.Add(this);
    args.Add(source);
    args.AddInt(targetList->count);
    args.AddList(targetList);

    XQueue& queueRun = jobQueue != NULL ? *jobQueue : queue;

    if (isInstantRun)
        XWorkerBroadcast::Broadcast(&args);
    else
        queueRun.EnqueueJob((void*)(char*)XWorkerBroadcast::Broadcast, &args);

    return true;
}

}
