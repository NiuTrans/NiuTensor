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
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-03-01
*/

#include "XWorkerCollect.h"
#include "../tensor/core/CHeader.h"

namespace nts { // namespace nts(NiuTrans.Tensor)


/* constructor */
XWorkerCollect::XWorkerCollect()
{
    type = XWORKER_TYPE_COLLECT;
    collectMode = DATA_COLLECT_P2P;
}

/* de-constructor */
XWorkerCollect::~XWorkerCollect()
{
}

/* set the collection type */
void XWorkerCollect::SetCollectMode(DATA_COLLECT_TYPE myMode)
{
    collectMode = myMode;
}

/* 
P2P data collection
target += source

>> source - the source tensor
>> target - the target tensor
*/
void XWorkerCollect::CollectP2P(XTensor * source, XTensor * target)
{
    CheckNTErrors(source != NULL, "The source tensor should not be NULL!");
    CheckNTErrors(target != NULL, "The target tensor should not be NULL!");
    CheckNTErrors(IsSameShaped(*source, *target), "The two tensors should be of the same shape!");

    /* target += source */
    if (source != target) {
        XTensor * sourceOnSite = source;
        if (source->devID != target->devID) {
            sourceOnSite = new XTensor(target);
            _CopyValues(source, sourceOnSite);
        }

        _Sum(target, sourceOnSite, target);

        if (sourceOnSite != source)
            delete sourceOnSite;
    }
}
    
/*
P2P data collection
target += source

>> source - the source tensor
>> target - the target tensor
>> isGrad - indicates whether we want to collect gradient
*/
void XWorkerCollect::CollectP2P(XTensorKeeper * source, XTensorKeeper * target, const bool isGrad)
{
    CheckNTErrors(source != NULL, "The source tensor keeper should not be NULL!");
    CheckNTErrors(target != NULL, "The target tensor keeper should not be NULL!");

    if(isGrad)
        CollectP2P(source->grad, target->grad);
    else
        CollectP2P(source->tensor, target->tensor);
}
    
/* wrapper of Collect */
void XWorkerCollect::CollectDataP2P(XList * args)
{
    int paramCount = 0;
    
    XWorkerCollect * collecter = (XWorkerCollect*)args->GetItem(paramCount++);
    XTensorKeeper * source = (XTensorKeeper*)args->GetItem(paramCount++);
    XTensorKeeper * target = (XTensorKeeper*)args->GetItem(paramCount++);
    bool isGrad = (bool)args->GetInt(paramCount++);
    
    if(collecter != NULL)
        collecter->CollectP2P(source, target, isGrad);
}

/*
add a new job of collecting data
>> jobQueue - the queue where we run the job
>> source - where we collect the data from
>> target - where we place the data (on the server end)
>> isGrad - indicates whether we want to collect gradient
*/
bool XWorkerCollect::AddJobCollectDataP2P(XQueue * jobQueue, XTensorKeeper * source, XTensorKeeper * target, const bool isGrad)
{
    CheckNTErrors(source != NULL, "No input soure tensor!");
    CheckNTErrors(target != NULL, "No input target tensor!");
    
    XList args;
    args.Add(this);
    args.Add(source);
    args.Add(target);
    args.AddInt((int)isGrad);
    
    XQueue& queueRun = jobQueue != NULL ? *jobQueue : queue;
    
    if (isInstantRun)
        XWorkerCollect::CollectDataP2P(&args);
    else
        queueRun.EnqueueJob((void*)(char*)XWorkerCollect::CollectDataP2P, &args);
    
    return true;
}

/*
add a new job of collecting gradient
>> jobQueue - the queue where we run the job
>> source - where we collect the data from
>> target - where we place the data (on the server end)
*/
bool XWorkerCollect::AddJobCollectGradP2P(XQueue * jobQueue, XTensorKeeper * source, XTensorKeeper * target)
{
    return AddJobCollectDataP2P(jobQueue, source, target, true);
}

/*
add a new job of collecting data in standard tensors
>> jobQueue - the queue where we run the job
>> source - where we collect the data from
>> target - where we place the data (on the server end)
*/
bool XWorkerCollect::AddJobCollectTensorP2P(XQueue * jobQueue, XTensorKeeper * source, XTensorKeeper * target)
{
    return AddJobCollectDataP2P(jobQueue, source, target, false);
}
    
/*
all-reduce: the well-known all-reduce method
every tensor is involved in every data transmition. The final outcome
is that all input tensors share the same value (i.e., the sum of them).

>> all - the tensors for sum
>> isGrad - indicates whether we collect gradient
*/
void XWorkerCollect::CollectAllReduce(XList * all, const bool isGrad)
{
    ShowNTErrors("TODO!");
}
    
/* wrapper of CollectAllReduce via all-reduce */
void XWorkerCollect::CollectDataAllReduce(XList * args)
{
    int paramCount = 0;
    
    XWorkerCollect * collecter = (XWorkerCollect*)args->GetItem(paramCount++);
    XList * all = (XList*)args->GetItem(paramCount++);
    bool isGrad = (bool)args->GetInt(paramCount++);
    
    if(collecter != NULL)
        collecter->CollectAllReduce(all, isGrad);
}

/*
add a new job of collecting data via all-reduce
>> jobQueue - the queue where we run the job
>> all - the tensors for sum
>> isGrad - indicates whether we collect gradient
*/
bool XWorkerCollect::AddJobCollectDataAllReduce(XQueue * jobQueue, XList * all, const bool isGrad)
{
    CheckNTErrors(all != NULL, "No input tensor keeper list!");
    
    XList args;
    args.Add(this);
    args.Add(all);
    args.AddInt((int)isGrad);
    
    XQueue& queueRun = jobQueue != NULL ? *jobQueue : queue;
    
    if (isInstantRun)
        XWorkerCollect::CollectDataAllReduce(&args);
    else
        queueRun.EnqueueJob((void*)(char*)XWorkerCollect::CollectDataAllReduce, &args);
    
    return true;
}

/* add a new job of collecting gradient via all-reduce */
bool XWorkerCollect::AddJobCollectGradAllReduce(XQueue * jobQueue, XList * all)
{
    return AddJobCollectDataAllReduce(jobQueue, all, true);
}

/* add a new job of collecting data in standard tensors via all-reduce */
bool XWorkerCollect::AddJobCollectTensorAllReduce(XQueue * jobQueue, XList * all)
{
    return AddJobCollectDataAllReduce(jobQueue, all, false);
}

}
