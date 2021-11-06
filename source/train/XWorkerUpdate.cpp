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

#include "XWorkerUpdate.h"

namespace nts { // namespace nts (NiuTrans.Tensor)

/* constructor */
XWorkerUpdate::XWorkerUpdate()
{
    type = XWORKER_TYPE_UPDATE;
    optimizer = NULL;
}

/* de-constructor */
XWorkerUpdate::~XWorkerUpdate()
{
}

/* set the optimizer */
void XWorkerUpdate::SetOptimizer(XOptimizer * myOptimizer)
{
    optimizer = myOptimizer;
}

/* get the optimizer */
XOptimizer * XWorkerUpdate::GetOptimizer()
{
    return optimizer;
}

/* 
update a parameter of a model 
>> paramKeeper -  the parameter keeper
>> optimizer - the optimizer
*/
void XWorkerUpdate::UpdateParameter(XTensorKeeper * paramKeeper, XOptimizer * optimizer)
{

    CheckNTErrors(paramKeeper->flag == PARAM_STATE_COLLECTED, "The state of the parameter is wrong!");

    XTensor * param = paramKeeper->tensor;
    XTensor * grad = paramKeeper->grad;

    CheckNTErrors(grad != NULL, "No gradient!");

    /* update the parameter */
    optimizer->UpdateParam(param, grad);

    /* set the flag */
    paramKeeper->flag = PARAM_STATE_UPDATED;
}

/* 
wrapper of UpdateParameter 
>> args - arguments of the update
*/
void XWorkerUpdate::Update(XList * args)
{
    int paramCount = 0;

    CheckNTErrors(args != NULL && args->count == 3, "Illegal argument list!");

    XWorkerUpdate * updater = (XWorkerUpdate*)args->GetItem(paramCount++);
    XTensorKeeper * paramKeeper = (XTensorKeeper*)args->GetItem(paramCount++);
    XOptimizer * optimizer = (XOptimizer*)args->GetItem(paramCount++);

    CheckNTErrors(updater != NULL, "No updater!");

    updater->UpdateParameter(paramKeeper, optimizer);
}

/* 
add a new job of model update (for a parameter) 
>> jobQueue - the queue for sub-jobs executed in the job
>> paramKeeper -  the parameter keeper
>> optimizer - the optimizer
*/
bool XWorkerUpdate::AddJobUpdate(XQueue * jobQueue,
                                 XTensorKeeper * paramKeeper,
                                 XOptimizer * optimizer)
{
    CheckNTErrors(paramKeeper != NULL, "No input parameter keeper!");
    CheckNTErrors(optimizer != NULL, "No optimizer!");

    XList args;
    args.Add(this);
    args.Add(paramKeeper);
    args.Add(optimizer);
    
    XQueue& queueRun = jobQueue != NULL ? *jobQueue : queue;

    if (isInstantRun)
        XWorkerUpdate::Update(&args);
    else
        queueRun.EnqueueJob((void*)(char*)XWorkerUpdate::Update, &args);

    return true;
}

/* 
update a number of parameters simultaneously 
>> updaters - a batch of updaters
>> paramKeepers - a batch of parameter keepers
>> optimizers - a batch of optimizers
*/
void XWorkerUpdate::UpdateParameterBatch(XList * updaters, 
                                         XList * paramKeepers, 
                                         XList * optimizers)
{
    CheckNTErrors(updaters != NULL, "No updaters!");
    CheckNTErrors(paramKeepers != NULL, "No paramter keepers!");
    CheckNTErrors(optimizers != NULL, "No optimizers!");
    CheckNTErrors(updaters->count != paramKeepers->count, 
                  "Updaters and parameter keepers are not of the same number!");
    CheckNTErrors(updaters->count != optimizers->count, 
                  "Updaters and optimizers are not of the same number!");

    for (int i = 0; i < updaters->count; i++) {
        XWorkerUpdate * updater = (XWorkerUpdate*)updaters->GetItem(i);
        XTensorKeeper * param = (XTensorKeeper*)paramKeepers->GetItem(i);
        XOptimizer * optimizer = (XOptimizer*)optimizers->GetItem(i);
        XQueue * queue = updater->GetJobQueue();

        /* we update the parameter in each individual queue */
        updater->AddJobUpdate(queue, param, optimizer);
        updater->AddJobEnqueueFinished(queue);
    }
}

/* wrapper of UpdateParameterBatch */
void XWorkerUpdate::UpdateBatch(XList * args)
{
    int paramCount = 0;

    CheckNTErrors(args != NULL && args->count == 4, "Illegal argument list!");

    XWorkerUpdate * primitiveUpdater = (XWorkerUpdate*)args->GetItem(paramCount++);
    XList * updaters = (XList*)args->GetItem(paramCount++);
    XList * paramKeepers = (XList*)args->GetItem(paramCount++);
    XList * optimizers = (XList*)args->GetItem(paramCount++);

    CheckNTErrors(primitiveUpdater != NULL, "No updater!");

    primitiveUpdater->UpdateParameterBatch(updaters, paramKeepers, optimizers);
}

/* 
add a new job of parameter update (for a batch)
>> jobQueue - the queue for running the primitive job
>> updaters - a batch of updaters
>> paramKeepers - a batch of parameter keepers
>> optimizers - a batch of optimizers
*/
bool XWorkerUpdate::AddJobUpdateBatch(XQueue * jobQueue, 
                                      XList * updaters,
                                      XList * paramKeepers, 
                                      XList * optimizers)
{
    CheckNTErrors(updaters != NULL, "No updaters!");
    CheckNTErrors(paramKeepers != NULL, "No paramter keepers!");
    CheckNTErrors(optimizers != NULL, "No optimizers!");

    XList args;
    args.Add(this);
    args.Add(updaters);
    args.Add(paramKeepers);
    args.Add(optimizers);

    XQueue& queueRun = jobQueue != NULL ? *jobQueue : queue;

    if (isInstantRun)
        XWorkerUpdate::UpdateBatch(&args);
    else
        queueRun.EnqueueJob((void*)(char*)XWorkerUpdate::UpdateBatch, &args);

    return true;
}

}
