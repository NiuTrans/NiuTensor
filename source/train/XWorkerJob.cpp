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
* The worker of running the neural network.
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-02-24
*/

#include "XWorkerJob.h"
#include "../tensor/XList.h"
#include "../tensor/core/CHeader.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* constructor */
XWorkerJob::XWorkerJob() 
{
    type = XWORKER_TYPE_JOB;
    model = NULL;
    Clear();
}

/* de-constructor */
XWorkerJob::~XWorkerJob()
{
    for (int i = 0; i < inputs.count; i++)
        delete (XTensor*)inputs[i];

    for (int i = 0; i < outputs.count; i++)
        delete (XTensor*)outputs[i];

    for (int i = 0; i < golds.count; i++)
        delete (XTensor*)golds[i];

    for (int i = 0; i < losses.count; i++)
        delete (XTensor*)losses[i];
}

/* set the model */
void XWorkerJob::SetModel(XModel * myModel)
{
    model = myModel;
}

/* get the model */
XModel * XWorkerJob::GetModel()
{
    return model;
}

/* set the state of the worker */
void XWorkerJob::SetState(XWORKER_STATE myState)
{
    state = myState;
    record.state = myState;
}

/* clear the worker */
void XWorkerJob::Clear()
{
    for (int i = 0; i < inputs.count; i++)
        delete (XTensor*)inputs[i];
    inputs.Clear();
    inputs.Add(new XTensor());

    for (int i = 0; i < outputs.count; i++)
        delete (XTensor*)outputs[i];
    outputs.Clear();
    outputs.Add(new XTensor());

    for (int i = 0; i < golds.count; i++)
        delete (XTensor*)golds[i];
    golds.Clear();
    golds.Add(new XTensor());

    for (int i = 0; i < losses.count; i++)
        delete (XTensor*)losses[i];
    losses.Clear();
    losses.Add(new XTensor());

    record.Clear();

    SetState(XWORKER_UNSTARTED);
}

/* get the input list */
XList * XWorkerJob::GetInput()
{
    return &inputs;
}

/* get the output list */
XList * XWorkerJob::GetOutput()
{
    return &outputs;
}

/* get the gold standard */
XList * XWorkerJob::GetGold()
{
    return &golds;
}

/* get the loss */
XList * XWorkerJob::GetLoss()
{
    return &losses;
}

/* get the record of the run */
XNNRecord * XWorkerJob::GetRecord()
{
    return &record;
}

/* record some stuff */
void XWorkerJob::RecordMe()
{
    float lossAll = 0;
    int sampleNum = 0;

    for (int i = 0; i < losses.count; i++) {
        XTensor* loss = (XTensor*)losses[i];
        lossAll += ReduceSumAllValue(*loss);
        sampleNum += loss->GetSize();
    }

    record.lossAll = lossAll;
    record.sampleNum = sampleNum;

    int predictNum = 0;

    for (int i = 0; i < outputs.count; i++) {
        XTensor* output = (XTensor*)outputs[i];
        predictNum += output->GetSize();
    }

    record.predictNum = predictNum;
}

/* get the sum of losses over samples */
float XWorkerJob::GetLossAll()
{
    return record.lossAll;
}
    
/* get the number of samples */
int XWorkerJob::GetSampleNum()
{
    return record.sampleNum;
}

/* get the number of outputs (predictoins) */
int XWorkerJob::GetPredictNum()
{
    return record.predictNum;
}

/* 
add a new job of model refreshment 
>> myModel - the model
<< return - succeeded or not
*/
bool XWorkerJob::AddJobRefresh(XModel * myModel)
{
    //fprintf(stderr, "refresh 0\n");

    CheckNTErrors(myModel != NULL, "no parameter keeper!");

    XList args(1);
    args.Add(myModel);

    if(isInstantRun)
        XModel::Refresh(&args);
    else
        queue.EnqueueJob((void*)(char*)XModel::Refresh, &args);

    //fprintf(stderr, "refresh 1\n");

    return true;
}

/* 
add a new job of neural network forward and backward computation (with the input) 
>> myModel - the model
>> inputs - inputs of the neural network
>> outputs - outputs of the neural network
>> golds - gold standards
>> losses - losses of the outputs respect to the gold standards
<< return - succeeded or not
*/
bool XWorkerJob::AddJobNeuralNet(XModel * myModel,
                                 XList * inputs, XList * outputs, 
                                 XList * golds, XList * losses)
{
    CheckNTErrors(myModel != NULL, "no input neural network!");
    CheckNTErrors(inputs != NULL, "no inputs of the model!");
    CheckNTErrors(outputs != NULL, "no outputs of the model!");

    XList args;
    args.Add(myModel);
    args.Add(inputs);
    args.Add(outputs);
    args.Add(golds);
    args.Add(losses);

    if(isInstantRun)
        XModel::Run(&args);
    else
        queue.EnqueueJob((void*)(char*)XModel::Run, &args);

    SetState(XWORKER_STARTED);

    return true;
}

/* wrapper of RecordMe */
void XWorkerJob::RecordMeStatic(XList* args)
{
    //fprintf(stderr, "record static 0\n");

    CheckNTErrors(args != NULL && args->count > 0, "Illegal arguments!");

    XWorkerJob * worker = (XWorkerJob*)args->GetItem(0);
    XNNRecord * serverRecord = (XNNRecord *)args->GetItem(1);

    worker->RecordMe();

    /* push information to the server end */
    MUTEX_LOCK(serverRecord->mutex);
    serverRecord->Update(*worker->GetRecord());
    MUTEX_UNLOCK(serverRecord->mutex);

    worker->SetState(XWORKER_FINISHED);

    //fprintf(stderr, "record static 1\n");
}

/* 
add a new job of recording the running of the nerual network 
>> serverRecord - the model record on the server side
*/
bool XWorkerJob::AddJobRecord(XNNRecord * serverRecord)
{
    XList args;
    args.Add(this);
    args.Add(serverRecord);

    if (isInstantRun)
        XWorkerJob::RecordMeStatic(&args);
    else
        queue.EnqueueJob((void*)(char*)XWorkerJob::RecordMeStatic, &args);

    return true;
}

}  /* end of the nts (NiuTrans.Tensor) namespace */

