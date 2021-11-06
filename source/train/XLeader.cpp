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
* A "leader" manages a number of "workers". The leader recieves jobs from
* the central server (can be remote), or acts as an independent server itself.
* For workers, the leader is the one who issues orders and organizes them.
* Note that the leader and workers must be on the same machine. In case of
* multi-machine training, one can deploy different leaders on different
* machines. BUT, at this time, we need an additional way of distributing
* data across machines.
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-02-25
*/

#include "XLeader.h"

/* the nts (NiuTrans.Tensor) namespace */
namespace nts {

/* constructor */
XLeader::XLeader()
{
    id = -1;
    paramMap = NULL;
    modelNum = 0;
}

/* de-constructor */
XLeader::~XLeader()
{
    DestroyParamMap();
}

/* intialize the leader */
void XLeader::Init()
{
    for (int i = 0; i < jworkers.count; i++)
        delete (XWorkerJob*)jworkers.GetItem(i);
    jworkers.Clear();

    for (int i = 0; i < cworkers.count; i++)
        delete (XWorkerCollect*)cworkers.GetItem(i);
    cworkers.Clear();

    for (int i = 0; i < uworkers.count; i++)
        delete (XWorkerUpdate*)uworkers.GetItem(i);
    uworkers.Clear();

    for (int i = 0; i < bworkers.count; i++)
        delete (XWorkerBroadcast*)bworkers.GetItem(i);
    bworkers.Clear();

    for(int i = 0; i < aworkers.count; i++)
        delete (XWorker*)aworkers.GetItem(i);
    aworkers.Clear();

    serverRecord.Clear();
}

/* set id */
void XLeader::SetID(int myID)
{
    id = myID;
}

/* get id */
int XLeader::GetID()
{
    return id;
}

/* 
Set the server model. It distributes the server-side parameters on different devices.
>> config - the configuration
>> model - the base model
>> memberModels - the models that run on different devices. We can place
                  the server-side parameters on different member models.
*/
void XLeader::SetServerModel(XConfig * config, XModel * model, XList * memberModels)
{
    serverModel.Clear();
    for (int i = 0; i < model->paramNum; i++) {
        XTensor * param = model->params[i].tensor;
        serverModel.AddParam(param);
    }

    /* TODO: we can place parameters on different devices */
}

/* 
set the server model. It distributes the server-side parameters on different devices.
>> config - the configuration
>> model - the base model*/
void XLeader::SetServerModel(XConfig * config, XModel * model)
{
    XList members;
    for (int i = 0; i < jworkers.count; i++) {
        XModel * member = ((XWorkerJob*)jworkers[i])->GetModel();
        members.Add(member);
    }

    SetServerModel(config, model, &members);
}

/* get server model */
XModel * XLeader::GetServerModel()
{
    return &serverModel;
}
    
/* initialize the models for running them */
void XLeader::InitForRun()
{
    serverModel.InitForRun();

    for (int i = 0; i < jworkers.count; i++) {
        XModel* model = ((XWorkerJob*)jworkers[i])->GetModel();
        model->InitForRun();
    }

    XList workers;
    workers.AddList(&jworkers);
    workers.AddList(&cworkers);
    workers.AddList(&uworkers);
    workers.AddList(&bworkers);
    workers.AddList(&aworkers);

    for (int i = 0; i < workers.count; i++) {
        XWorker* worker = (XWorker*)workers[i];
        CheckNTErrors(worker->IsEmpty(), "Something is wrong with the finishedQueue!");
    }
}

/* set grad = 0 */
void XLeader::ResetParamGrad()
{
    for (int i = 0; i < serverModel.paramNum; i++) {
        XTensor* param = serverModel.params[i].tensor;
        if (param->grad != NULL) {
            param->grad->SetZeroAll();
        }
    }

    for (int j = 0; j < jworkers.count; j++) {
        XWorkerJob * worker = (XWorkerJob*)jworkers[j];
        XModel * model = worker->GetModel();
        for (int i = 0; i < model->paramNum; i++) {
            XTensor* param = model->params[i].tensor;
            if (param->grad != NULL) {
                param->grad->SetZeroAll();
            }
        }
    }
}

/* 
prepare for running 
>> config - the configuration
>> model - the model that we run
*/
void XLeader::MakeAll(XConfig * config, XModel * model)
{
    SetServerModel(config, model);
    ResetParamGrad();
    MakeParamMap();
}

/* get loss */
float XLeader::GetLoss()
{
    return serverRecord.lossAll;
}
    
/* get sample number */
int XLeader::GetSampleNum()
{
    return serverRecord.sampleNum;
}

/* get prediction number */
int XLeader::GetPredictNum()
{
    return serverRecord.predictNum;
}

/* 
set the communication mode 
>> myMode - the mode
*/
void XLeader::SetMode(XLEADER_MODE myMode)
{
    mode = myMode;
}

/* set the flag of instant run */
void XLeader::SetInstantRun(bool flag)
{
    for (int i = 0; i < jworkers.count; i++) {
        XWorkerJob * worker = (XWorkerJob*)jworkers.GetItem(i);
        worker->SetInstantRun(flag);
    }

    for (int i = 0; i < cworkers.count; i++) {
        XWorkerJob * worker = (XWorkerJob*)cworkers.GetItem(i);
        worker->SetInstantRun(flag);
    }

    for (int i = 0; i < uworkers.count; i++) {
        XWorkerJob * worker = (XWorkerJob*)uworkers.GetItem(i);
        worker->SetInstantRun(flag);
    }

    for (int i = 0; i < bworkers.count; i++) {
        XWorkerJob * worker = (XWorkerJob*)bworkers.GetItem(i);
        worker->SetInstantRun(flag);
    }

    for (int i = 0; i < aworkers.count; i++) {
        XWorker * worker = (XWorker*)aworkers.GetItem(i);
        worker->SetInstantRun(flag);
    }
}

/* start the workers */
void XLeader::Start()
{
    serverModel.CheckParam();

    for (int i = 0; i < jworkers.count; i++) {
        XWorkerJob * worker = (XWorkerJob*)jworkers.GetItem(i);
        worker->GetModel()->CheckParam();
        worker->Start();
    }

    for (int i = 0; i < cworkers.count; i++) {
        XWorkerCollect * worker = (XWorkerCollect*)cworkers.GetItem(i);
        worker->Start();
    }

    for (int i = 0; i < uworkers.count; i++) {
        XWorkerUpdate * worker = (XWorkerUpdate*)uworkers.GetItem(i);
        worker->Start();
    }

    for (int i = 0; i < bworkers.count; i++) {
        XWorkerBroadcast * worker = (XWorkerBroadcast*)bworkers.GetItem(i);
        worker->Start();
    }

    for (int i = 0; i < aworkers.count; i++) {
        XWorker * worker = (XWorker*)aworkers.GetItem(i);
        worker->Start();
    }
}

/* 
add a number of job workers (given their device ids) 
>> model - the neural network
>> n - number of the models
>> devIDs - the array of device ids
*/
void XLeader::AddJobWorker(XModel * model, int n, const int * devIDs)
{
    /* we keep the input model */
    if (n >= 1) {
        XWorkerJob * worker = new XWorkerJob();
        worker->SetModel(model);
        jworkers.Add(worker);
    }

    /* we clone the input model */
    for (int i = 1; i < n; i++) {
        XWorkerJob * worker = new XWorkerJob();
        worker->SetModel(model->Clone(devIDs[i]));
        jworkers.Add(worker);
    }
}

/* 
add a data-collecting worker 
>> mode - the data-transfer mode of the worker
*/
void XLeader::AddCollectWorker(DATA_COLLECT_TYPE mode)
{
    XWorkerCollect * worker = new XWorkerCollect();
    worker->SetCollectMode(mode);
    cworkers.Add(worker);
}

/* add model-update workers */
void XLeader::AddUpdateWorker(int n)
{
    for (int i = 0; i < n; i++) {
        XWorkerUpdate* worker = new XWorkerUpdate();
        uworkers.Add(worker);
    }
}

/*  add a data-broadcasting worker */
void XLeader::AddBroadcastWorker()
{
    XWorkerBroadcast* worker = new XWorkerBroadcast();
    bworkers.Add(worker);
}

/* 
add parameter worker (or a pipeline) 
>> n - number of workers
*/
void XLeader::AddAuxiliaryWorker(int n)
{
    for (int i = 0; i < n; i++) {
        XWorker * worker = new XWorker();
        aworkers.Add(worker);
    }
}
    
/* destroy the parameter map (and gradient map) */
void XLeader::DestroyParamMap()
{
    for(int i = 0; i < serverModel.paramNum; i++){
        if(paramMap != NULL)
            delete[] paramMap[i];
    }
    delete[] paramMap;
    modelNum = 0;
}

/* generate the map of parameters */
void XLeader::MakeParamMap()
{
    int modelCount = CountModels();
    
    if(modelCount != modelNum){
        DestroyParamMap();
        paramMap = new XTensorKeeper*[serverModel.paramNum];
    }
    
    for(int i = 0; i < serverModel.paramNum; i++){
        if(modelCount != modelNum){
            paramMap[i] = new XTensorKeeper[modelCount];
        }
        
        for (int j = 0, c = 0; j < jworkers.count; j++) {
            XWorker * worker = (XWorker*)jworkers[j];
            if (worker->GetWorkerType() == XWORKER_TYPE_JOB) {
                XModel * model = ((XWorkerJob*)jworkers[j])->GetModel();
                paramMap[i][c].tensor = model->params[i].tensor;
                paramMap[i][c].grad = model->params[i].tensor->grad;
                paramMap[i][c].flag = PARAM_STATE_NOT_READY;
                paramMap[i][c].trainFlag = PARAM_STATE_NOT_READY;
                c++;
            }
            else {
                ShowNTErrors("TODO: support a new XWorker type!");
            }
        }
    }
    
    modelNum = modelCount;
}

/* count all the models */
int XLeader::CountModels()
{
    int modelCount = 0;
    for (int i = 0; i < jworkers.count; i++) {
        XWorker* worker = (XWorker*)jworkers[i];
        if (worker->GetWorkerType() == XWORKER_TYPE_JOB) {
            modelCount += worker->GetModelNum();
            CheckNTErrors(worker->GetModelNum() == 1, "Wrong model number!");
        }
        else {
            ShowNTErrors("TODO: support a new XWorker type!");
        }
    }
    
    CheckNTErrors(modelCount == jworkers.count, "We assume that a worker just has one model!");

    return modelCount;
}

} /* end of the nts (NiuTrans.Tensor) namespace */
