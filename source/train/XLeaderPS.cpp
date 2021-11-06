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
*/

#include "XLeaderPS.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* constructor */
XLeaderPS::XLeaderPS()
{
}

/* deconstructor */
XLeaderPS::~XLeaderPS()
{
}

/* 
create workers and other stuff
>> config - configuration
>> model - the model that we run
>> devIDs - device ids of the workers (the first id is for server)
>> jobWorkerNum - number of job workers
*/
void XLeaderPS::MakeAll(XConfig * config, XModel * model, const int * devIDs, const int jobWorkerNum)
{
    Init();
    AddJobWorker(model, jobWorkerNum, devIDs);
    AddCollectWorker();
    AddUpdateWorker();
    AddBroadcastWorker();
    AddAuxiliaryWorker(model->paramNum);

    XLeader::MakeAll(config, model);
}

/*
wait for finished states (i.e., all workers finish their jobs)
>> activeJobWorkers - indicates whether each job worker is active
>> isToUpdate - indicates whether the model is updated
*/
void XLeaderPS::WaitForFinishing(const int* activeJobWorkers, const int isToUpdate)
{
    int activeCount = 0;
    for (int i = 0; i < jworkers.count; i++) {
        if (activeJobWorkers[i] > 0) {
            XWorker* worker = (XWorker*)jworkers[i];
            worker->DequeueFinishedJob();
            activeCount++;
            CheckNTErrors(worker->GetFinishedNumInQueue() == 0, "Incorrect job number!");
        }
    }

    if (activeCount > 0 && isToUpdate) {
        for (int i = 0; i < cworkers.count; i++) {
            XWorker* worker = (XWorker*)cworkers[i];
            for (int j = 0; j < serverModel.paramNum * activeCount; j++)
                worker->DequeueFinishedJob();
            CheckNTErrors(worker->GetFinishedNumInQueue() == 0, "Incorrect job number!");
        }

        for (int i = 0; i < uworkers.count; i++) {
            XWorker* worker = (XWorker*)uworkers[i];
            for (int j = 0; j < serverModel.paramNum; j++)
                worker->DequeueFinishedJob();
            CheckNTErrors(worker->GetFinishedNumInQueue() == 0, "Incorrect job number!");
        }

        for (int i = 0; i < bworkers.count; i++) {
            XWorker* worker = (XWorker*)bworkers[i];
            for (int j = 0; j < serverModel.paramNum; j++)
                worker->DequeueFinishedJob();
            CheckNTErrors(worker->GetFinishedNumInQueue() == 0, "Incorrect job number!");
        }
    }
}

/*
run the model (for one time). Basically this is a map-reduce process.
>> config - the configuration
>> dataDistributor - data distributor
>> optimizer - the optimization method
<< return - if we can fetch the new data
*/
bool XLeaderPS::Run(XConfig* config, DataDistributeBase* dataDistributor, XOptimizer* optimizer)
{
    CheckNTErrors(jworkers.count > 0, "No jworkers!");
    CheckNTErrors(cworkers.count > 0, "No cworkers!");
    CheckNTErrors(uworkers.count > 0, "No uworkers!");
    CheckNTErrors(bworkers.count > 0, "No bworkers!");
    CheckNTErrors(aworkers.count > 0, "No pworkers!");

    bool isToUpdate = (optimizer != NULL);
    int activeJobCount = 0;
    int* active = new int[jworkers.count];

    InitForRun();

    /* run models on job workers */
    activeJobCount = RunModel(config, dataDistributor, active);

    /* update the model on the server side */
    if (activeJobCount > 0 && isToUpdate)
        RunUpdate(config, optimizer, active);

    WaitForFinishing(active, isToUpdate);

    for (int i = 0; i < jworkers.count; i++) {
        XWorkerJob* worker = (XWorkerJob*)jworkers[i];
        worker->Clear();
    }

    delete[] active;

    return activeJobCount > 0;
}

/*
run the model
>> config - the configuration
>> dataDistributor - to load batches of samples
>> active - flag for each job worker (1 = active, 0 = not active)
<< return - number of active job workers
*/
int XLeaderPS::RunModel(XConfig* config, DataDistributeBase* dataDistributor, int* active)
{
    int activeJobCount = 0;

    for (int i = 0; i < jworkers.count; i++)
        active[i] = 0;

    /* Feed the input to each worker and geneate the output.
    For each worker, we define a job queue and enqueue jobs
    into it.
    */
    for (int i = 0; i < jworkers.count; i++) {
        XWorkerJob* worker = (XWorkerJob*)jworkers[i];
        XModel* jmodel = worker->GetModel();

        /* get a batch of samples */
        bool fetched = dataDistributor->GetBatchSimple(worker->GetInput(), worker->GetGold());

        if (fetched) {
            /* job in queue 1: refresh the model */
            worker->AddJobRefresh(jmodel);

            /* job in queue 1: run the model */
            worker->AddJobNeuralNet(jmodel,
                                    worker->GetInput(), worker->GetOutput(),
                                    worker->GetGold(), worker->GetLoss());

            /* job in queue 1: make a record of the run */
            worker->AddJobRecord(&serverRecord);

            /* job in queue 1: mark finished */
            worker->AddJobEnqueueFinished();

            active[i] = 1;
            activeJobCount++;
        }
    }

    return activeJobCount;
}

/*
update the model in a standard server-worker manner
>> config - the configuration
>> optimizer - the optimizer
>> active - flag for each job worker (1 = active, 0 = not active)
*/
void XLeaderPS::RunUpdate(XConfig* config, XOptimizer* optimizer, const int* active)
{
    /* workers */
    XWorkerCollect* collecter = (XWorkerCollect*)cworkers.GetItem(0);
    XWorkerUpdate* updater = (XWorkerUpdate*)uworkers.GetItem(0);
    XWorkerBroadcast* broadcaster = (XWorkerBroadcast*)bworkers.GetItem(0);

    /* parameter map */
    MakeParamMap();

    /* all member models */
    XList membersAll(jworkers.count);

    /* job queues */
    XList jobQueues;

    for (int i = 0; i < jworkers.count; i++) {
        XWorkerJob* worker = (XWorkerJob*)jworkers[i];
        membersAll.Add(worker->GetModel());
    }

    for (int i = 0; i < aworkers.count; i++) {
        XWorker* worker = (XWorker*)aworkers[i];
        jobQueues.Add(worker->GetJobQueue());
    }

    CheckNTErrors(modelNum == jworkers.count, "We assume that a worker has one model only!");
    CheckNTErrors(jobQueues.count == serverModel.paramNum, "Incompatiable model!");

    /* jobs in queue 2 (say jobQueue): collect the (gradient) data.
    This is a reduce process. Then we add a job to to update the model. followed
    by a job to broadcast the lastest parameters to workers. NOTE that we
    would update a worker to the latest model parameters, even if it is not
    involved in this run. */

    int finished = 0;

    for (int j = 0; j < serverModel.paramNum; j++)
        serverModel.params[j].flag = PARAM_STATE_NOT_READY;

    /* counts how many member models are collected for each parameter */
    int* finishedCount = new int[serverModel.paramNum];
    memset(finishedCount, 0, sizeof(int) * serverModel.paramNum);

    /* flag active models */
    int modelCount = 0;
    int activeModelCount = 0;
    int* modelFlag = new int[modelNum];
    for (int i = 0; i < jworkers.count; i++) {
        XWorkerJob* worker = (XWorkerJob*)jworkers[i];
        for (int j = 0; j < worker->GetModelNum(); j++) {
            modelFlag[modelCount++] = active[i];
            if (active[i] != 0)
                activeModelCount++;
        }
    }

    XList* paramList = new XList[serverModel.paramNum];

    CheckNTErrors(modelCount == modelNum, "Wrong model number!");

    /* This is a simple implementation of the do-and-wait process */
    while (1) {
        for (int j = 0; j < serverModel.paramNum; j++) {

            XTensorKeeper& paramServer = serverModel.params[j];

            /* isGradFinished is true only if the model finishes the computation
               (in another thread) */
            if (paramServer.flag != PARAM_STATE_NOT_READY || !paramServer.tensor->isGradFinished)
                continue;

            /* set the gradient tensor */
            if (paramServer.grad != paramServer.tensor->grad)
                paramServer.grad = paramServer.tensor->grad;

            /* check if all the models (or part of them) are ready */
            for (int i = 0; i < jworkers.count; i++) {
                
                /* skip the inactive model */
                if (modelFlag[i] == 0)
                    continue;

                XTensorKeeper& paramWorker = paramMap[j][i];

                /* isGradFinished is true only if the model finishes the computation
                  (in another thread) */
                if (paramWorker.flag == PARAM_STATE_NOT_READY && paramWorker.tensor->isGradFinished) {

                    /* get the gradient */
                    paramWorker.grad = paramWorker.tensor->grad;

                    /* the job queue of updating parameter j */
                    XQueue* jobQueue = (XQueue*)jobQueues.GetItem(j);

                    /* data transmit */
                    collecter->AddJobCollectGradP2P(jobQueue, &paramWorker, &paramServer);
                    collecter->AddJobEnqueueFinished(jobQueue);

                    /* We keep the worker parameter in a list. It would be used when we broadcast
                       the updated paramter to the workers, that is, this is a list of worker
                       parameters. */
                    paramList[j].Add(&paramWorker);

                    /* reset the flag */
                    paramWorker.flag = PARAM_STATE_COLLECTED;
                    finished++;
                    finishedCount[j]++;

                    /* we call model update (in another thread) and then
                       broadcast the new parameters to member models
                      (in another thread) */
                    if (finishedCount[j] == activeModelCount) {
                        paramServer.flag = PARAM_STATE_COLLECTED;
                        if (updater != NULL) {

                            /* update the parameters */
                            updater->AddJobUpdate(jobQueue, &paramServer, optimizer);
                            updater->AddJobEnqueueFinished(jobQueue);

                            /* broadcast the new parameter to other models */
                            broadcaster->AddJobBroadcast(jobQueue, &paramServer, &paramList[j]);
                            broadcaster->AddJobEnqueueFinished(jobQueue);
                        }
                    }
                    else if (finishedCount[j] > activeModelCount) {
                        ShowNTErrors("Something is wrong with finishedCount!");
                    }
                }
            }
        }

        /* finishes if all data tensors are processed */
        if (finished == serverModel.paramNum * activeModelCount)
            break;

        XSleep(SLEEP_TIME_IN_WAITING_JOB_WORKERS);
    }

    delete[] finishedCount;
    delete[] modelFlag;
    delete[] paramList;
}

}
