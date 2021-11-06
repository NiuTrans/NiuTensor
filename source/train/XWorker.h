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
* The base class of worker. It maintains a job queue and offers utilities
* of controlling the working pipeline.
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-02-24
* People started to go back to the normal life after the Spring Festival.
* Traffic jams again.
*/

#ifndef __XWORKER_H__
#define __XWORKER_H__

#include "XModel.h"
#include "../tensor/XQueue.h"
#include "../tensor/XUtility.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
state of a worker
1) unstarted
2) started
3) finished
*/
enum XWORKER_STATE { XWORKER_UNSTARTED, XWORKER_STARTED, XWORKER_FINISHED };

/*
worker type
*/
enum XWORKER_TYPE { XWORKER_TYPE_UNKNOWN, XWORKER_TYPE_JOB, XWORKER_TYPE_COLLECT, XWORKER_TYPE_UPDATE, XWORKER_TYPE_BROADCAST };

/* the worker class */
class XWorker
{
protected:
    /* type of the worker */
    XWORKER_TYPE type;

    /* id of the device where we run the worker (we suppose that
    the worker is insite. */
    int devID;

    /* id of the worker */
    int id;

    /* the queue of jobs */
    XQueue queue;

    /* state of the worker */
    XWORKER_STATE state;

    /* fire the flag of instant run */
    bool isInstantRun;

    /* the queue of counting finished jobs */
    XQueue finishedQueue;
    
public:
    /* constructor */
    XWorker();

    /* de-constructor */
    ~XWorker();

    /* get worker type */
    XWORKER_TYPE GetWorkerType();

    /* set device id */
    void SetDeviceID(int myDevID);

    /* get device id */
    int GetDeviceID();

    /* set worker id */
    void SetID(int myID);

    /* get worker id */
    int GetID();

    /* get job queue */
    XQueue * GetJobQueue();

    /* set the flag of instant run */
    void SetInstantRun(bool flag = true);

    /* enqueue a new job */
    void AddJob(void * job, XList * jobArgs);

    /* start the work */
    void Start();

    /* stop the work */
    void Stop();

    /* get the number of the remaining jobs */
    int GetJobNum();

    /* get the number of the models for this worker */
    int GetModelNum();

    /* whether the job queue is empty? */
    bool IsEmpty();

    /* enqueue a counting job of a finished job */
    void EnqueueFinishedJob();

    /* dequeue a counting job of a finished job */
    void DequeueFinishedJob();

    /* wrapper of EnqueueFinished() */
    static
    void EnqueueFinished(XList* args);

    /* wrapper of DequeueFinished() */
    static
    void DequeueFinished(XList* args);

    /* add a job of enqueuing a counting a finished job */
    void AddJobEnqueueFinished(XQueue * jobQueue = NULL);

    /* add a job of dequeuing a counting a finished job */
    void AddJobDequeueFinished(XQueue* jobQueue = NULL);
    
    /* get number of unflaged finished job */
    int GetFinishedNumInQueue();
};

}

#endif
