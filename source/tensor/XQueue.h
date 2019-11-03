/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2017, Natural Language Processing Lab, Northestern University. 
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
 * 
 * This is an implementation of queue. Actually we intend to use it to maintain
 * a priority job list
 *
 *
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2017-04-05
 * I came back from the holiday - while Tongran and Dingdang are still in Beijing
 * (working and playing??)
 *
 * Parts of the code is copied from Duquan's work. Thanks :)
 */

#ifndef __XQUEUE_H__
#define __XQUEUE_H__

#include "XGlobal.h"
#include "XThread.h"
#include "XStream.h"
#include "XDevice.h"
#include "XList.h"

/* the nts (NiuTrans.Tensor) namespace */
namespace nts{

#define MAX_QUEUE_SIZE 1024 * 8

/*
job item used in queues
*/
class JobQueueNode
{
public:
    /* the job function */
    void * job;

    /* arguments of the job */
    TensorList * args;

public:
    /* constructor */
    JobQueueNode();

    /* de-constructor */
    ~JobQueueNode();
};

/*
This class provides standard utilities of Queue.
*/
class XQueue
{
private:
    /* mutex for the enqueue process */
    MUTEX_HANDLE enqueueMutex;

    /* mutex for the dequeue process */
    MUTEX_HANDLE dequeueMutex;

    /* conditional mutex for the dequeue process */
    COND_HANDLE  queueCond;

    /* mutex for the job queue */
    MUTEX_HANDLE jobQueueMutex;

    /* the array for the queue */
    void ** queue;

    /* max size of the queue */
    int size;

    /* number of item in queue */
    int itemCount;

    /* head of the queue */
    int head;

    /* tail of the queue */
    int tail;

    /* indicates whether we are using a job queue */
    bool isJobQueue;

    /* consume the job items in the queue */
    XThread jobDequeuer;

    /* argument list of jobDequeuer */
    TensorList * jobDequeuerArgs;

    /* indicates whether jobDequeuer stops */
    bool jobDequeuerBreak;

    /* running job count */
    int runningJobCount;

    /* job streams (we think that three streams is enough :)) */
    XStream * jobStream;
    XStream * jobStream1;
    XStream * jobStream2;

public:
    /* constuctor */
    XQueue(int mySize = MAX_QUEUE_SIZE);

    /* deconstructor */
    ~XQueue();

    /* put an item in the tail of the queue */
    void Enqueue(void * item);

    /* fetch an item from head of the queue */
    void * Dequeue();

    /* return if the queue is empty */
    bool IsEmpty();

    /* wait until the queue is empty */
    void WaitForEmptyJobQueue();

    /* run the job consumer */
    void RunJobConsumer(int jobDevID = 0);

    /* stop the job consumer */
    void StopJobConsumer();

    /* add a job item to process */
    void EnqueueJob(void * job, TensorList * jobArgs);

    /* job item consumer */
    static
    void DequeueJobs(TensorList * args);

    /* get the break flag */
    bool GetJobBreak();

    /* get job stream */
    XStream * GetJobStream(int n = 0);

    /* make job streams */
    void MakeJobStreams(int devID = INVALID_DEVICE_ID, int devID1 = INVALID_DEVICE_ID, int devID2 = INVALID_DEVICE_ID);
};

} /* end of the nts (NiuTrans.Tensor) namespace */

#endif