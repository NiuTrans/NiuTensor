/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2017, Natural Language Processing Lab, Northeastern University. 
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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2017-04-05
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include "XQueue.h"
#include "XDevice.h"
#include "XList.h"
#include "XUtility.h"

/* the nts (NiuTrans.Tensor) namespace */
namespace nts{

/**************************************
job item used in queues
*/

/* constructor */
JobQueueNode::JobQueueNode()
{
    job  = NULL;
    args = new XList(1);
}

/* de-constructor */
JobQueueNode::~JobQueueNode()
{
    delete args;
}

/**************************************
This class provides standard utilities of Queue.
*/

/* constuctor */
XQueue::XQueue(int mySize)
{
    queue = new void*[mySize];

    memset(queue, 0, sizeof(void*) * mySize);

    size = mySize;
    itemCount = 0;
    head = 0;
    tail = 0;
    isJobQueue = false;
    jobDequeuerArgs = new XList(1);
    jobDequeuerBreak = false;
    runningJobCount = 0;
    
    MUTEX_INIT(enqueueMutex);
    MUTEX_INIT(dequeueMutex);
    COND_INIT(queueCond);
    MUTEX_INIT(jobQueueMutex);
}

/* deconstructor */
XQueue::~XQueue()
{
    delete[] queue;
    delete jobDequeuerArgs;

    //if(isJobQueue)
    //    StopJobConsumer();

    MUTEX_DELE(enqueueMutex);
    MUTEX_DELE(dequeueMutex);
    COND_DELE(queueCond);
    MUTEX_DELE(jobQueueMutex);
}

/* 
put an item in the tail of the queue 
>> item - the item we intend to add into the queue
*/
void XQueue::Enqueue(void * item)
{

    MUTEX_LOCK(enqueueMutex);
    MUTEX_LOCK(dequeueMutex);

    CheckNTErrors((itemCount < size), "Put too many items into the queue!");

    queue[tail] = item;
    tail = (tail + 1) % size;
    itemCount++;
    
    COND_SIGNAL(queueCond);

    MUTEX_UNLOCK(dequeueMutex);
    MUTEX_UNLOCK(enqueueMutex);
}

/* 
fetch an item from head of the queue 
<< return - the head item of the queue
*/
void * XQueue::Dequeue()
{
    MUTEX_LOCK(dequeueMutex);

    while(itemCount == 0)
    {
#ifdef  WIN32
        MUTEX_UNLOCK(dequeueMutex);
#endif
        COND_WAIT(queueCond, dequeueMutex);
#ifdef  WIN32
        MUTEX_LOCK(dequeueMutex);
#endif
    }

    void * r = queue[head];
    head = (head + 1) % size;
    itemCount--;

    MUTEX_UNLOCK(dequeueMutex);

    return r;
}

/* return if the queue is empty */
bool XQueue::IsEmpty()
{
    return itemCount == 0;
}

/* wait until the queue is empty */
void XQueue::WaitForEmptyJobQueue()
{
    while(runningJobCount > 0){
        XSleep(10);
    }
}

int devids[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
int cpuid = -1;

/* 
run job consumer (in another thread) 
>> jobDevID - id of the device for running the jobs
*/
void XQueue::RunJobConsumer(int jobDevID)
{
    CheckNTErrors((jobDevID < 16), "device id is out of scope!");

    isJobQueue = true;
    jobDequeuerArgs->Clear();

    /* warning: this may cause unknown errors */
    jobDequeuerArgs->Add(this);
    jobDequeuerArgs->Add(jobDevID >= 0 ? (devids + jobDevID) : &cpuid);

    jobDequeuer.SetFunc((TFunction)DequeueJobs, jobDequeuerArgs);

    //jobDequeuer.Start();
    //jobDequeuer.LetItGo();
    jobDequeuer.StartNow();
}

/* stop the job consumer */
void XQueue::StopJobConsumer()
{
    jobDequeuerBreak = true;
    XSleep(10);

    EnqueueJob(NULL, NULL);

    jobDequeuer.End();
    isJobQueue = false;
}

/* add a job item to process */
void XQueue::EnqueueJob(void * job, XList * jobArgs)
{
    MUTEX_LOCK(jobQueueMutex);
    runningJobCount++;
    MUTEX_UNLOCK(jobQueueMutex);

    JobQueueNode * node = new JobQueueNode();
    node->job = job;
    if(jobArgs != NULL)
        node->args->AddList(jobArgs);
    Enqueue(node);
}

/* job item consumer */
void XQueue::DequeueJobs(XList * args)
{
    CheckNTErrors((args->count == 2), "Illegal arguments!");

    XQueue * q = (XQueue*)args->GetItem(0);
    int devID = *(int*)args->GetItem(1);

    int devIDBackup = -1;
    if(devID >= 0)
        XDevice::SetDevice(devID, devIDBackup);

    while(1){
        JobQueueNode * node = (JobQueueNode*)q->Dequeue();

        if(q->GetJobBreak())
            break;

        CheckNTErrors((node != NULL), "Illegal job!");

        /* process a job */
        ((TFunction)node->job)(node->args);

        delete node;

        MUTEX_LOCK(q->jobQueueMutex);
        q->runningJobCount--;
        MUTEX_UNLOCK(q->jobQueueMutex);

    }

    if(devID >= 0)
        XDevice::SetDevice(devIDBackup);
}

/* get the break flag */
bool XQueue::GetJobBreak()
{
    return jobDequeuerBreak;
}

/* get the number of jobs */
int XQueue::GetJobNum()
{
    MUTEX_LOCK(jobQueueMutex);
    int c = runningJobCount;
    MUTEX_UNLOCK(jobQueueMutex);

    return c;
}
    
/*
get the number of items in the queue. Note that
this function is not the same as GetJobNum() because
"items" are the real elements we put into the queue.
"jobs" only make sense when the queue is running as a
job queue.
*/
int XQueue::GetItemNum()
{
    MUTEX_LOCK(enqueueMutex);
    MUTEX_LOCK(dequeueMutex);
    
    int c = itemCount;
    
    MUTEX_UNLOCK(dequeueMutex);
    MUTEX_UNLOCK(enqueueMutex);
    
    return c;
}

} /* end of the nts (NiuTrans.Tensor) namespace */
