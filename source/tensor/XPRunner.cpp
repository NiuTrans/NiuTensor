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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2016-03-09
 *
 */

#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include "XPRunner.h"
#include "XGlobal.h"

/* the nts (NiuTrans.Tensor) namespace */
namespace nts{

/*
The XPRunner maintains a the parallel processing resources, e.g., a pool
of threads. It can provide the parallel computation interface for someone
that needs to do something parallel, e.g., speed-up matrix operation by
multi-threading.
*/

XPRunner * globalPRunner = NULL;


/****************************
general methods
*/

 /* constructor */
XPRunner::XPRunner()
{
    method = PRUNNER_SINGLE;

    /* multi-threading */
    threads = NULL;
    threadNum = 0;
    minimumOPNum = INT_MAX;
    MUTEX_INIT(mutex);
    isMultiThreaded = true;
    availableThreadNum = 0;
    runningThreadNum = 0;
    runningThreads = new int[MAX_THREAD_NUM];
    memset(runningThreads, 0 ,sizeof(int) * MAX_THREAD_NUM);
    runningStates = new int[MAX_THREAD_NUM];
    memset(runningStates, 0 ,sizeof(int) * MAX_THREAD_NUM);
    availableThreads = new int[MAX_THREAD_NUM];
    memset(availableThreads, 0 ,sizeof(int) * MAX_THREAD_NUM);
}

/* deconstructor */
XPRunner::~XPRunner()
{
    KillThreads();
    MUTEX_DELE(mutex);
    delete[] runningThreads;
    delete[] runningStates;
    delete[] availableThreads;
}

/* 
initialization 
>> myThreadNum - number of required threads
*/
void XPRunner::Init(int myThreadNum)
{
    CreateThreads(myThreadNum);

    if(myThreadNum > 0)
        method = PRUNNER_MULTIPLE;
}


/****************************
methods for multi-threading
*/

/* 
initialization 
>> tNum - number of required threads
*/
void XPRunner::CreateThreads(int tNum)
{
    if(tNum > MAX_THREAD_NUM){
        XPRINT2(0, stderr, "[XPRunner::CreateThreads] Error! Too many threads[%d>%d]!\n", tNum, MAX_THREAD_NUM);
        exit(1);
    }

    threads = new XThread[tNum];
    for(int i = 0; i < tNum; i++){
        if(!threads[i].Start()){
            XPRINT1(0, stderr, "[XPRunner::CreateThreads] Error! cannot create thread %d\n", i);
            exit(1);
        }
    }

#ifdef _WIN32
    Sleep(300);
#else
    usleep(300 * 1000);
#endif

    threadNum = tNum;

    minimumOPNum = MIN_OPERATION_NUM;


}

/* kill all threads */
void XPRunner::KillThreads()
{
    for(int i = 0; i < threadNum; i++){
        //threads[i].End();
    }
#ifdef    _WIN32
    //Sleep(300);
#else
    //sleep(0.3);
#endif
    delete[] threads;
    threads = NULL;
}


/* 
run a set of jobs in parallel 
>> jobFunctions - the function for each job
>> jobArgs - the list of arguments for each job
>> sleepTime - time to sleep (in ms) for each round
*/
void XPRunner::Run(TensorList * jobFunctions, TensorList * jobArgs, float sleepTime)
{
    if(threadNum <= 0){
        XPRINT(1, stderr, "Error! No threads were created!\n");
        exit(1);
    }

    runningThreadNum = 0;
    availableThreadNum = 0;

    memset(runningStates, 0, sizeof(int) * MAX_THREAD_NUM);

    int c = jobFunctions->count;
    int unfinished = c;

    MUTEX_LOCK(mutex);

    while(unfinished > 0){

        /* get the list of threads that are ready to process the job */
        for(int i = 0; i < threadNum; i++){
            if(runningStates[i] == 2 && threads[i].jobCount == 0){
                /* a job has been finished*/
                unfinished--;
                availableThreads[availableThreadNum++] = i;
                runningStates[i] = 1;
#ifdef _WIN32
                MUTEX_LOCK(threads[i].workingMutex);
                COND_RESET(threads[i].jobCond);
                MUTEX_UNLOCK(threads[i].workingMutex);
#endif
            }
            else if(runningStates[i] == 0 && threads[i].jobCount == 0){
                availableThreads[availableThreadNum++] = i;
                runningStates[i] = 1;
#ifdef _WIN32
                MUTEX_LOCK(threads[i].workingMutex);
                COND_RESET(threads[i].jobCond);
                MUTEX_UNLOCK(threads[i].workingMutex);
#endif
            }
        }

        /* assign the jobs */
        for(int i = availableThreadNum - 1; i >= 0 && c > 0; i--){
            /* the function to run*/
            TFunction function = (TFunction)jobFunctions->GetItem(jobArgs->count - c);

            /* the arguments that are passed to the function */
            volatile TensorList * args = (TensorList*)jobArgs->GetItem(jobArgs->count - c);

            /* thread */
            XThread * thread  = threads + availableThreads[i];

            thread->argv = args;
            thread->function = function;

            MUTEX_LOCK(thread->workingMutex);
            thread->working = 1;
            MUTEX_UNLOCK(thread->workingMutex);

#ifdef USE_PTHREAD
            MUTEX_LOCK(thread->mutex);
            thread->jobCount++;
            MUTEX_UNLOCK(thread->mutex);
            //COND_BROADCAST(thread->cond);
            COND_SIGNAL(thread->cond);
            //MUTEX_UNLOCK(thread->mutex);
#else
#ifdef _WIN32
            /* reset various locks */
            MUTEX_LOCK(thread->workingMutex);
            thread->jobCount++;
            COND_RESET(thread->jobCond);
            //COND_RESET(thread->gCond);
            MUTEX_UNLOCK(thread->workingMutex);

            /* inform the job */
            //ResumeThread(threads[i].hnd);
            COND_SIGNAL(thread->jobCond);
#endif
#endif

            /* a job is under processing */
            c--;
            availableThreadNum--;
            runningStates[availableThreads[i]] = 2;
        }

        if(sleepTime > 0){
#ifdef _WIN32
            Sleep((DWORD)sleepTime);
#else
            sleep(sleepTime/1000);
#endif
        }
    }

    MUTEX_UNLOCK(mutex);
}

/* 
get the number of parallel jobs to run 
size - number of operations we need
*/
int XPRunner::GetJobNum(int size)
{
    int jobNum = int((float)size/minimumOPNum);

    return MIN(jobNum, threadNum);
}

} /* end of the nts (NiuTrans.Tensor) namespace */
