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
 * a naive implementation of thread pool (actually it is a pool)
 *
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2016-03-08
 *
 */

#ifndef __XTHREAD_H__
#define __XTHREAD_H__

#include "XList.h"

#ifndef _WIN32
#define USE_PTHREAD // for linux
#endif

//////////////////////////////////////////////////
// neccessary libs
#ifdef USE_PTHREAD
#include <pthread.h> // use "-lpthread" when compiling on linux systems
#else
#ifdef _WIN32
#include <windows.h>
#include <process.h>
#endif
#endif

/* the nts (NiuTrans.Tensor) namespace */
namespace nts{

#if(defined(_WIN32) && !defined (__CYGWIN__))
#define CRFPP_USE_THREAD 1
#define BEGINTHREAD(src, stack, func, arg, flag, id) \
                   (HANDLE)_beginthreadex((void *)(src), (unsigned)(stack), \
                   (unsigned(_stdcall *)(void *))(func), (void *)(arg), \
                   (unsigned)(flag), (unsigned *)(id))
#endif

typedef void (*TFunction) (volatile XList*);

/*
This is a class that wraps the standard implementation of threading
(for both windows and linux OS)
*/
class XThread
{

public:
    /* thread id */
    THREAD_HANDLE hnd;

    /* to information outside caller */
    MUTEX_HANDLE gMutex;

    /* working state */
    int working;

    /* a lock to protect the working state */
    MUTEX_HANDLE workingMutex;

    /* to inform the job when it is ready */
    COND_HANDLE jobCond;

    /* indicate whether the thread is running */
    bool isRunning;

#ifdef USE_PTHREAD

    /* a mutex lock */
    MUTEX_HANDLE mutex;

    /* condition lock */
    COND_HANDLE cond;

    /* scheduling for threads */
    sched_param schedParam;
#else
#endif

public:
    /* function to run */
    TFunction function;

    /* arguments (for the function to run) */
    XList argv;

    /* a flag to break */
    volatile
    bool toBreak;

    /* number of jobs that are waiting */
    volatile
    int jobCount;

public:
    /* constructor */
    XThread();

    /* deconstructor */
    ~XThread();

public:
    /* a wrapper for the start-routine parameter in pthread_create */
    static void * Wrapper(void * ptr);

    /* initialize the thread with the function and its parameters */
    void SetFunc(TFunction myFunc, XList * myArgv);

    /* 
    Core of the thread. It is very very native impelementation.
    We loop and wait for a singnal to activate the job processing.
    After that, we wait again if there is no new job.
    */
    void Run();

    /* create and run the thread */
    bool Start();

    /* end the thread */
    void End();

    /* wait for thread termination */
    void Join();

    /* let the thread process a job */
    void LetItGo();
    
    /* create the thread and run it immediately (a combination of
       Start() and LetItGo() */
    bool StartNow();

    /* waith for a singal */
    static
    void Wait(COND_HANDLE * c, MUTEX_HANDLE * m);
};

/*
a counter with mutex
*/
class XCounter
{
private:
    /* count */
    int count;

    /* lock */
    MUTEX_HANDLE mutex;

public:
    /* constructor */
    XCounter();

    /* deconstructor */
    ~XCounter();

    /* add the counter by 1 */
    void Add();

    /* get the counting number */
    int Get();
};

} /* end of the nts (NiuTrans.Tensor) namespace */

#endif
