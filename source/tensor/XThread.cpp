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
 * a naive implementation of thread pool (actually it is a pool)
 *
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2016-03-08
 *
 */

#include "XGlobal.h"
#include "XThread.h"

/* the nts (NiuTrans.Tensor) namespace */
namespace nts{

/* constructor */
XThread::XThread()
{
#ifdef USE_PTHREAD
    MUTEX_INIT(mutex);
    COND_INIT(cond);
#endif
    MUTEX_INIT(gMutex);
    function = NULL;
    argv = NULL;
    toBreak = false;
    jobCount = 0;
    working = 0;
    MUTEX_INIT(workingMutex);
    COND_INIT(jobCond);
    isRunning = false;
    hnd = 0;
}

/* de-constructor */
XThread::~XThread()
{
    End();
#ifdef USE_PTHREAD
    MUTEX_DELE(mutex);
    COND_DELE(cond);
#endif
    MUTEX_DELE(gMutex);
    MUTEX_DELE(workingMutex);
    COND_DELE(jobCond);
};

/* a wrapper for the start-routine parameter in pthread_create */
void * XThread::Wrapper(void * ptr) 
{
    XThread * p = (XThread *)ptr;
    p->Run();
    return 0;
}


/* 
Tunning for this thread. It is very very native implementation.
We loop and wait for a signal to activate the job processing.
After that, we wait again if there is no new job.
*/
void XThread::Run()
{
#ifdef _WIN32
    //COND_RESET(gCond);
#endif    

    while(1){
#ifdef USE_PTHREAD
        /* waiting for the job */
        MUTEX_LOCK(mutex);
        while(jobCount == 0){
            COND_WAIT(cond, mutex); // it unlocks the mutex first
                                    // and then wait
        }
#else
#ifdef _WIN32
        //SuspendThread(hnd);
        COND_WAIT(jobCond, gMutex);
#endif
#endif

        if(toBreak){
#ifdef USE_PTHREAD
            MUTEX_UNLOCK(mutex);
#endif
            break;
        }

        /* do what you want to do*/
        function(argv);

#ifdef USE_PTHREAD
        jobCount--;
        MUTEX_UNLOCK(mutex);
#else
#ifdef _WIN32
        MUTEX_LOCK(workingMutex);
        working = 0;
        jobCount--;
        MUTEX_UNLOCK(workingMutex);
#endif
#endif
    }
}

/* create and run the thread */
bool XThread::Start() 
{
    toBreak = false;
    isRunning = true;

#ifdef USE_PTHREAD
    int r = pthread_create(&hnd, NULL, &Wrapper, static_cast<void *>(this));
    if(r != 0)
        return false;
#else
#ifdef _WIN32
    DWORD id;
    hnd = BEGINTHREAD(0, 0, &Wrapper, this, 0, &id);
    if(hnd == 0)
        return false;
#else
    Run();
#endif
#endif
    return true;
}

/* end the thread */
void XThread::End()
{
    toBreak = true;
    if(isRunning == false)
        return;

    while(jobCount > 0){
#ifdef _WIN32
        Sleep(200);
#else
        usleep(200 * 1000);
#endif
    };


#ifdef USE_PTHREAD
    //MUTEX_LOCK(mutex);
    jobCount++;
    //COND_BROADCAST(cond);
    //COND_SIGNAL(cond);
    //MUTEX_UNLOCK(mutex);
    COND_BROADCAST(cond);
#else
    COND_SIGNAL(jobCond);
#endif

    Join();

    isRunning = false;
}

/* wait for thread termination */
void XThread::Join() 
{
#ifdef USE_PTHREAD
    pthread_join(hnd, 0);
#else
#ifdef _WIN32
    WaitForSingleObject(hnd, INFINITE);
    CloseHandle(hnd); // are you sure if you want to do this?
#endif
#endif
}

/* let the thread process a job */
void XThread::LetItGo()
{
#ifdef USE_PTHREAD
    MUTEX_LOCK(mutex);
    jobCount++;
    MUTEX_UNLOCK(mutex);
#else
#ifdef _WIN32
    /* reset various locks */
    MUTEX_LOCK(workingMutex);
    jobCount++;
    COND_RESET(jobCond);
    MUTEX_UNLOCK(workingMutex);

    /* inform the job */
    COND_SIGNAL(jobCond);
#endif
#endif
}

/* waith for a singal */
void XThread::Wait(COND_HANDLE * c, MUTEX_HANDLE * m)
{
#ifdef USE_PTHREAD
        MUTEX_LOCK(*m);
        COND_WAIT(*c, *m);
        MUTEX_UNLOCK(*m);
#else
#ifdef _WIN32
        COND_WAIT(*c, *m);
#endif
#endif
}

/***********************************************
a counter with mutex 
*/

/* constructor */
XCounter::XCounter()
{
    count = 0;
    MUTEX_INIT(mutex);
}

/* deconstructor */
XCounter::~XCounter()
{
    MUTEX_DELE(mutex);
}

/* add the counter by 1 */
void XCounter::Add()
{
    MUTEX_LOCK(mutex);
    count++;
    MUTEX_UNLOCK(mutex);
}

/* get the counting number */
int XCounter::Get()
{
    MUTEX_LOCK(mutex);
    int c = count;
    MUTEX_UNLOCK(mutex);

    return c;
}

} /* end of the nts (NiuTrans.Tensor) namespace */
