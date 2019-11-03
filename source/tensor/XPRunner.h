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

#ifndef __XPRUNNER_H__
#define __XPRUNNER_H__

#include "XThread.h"
#include "XList.h"

/* the nts (NiuTrans.Tensor) namespace */
namespace nts{

#define MIN_OPERATION_NUM 1024 * 4
#define MAX_JOB_NUM 32
#define MAX_THREAD_NUM 32

#define PRUNNER_SINGLE 0
#define PRUNNER_MULTIPLE 1
#define PRUNNER_GPU 2

/*
The XPRunner maintains a the parallel processing resources, e.g., a pool
of threads. It can provide the parallel computation interface for someone
that needs to do something parallel, e.g., speed-up matrix operation by
multi-threading.
*/
class XPRunner
{
public:
    /* 
    method of parallelization
    // 0: single job; 1: multi-threading; 2: gpu
    */
    int method;
public:
    /* a set of threads */
    XThread * threads;

    /* max number of threads */
    int threadNum;

    /* a mutex lock */
    MUTEX_HANDLE mutex;

    /* 
    Minimum number of atomic operations for a thread.
    It is used to avoid large overhead of too many "tiny" jobs.
    */
    int minimumOPNum;

    /* if multi-threading is activated */
    bool isMultiThreaded;

    /* list of running threads */
    int * runningThreads;

     /* list of threads states */
    int * runningStates;

    /* number of running threads */
    int runningThreadNum;

    /* list of available threads */
    int * availableThreads;

    /* number of available threads */
    int availableThreadNum;

/* general methods */
public:
    /* constructor */
    XPRunner();

    /* deconstructor */
    ~XPRunner();

    /* initialization */
    void Init(int myThreadNum);

/* methods for multi-threading */
public:
    /* initialization */
    void CreateThreads(int tNum);

    /* kill all running threads in the pool */
    void KillThreads();

    /* run a set of jobs in parallel */
    void Run(TensorList * jobFunctions, TensorList * jobArgs, float sleepTime = 0);

    /* get the number of parallel jobs to run */
    int GetJobNum(int size);
};

extern XPRunner * globalPRunner;

} /* end of the nts (NiuTrans.Tensor) namespace */

#endif