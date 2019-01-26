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
 * This is for streaming (on GPU), i.e., run jobs in different stream for 
 * GPU Async capabilities.
 *
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2016-03-09
 *
 */

#ifndef __XSTREAM_H__
#define __XSTREAM_H__

/* the CUDA stuff */
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#endif

/* the nts (NiuTrans.Tensor) namespace */
namespace nts{

#define MAX_CUDA_EVENT_NUM_IN_A_STREAM 128

/*
This class defines the stream used in pipelining jobs. E.g., one can put
a sequence of jobs in a stream and asychronously do something else. Basically
we can use multiply streams to hide the data transfer cost on GPUs by using
job overlaps.
*/
class XStream
{
public:
#ifdef USE_CUDA
    /* the cuda stream */
    cudaStream_t stream;

    /* list of cuda events for synchronize different streams */
    cudaEvent_t * events;

    /* max number of the events */
    int maxEventNum;

    /* number of used events */
    int usedEventNum;
#else
    /* virtual pointer */
    void * stream;
#endif


    /* device that holds the stream */
    int devID;

public:
    /* constructor */
    XStream(int priority = 0, int devID = 0, int maxEventNum = MAX_CUDA_EVENT_NUM_IN_A_STREAM);

    /* deconstructor */
    ~XStream();

    /* create the stream */
    void Create(int priority = 0, int devID = 0);

    /* destroy the stream */
    void Destroy();

    /* clear it */
    void Clear();

    /* judge if all the jobs in the stream have been finished */
    bool IsFinished();

    /* stream synchronize */
    void StreamSynchronize();

    /* thread synchronize */
    static 
    void ThreadSynchronize();

    /* device synchronize */
    static 
    void DeviceSynchronize(int devID);

    /* make a dependency of two streams. i.e., current stream must wait for the last job finished in another stream */
    void MakeDependency(XStream * precedingStream);

#ifdef USE_CUDA
    /* get the stream */
    cudaStream_t * Get();

    /* make a event */
    cudaEvent_t * MakeEvent();
#endif  
};

} /* end of the nts (NiuTrans.Tensor) namespace */

#endif
