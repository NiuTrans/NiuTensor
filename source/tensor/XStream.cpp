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
 *
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2016-03-09
 *
 */

#include "stdio.h"
#include "stdlib.h"
#include "XGlobal.h"
#include "XStream.h"
#include "XDevice.h"

/* the nts (NiuTrans.Tensor) namespace */
namespace nts{

/*
This class defines the stream used in pipelining jobs. E.g., one can put
a sequence of jobs in a stream and asynchronously do something else. Basically
we can use multiply streams to hide the data transfer cost on GPUs by using
job overlaps.
*/

/* constructor */
XStream::XStream(int priority, int myDevID, int myMaxEventNum)
{
    devID = myDevID;
#ifdef USE_CUDA
    if(myDevID >= 0){
        int backupDevID = XDevice::GetGPUDevice();
        XDevice::SetGPUDevice(myDevID);
        events = new cudaEvent_t[myMaxEventNum];
        XDevice::SetGPUDevice(backupDevID);

        maxEventNum = myMaxEventNum;
        usedEventNum = 0;
    }
    else{
        maxEventNum = 0;
        usedEventNum = 0;
    }
#endif

    Create(priority, devID);
}

/* deconstructor */
XStream::~XStream()
{
    Destroy();
#ifdef USE_CUDA
    delete[] events;
#endif
}

/* create the stream */
void XStream::Create(int priority, int myDevID)
{
    if(myDevID < 0)
        return;

#ifdef USE_CUDA
    int backupDevID = XDevice::GetGPUDevice();
    XDevice::SetGPUDevice(myDevID);
    //cudaStreamCreateWithPriority(&stream, cudaStreamDefault, priority);
    CheckNTErrors((cudaStreamCreate(&stream) == cudaSuccess), 
                  "cannot create the cuda stream!");
    XDevice::SetGPUDevice(backupDevID);
#endif
    devID = myDevID;
}

/* destroy the stream */
void XStream::Destroy()
{
    if(devID < 0)
        return;

#ifdef USE_CUDA
    int backupDevID = XDevice::GetGPUDevice();
    XDevice::SetGPUDevice(devID);
    cudaStreamDestroy(stream);
    XDevice::SetGPUDevice(backupDevID);
    Clear();
#endif
}

/* clear it */
void XStream::Clear()
{
#ifdef USE_CUDA
    int backupDevID = XDevice::GetGPUDevice();
    XDevice::SetGPUDevice(devID);
    for(int i = 0; i < usedEventNum; i++){
        cudaEventDestroy(events[i]);
    }
    usedEventNum = 0;
    XDevice::SetGPUDevice(backupDevID);
#endif
}

/* judge if all the jobs in the stream have been finished */
bool XStream::IsFinished()
{
#ifdef USE_CUDA
    if(cudaStreamQuery(stream) == cudaSuccess)
        return true;
    else
        return false;
#else
    return true;
#endif
}

void XStream::StreamSynchronize()
{
#ifdef USE_CUDA
    int devIDBackup = XDevice::GetGPUDevice();
    if(devID != devIDBackup)
        XDevice::SetGPUDevice(devID);
    cudaStreamSynchronize(stream);
    if(devID != devIDBackup)
        XDevice::SetGPUDevice(devIDBackup);
#endif
}

void XStream::ThreadSynchronize()
{
#ifdef USE_CUDA
    cudaThreadSynchronize();
#endif
}

void XStream::DeviceSynchronize(int devID)
{
#ifdef USE_CUDA
    int devIDBackup = XDevice::GetGPUDevice();
    cudaGetDevice(&devIDBackup);
    if(devID != devIDBackup)
        XDevice::SetGPUDevice(devID);
    cudaDeviceSynchronize();
    if(devID != devIDBackup)
        XDevice::SetGPUDevice(devIDBackup);
#endif
}

/* make a dependency of two streams. i.e., current stream must wait for the last job finished in another stream */
void XStream::MakeDependency(XStream * precedingStream)
{
#ifdef USE_CUDA
    cudaEvent_t * e = precedingStream->MakeEvent();
    cudaEventRecord(*e, precedingStream->stream);
    cudaStreamWaitEvent(stream, *e, 0);
#endif
}


/* get the stream */
#ifdef USE_CUDA
inline cudaStream_t * XStream::Get()
{
    return &stream;
}

/* make a event */
inline cudaEvent_t * XStream::MakeEvent()
{
    int backupDevID = XDevice::GetGPUDevice();
    XDevice::SetGPUDevice(devID);
    CheckNTErrors((usedEventNum < maxEventNum), "Too many events are required!");
    cudaEvent_t * e = events + usedEventNum++;
    cudaEventCreate(e);
    XDevice::SetGPUDevice(backupDevID);
    return e;
}
#endif

} /* end of the nts (NiuTrans.Tensor) namespace */

