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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2016-06-23
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "XDevice.h"
#include "XGlobal.h"
#include "XThread.h"
#include "XList.h"

/* the nts (NiuTrans.Tensor) namespace */
namespace nts{

/*
for managing the devices
*/
XDevManager GDevs;

/* constructor */
XDevice::XDevice()
{
    stream = NULL;
    isInitialized = false;
    Clear();

#ifdef USE_CUDA
    MUTEX_INIT(cublasMutex);
    isHandleReady = false;
#endif
}

/* de-constructor */
XDevice::~XDevice()
{
#ifdef USE_CUDA
    MUTEX_DELE(cublasMutex);
    if(isHandleReady)
        cublasDestroy(cublasHandle);
    if(stream != NULL)
        delete stream;
    curandDestroyGenerator(gen);
#endif
}

/* initialize it and get the device information */
void XDevice::Init(int myDevID)
{
    Clear();

    devID = myDevID;
    seed = rand();

    /* CPU information */
    if(devID < 0){
        strcpy(name, "CPU");
        strcpy(name2, "CPU");
    }
    /* GPU information */
    else{
#ifdef USE_CUDA
        cudaDeviceProp prop;

        cudaSetDevice(myDevID);

        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, seed);

        if(cudaGetDeviceProperties(&prop, devID) != cudaSuccess){
            XPRINT1(0, stderr, "cannot get GPU(%d) information.", devID);
            exit(1);
        }

#ifdef CUDA_UVA
        if ((prop.major >= 2)
#ifdef _WIN32
            && prop.tccDriver
#endif
           )
        {
            /* this is a P2P capable GPU */
            isUVASupported = true;
        }
#endif

#ifdef USE_CUDA_RESURSION
        CheckNTErrors((prop.major > 3), "The code requires cuda computation ability > 3!");
#endif

        CheckNTErrors((prop.warpSize == 32), "warp != 32 may result in problems in this version of code!");

        memSize = (int)prop.totalGlobalMem;
        GPUWarpSize = prop.warpSize;
        GPUMaxGridSize[0] = prop.maxGridSize[0];
        GPUMaxGridSize[1] = prop.maxGridSize[1];
        GPUMaxGridSize[2] = prop.maxGridSize[2];
        GPUMaxBlockSize[0] = prop.maxThreadsDim[0];
        GPUMaxBlockSize[1] = prop.maxThreadsDim[1];
        GPUMaxBlockSize[2] = prop.maxThreadsDim[2];
        GPUMaxThreadNum = GPUMaxGridSize[0] * GPUMaxGridSize[1] * GPUMaxGridSize[2] *
                          GPUMaxBlockSize[0] * GPUMaxBlockSize[1] * GPUMaxBlockSize[2];
        GPUMaxThreadNumPerBlock = MAX_CUDA_THREAD_NUM_PER_BLOCK;
        strcpy(name, prop.name); 

        if(isUVASupported){
            cudaDeviceEnablePeerAccess(myDevID, 0);
            sprintf(name2, "GPU-%d[UVA] %s", devID, name);
        }
        else
            sprintf(name2, "GPU-%d %s", devID, name);

        stream = new XStream(0, devID);
#endif
    }

    isInitialized = true;
}

/* clear it */
void XDevice::Clear()
{
    devID = -100;
    memSize = 0;
    GPUWarpSize = 0;

    memset(GPUMaxGridSize, 0, sizeof(int) * 3);
    memset(GPUMaxBlockSize, 0, sizeof(int) * 3);

    GPUMaxThreadNum = 0;

    name[0] = 0;
    name2[0] = 0;

    isUVASupported = false;
    // TODO: cublasDestroy(cublasHandle);
}

#ifdef USE_CUDA

/* get cublas handle */
cublasHandle_t * XDevice::GetCublasHandle()
{
    if (!isInitialized)
        Init(devID);

    if(!isHandleReady){
        MUTEX_LOCK(cublasMutex);
        int devIDBackup = 0;
        ProtectCudaDev(devID, devIDBackup);
        CheckNTErrors(cublasCreate(&cublasHandle) == CUBLAS_STATUS_SUCCESS, 
                     "Cannot create the cublas handle.");
        isHandleReady = true;
        BacktoCudaDev(devID, devIDBackup);
        MUTEX_UNLOCK(cublasMutex);
    }

    return &cublasHandle;
}

/* get the stream of cuda */
cudaStream_t * XDevice::GetCudaStream()
{
    if (!isInitialized)
        Init(devID);

    CheckNTErrors(stream != NULL, "the stream is not initialized!");

    return &stream->stream;
}

#endif // USE_CUDA

/* switch to a device */
void XDevice::SetGPUDevice(int devID)
{
    if(devID < 0)
        return;

#ifdef USE_CUDA
    cudaError_t error = cudaSetDevice(devID);

    if (error != cudaSuccess){
        fprintf(stderr, "Error! Calling cudaSetDevice(%d) fails(%d:%s)\n",
                devID, error, cudaGetErrorString(error));
        exit(1);
    }
#else
    ShowNTErrors("Please specifly USE_CUDA and recompile the code!");
#endif
} // USE_CUDA

/* switch to a device (with fast GPU execution mode) */
void XDevice::SetGPUDeviceFast(int devID)
{
    SetGPUDevice(devID);
    SetFastFlags();
}

/* get the id of the current GPU device */
int XDevice::GetGPUDevice()
{
#ifdef USE_CUDA
    int devID;
    cudaError_t error = cudaGetDevice(&devID);

    if (error != cudaSuccess){
        fprintf(stderr, "Error! Calling cudaGetDevice(%d) fails(%d:%s)\n",
                devID, error, cudaGetErrorString(error));
        exit(1);
    }

    return devID;
#else
    ShowNTErrors("Please specify USE_CUDA and recompile the code!");
    return -1;
#endif
}

/* reset cuda flag for more efficient cuda execution. It should be called after "SetGPUDevice" when
   no GPU context has been established. */
void XDevice::SetFastFlags()
{
#ifdef USE_CUDA
    cudaError_t error = cudaSetDeviceFlags(cudaDeviceScheduleSpin|cudaDeviceLmemResizeToMax);
    if(error != cudaSuccess){
        fprintf(stderr, "Error! Calling cudaSetDeviceFlags fails(%d:%s)\n", error, cudaGetErrorString(error));
        exit(1);
    }
#endif
}

/* reset the cuda flag for more efficient cuda execution (all devices) */
void XDevice::SetFastFlagsAllDevices()
{
 #ifdef USE_CUDA
    int devNum = 0;
    cudaGetDeviceCount(&devNum);
    for (int i = 0; i < devNum; i++){
        cudaSetDevice(i);
        SetFastFlags();
    }
#endif
}

/* constructor */
XDevManager::XDevManager()
{
    Clear();
    Init();
}

/* de-constructor */
XDevManager::~XDevManager()
{
}


/* initialization */
void XDevManager::Init()
{
    srand((unsigned int)time(NULL));

    Clear();

    /* CPUs (we actually do not care about how many CPUs are using) */
    nCPU = 1;

    for(int i = 0; i < nCPU; i++)
        CPUs[0].Init(-1);

    /* GPUs */
    int GPUCount = 0;

#ifdef USE_CUDA
    if(cudaGetDeviceCount(&GPUCount) != cudaSuccess){
        XPRINT(0, stderr, "cannot get GPU information.");
        exit(1);
    }

    for(int i = 0; i < GPUCount; i++){
        GPUs[i].devID = i;
        //GPUs[i].Init(i);
    }

#endif

    nGPU = GPUCount;
}

/* clear it */
void XDevManager::Clear()
{
    for(int i = 0; i < MAX_CPU_NUM; i++)
        CPUs[i].Clear();

    for(int i = 0; i < MAX_GPU_NUM; i++)
        GPUs[i].Clear();
}

#ifdef USE_CUDA

/* get the handle of a given GPU */
cublasHandle_t * XDevManager::GetCudaHandle(const int devID)
{
    CheckNTErrors(devID < nGPU, "index of GPU is out of range.");

    return GPUs[devID].GetCublasHandle();
}

/* get the stream of a given GPU */
cudaStream_t * XDevManager::GetCudaStream(const int devID)
{
    CheckNTErrors(devID < nGPU, "index of GPU is out of range.");

    return GPUs[devID].GetCudaStream();
}

#endif

/* 
get grid and block sizes that max the potential 
this is for 1-dimension job assignment, e.g., segmenting vector
into blocks
>> devID - device ID
>> n - size of the job
>> gridSize - size of the grid, i.e., number of blocks along x
>> blockSize - size of the block, i.e., number of threads per block along x
<< return - succeed(0) or not
*/
int XDevManager::GetCudaThread(const int devID, const int n, int * gridSize, int * blockSize)
{
    if (!GPUs[devID].isInitialized)
        GPUs[devID].Init(devID);

    memset(gridSize, 0, sizeof(int) * 3);
    memset(blockSize, 0, sizeof(int) * 3);

    if(n <= 0 || devID >= nGPU)
        return 1;

    if(devID < 0){
        XPRINT(0, stderr, "WARNING! You are calling the grid and block size computation function on a CPU!");
        return 0;
    }

#ifdef USE_CUDA

    int nWarp = GPUs[devID].GPUMaxThreadNumPerBlock / GPUs[devID].GPUWarpSize;
    int bSize = nWarp * GPUs[devID].GPUWarpSize;

    unsigned int b = bSize;
    CheckNTErrors((!(b & (b-1))), "Block size must be in 2^x");

    int gSize = int(ceil(float(n)/bSize));

    CheckNTErrors((gSize <= GPUs[devID].GPUMaxGridSize[0]), "A too large grid size.");

    blockSize[0] = bSize;
    gridSize[0] = gSize;

    CheckNTErrors((blockSize[0] <= GPUs[devID].GPUMaxBlockSize[0]), "Cude block size is out of range!");
    CheckNTErrors((gridSize[0] <= GPUs[devID].GPUMaxGridSize[0]), "Cude grid size is out of range!");

#endif

    return 0;
}

#define pow2Num 13
unsigned int pow2[pow2Num] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};

/* 
get grid and block sizes that max the potential 
this is for 2-dimension job assignment, e.g., segmenting a matrix or vector
into blocks
>> devID - device ID
>> n - x size of the job
>> m - y size of the job
>> nLimit - max number of x
>> gridSize - size of the grid, i.e., number of blocks along x and y
>> blockSize - size of the block, i.e., number of threads per block along x and y
<< return - succeed(0) or not
*/
int XDevManager::GetCudaThread2D(const int devID, const int n, const int m, int nLimit, int * gridSize, int * blockSize)
{
    if (!GPUs[devID].isInitialized)
        GPUs[devID].Init(devID);

    memset(gridSize, 0, sizeof(int) * 3);
    memset(blockSize, 0, sizeof(int) * 3);

    if(n <= 0 || m <= 0)
        return 1;

    CheckNTErrors(devID >= 0 && devID < nGPU, "Invalid GPU device id!");

#ifdef USE_CUDA

    int bXSize = n;

    if(bXSize > nLimit)
        bXSize = nLimit;
    if(bXSize > GPUs[devID].GPUMaxThreadNumPerBlock)
        bXSize = GPUs[devID].GPUMaxThreadNumPerBlock;

    unsigned int b = bXSize;

    /* fit the number into pow(2,x) */
    if((b & (b-1))){
        bool ok = false;
        for(int i = 0; i < pow2Num - 1; i++){
            if(pow2[i] < b && b <= pow2[i + 1]){
                b = pow2[i + 1];
                bXSize = b;
                ok = true;
                break;
            }
        }
        CheckNTErrors((ok), "you have an illegal size of the x-axis in a cuda block!");
    }

    int bYSize = GPUs[devID].GPUMaxThreadNumPerBlock/bXSize;

    if(n * m < GPUs[devID].GPUMaxThreadNumPerBlock){
        if(n * m >= GPUs[devID].GPUWarpSize)
            bYSize = int(ceil((float)n * m / bXSize));
        else
            bYSize = int(ceil((float)GPUs[devID].GPUWarpSize / bXSize));
    }

    if(bYSize == 0)
        bYSize = 1;

    int gXSize = int(ceil(float(n)/bXSize));
    int gYSize = int(ceil(float(m)/bYSize));

    CheckNTErrors((!(b & (b-1))), "Block size (x-axis) must be in 2^x");
    CheckNTErrors((gXSize <= GPUs[devID].GPUMaxGridSize[0] && 
                   gYSize <= GPUs[devID].GPUMaxGridSize[1]), "A too large grid size.");

    blockSize[0] = bXSize;
    blockSize[1] = bYSize;
    gridSize[0] = gXSize;
    gridSize[1] = gYSize;

#endif

    return 0;
}

/* 
split a string 
>> inputString - a line of string
>> separator - separate by what
>> items - splitting result
<< return - how many items are there
*/
int SplitALine(char * inputString, const char * seperator, StrList* items)
{
    items->Clear();

    if(inputString == NULL || seperator == NULL)
        return 0;

    int inputLen = (int)strlen(inputString);
    int sepLen = (int)strlen(seperator);

    if(inputLen == 0)
        return 0;

    if(sepLen == 0){

        char * item = new char[inputLen + 1];
        strcpy(item, inputString);
        items->Add(item);
    }
    else{
        char * p = inputString;
        char * item = NULL;
        while(p != NULL){
            char * q = strstr(p, seperator);
            if(q == NULL){
                item = new char[inputLen - (p - inputString) + 1];
                memcpy(item, p, inputLen - (p - inputString) + 1);
                item[inputLen - (p - inputString)] = '\0'; // no use?
                p = NULL;
            }
            else{
                item = new char[q - p + 1];
                memcpy(item, p, q - p);
                item[q - p] = '\0';
                p = q + sepLen;
            }
            items->Add(item);
        }
    }

    return items->count;
}

/* 
get device ids for the given device information 
>> devInfo - device information, e.g.,
             devInfo = "0:CPU-1 1:GPU-0 2:CPU-1"
             means that the first device is CPU, the second device
             is GPU-0, the third device is CPU.
>> devIDs - device IDs specified by devInfo
<< return - number of devices
*/
int XDevManager::GetDeviceIDs(char * devInfo, int * devIDs)
{
	StrList* terms = new StrList(1);
    SplitALine(devInfo, " ", terms);

    for(int i = 0; i < terms->count; i++){
        int devC, devID;
        char dev[32] = "";
        char * curDevInfo = (char*)terms->GetItem(i);

        if(sscanf(curDevInfo, "%d:%s", &devC, dev) < 2){
            ShowNTErrors("Wrong device information. Use something like \"0:CPU-1 1:GPU-0 2:CPU-1\".");
        }

        char * p = strchr(dev, '-');

        if(devC != i || p == NULL || sscanf(p + 1, "%d", &devID) < 1){
            ShowNTErrors("Wrong device information. Use something like \"0:CPU-1 1:GPU-0 2:CPU-1\".");
        }

        *p = '\0';

        if(!strcmp(dev, "CPU")){
            devIDs[i] = -1;
        }
        else if(!strcmp(dev, "GPU")){
            devIDs[i] = devID;
        }
    }

    int devCount = terms->count;

    for(int i = 0; i < terms->count; i++)
        delete[] (char*)terms->GetItem(i);
    delete terms;

    return devCount;
}

/* show device IDs */
void XDevManager::ShowDeviceIDs(char * devInfo, char * msg)
{
    msg[0] = 0;
    int ids[MAX_DEVICE_NUM]; 
    int num = GetDeviceIDs(devInfo, ids);

    for(int i = 0; i < num; i++){
        if(i == 0)
            sprintf(msg, "%d", ids[i]);
        else
            sprintf(msg, "%s %d", msg, ids[i]);
    }
}

/* show device information */
void XDevManager::ShowDevInfo()
{
    XPRINT(1, stderr, "Device Information:\n");
    for(int i = 0; i < nCPU; i++){
        XPRINT(1, stderr, " - id:-1 CPU\n");
    }

    for(int i = 0; i < nGPU; i++){
        XPRINT2(1, stderr, " - id:%2d GPU %s\n", i, GPUs[i].name);
    }
}

/* get the device information in string */
char * XDevManager::GetDevString(int devID)
{
    if(devID < 0)
        return CPUs[0].name2;
    else{
        CheckNTErrors((devID < nGPU), "Illegal GPU id.");
        return GPUs[devID].name2;
    }
}

} /* end of the nts (NiuTrans.Tensor) namespace */

