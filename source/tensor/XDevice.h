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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2016-06-23
 *
 */

#ifndef __XDEVICE_H__
#define __XDEVICE_H__

#include "XThread.h"

#ifdef USE_CUDA

/* the CUDA stuff */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>

#endif

/* the nts (NiuTrans.Tensor) namespace */
namespace nts{

#define MAX_LENGTH_OF_DEVICE_NAME 64
#define MAX_CPU_NUM 16
#define MAX_GPU_NUM 16
#define MAX_DEVICE_NUM MAX_CPU_NUM+MAX_GPU_NUM
#define INVALID_DEVICE_ID -1000
#define CURRENT_GPU 1000
//#define CUDA_UVA 1 // Unified Virtual Address Space of Cuda

/*
a class that records the basic information for each GPU/CPU device
e.g., the memory limit, warp size of a GPU and etc.
*/
class XDevice
{
public:
    /* 
    device id 
    <0:  CPU memory
    >=0: GPU device ID
    */
    int devID;

    /* size of the memory */
    int memSize;

    /* warp size of an (Nvidia) GPU */
    int GPUWarpSize;

    /* indicates whether the device class has been initialized */
    bool isInitialized;

    /* 
    max grid size (or number of blocks) of an (Nvidia) GPU
    NOTE: the grid size is alone with three dimensions (x, y, z)
    */
    int GPUMaxGridSize[3];

    /*
    max block size (or number of threads per block) of an (Nvidia) GPU
    NOTE: the block size is alone with three dimensions (x, y, z)
    */
    int GPUMaxBlockSize[3];

    /* max thread number that is supported */
    int GPUMaxThreadNum;

    /* max (and optimal) thread number for a block */
    int GPUMaxThreadNumPerBlock;

    /* name of the device */
    char name[MAX_LENGTH_OF_DEVICE_NAME];

    /* name of the device */
    char name2[MAX_LENGTH_OF_DEVICE_NAME];

    /* specify whether Unified Virtual Address Space (UVA) is supported */
    bool isUVASupported;

    /* seed for random number generation */
    int seed;
    
#ifdef USE_CUDA
    /* mutex for handle (GPU cublas) */
    MUTEX_HANDLE cublasMutex;

    /* handle used for cublas */
    cublasHandle_t cublasHandle;

    /* specify if the handle is initialized */
    bool isHandleReady;
    
    /* specify if the generator is initialized */
    bool isGenReady;

    /* generater of random numbers */
    curandGenerator_t gen;
#endif


public:
    /* constructor */
    XDevice();

    /* de-constructor */
    ~XDevice();

    /* initialize it and get the device information */
    void Init(int myDevID);

    /* clear it */
    void Clear();

    /* reset it */
    void Reset();

#ifdef USE_CUDA
    /* get cublas handle */
    cublasHandle_t * GetCublasHandle();
#endif

    /* switch to a GPU device */
    static
    void SetGPUDevice(int devID);

    /* switch to a device (with fast GPU execution mode) */
    static
    void SetGPUDeviceFast(int devID);

    /* get current dev */
    static
    int GetGPUDevice();

    /* swith to a device (CPU or GPU) */
    static
    void SetDevice(int devID);

    /* swith to a device (CPU or GPU) with a backup of the device id */
    static
    void SetDevice(int devID, int &backupDevID);

    /* reset cuda flag for more efficient cuda execution */
    static
    void SetFastFlags();

    /* reset cuda flag for more efficient cuda execution (all devices) */
    static
    void SetFastFlagsAllDevices();
};

/*
a class for the management of devices
*/
class XDevManager
{
public:
    /* CPU device information */
    XDevice CPUs[MAX_CPU_NUM];

    /* number of CPUs */
    int nCPU;

    /* GPU device information */
    XDevice GPUs[MAX_GPU_NUM];

    /* number of GPUs */
    int nGPU;

    /* indicates whether the the management of devices has been initialized */
    bool isInitialized;

public:
    /* constructor */
    XDevManager();

    /* de-constructor */
    ~XDevManager();

    /* initialize it and get the CPU and GPU information */
    void Init();

    /* clear it */
    void Clear();

#ifdef USE_CUDA
    /* get the handle of GPU */
    cublasHandle_t * GetCudaHandle(const int devID);
#endif

    /* get grid and block sizes that max potential */
    int GetCudaThread(const int devID, const int n, int * gridSize, int * blockSize);

    /* get grid and block sizes that max potential (2-dimension assignment) */
    int GetCudaThread2D(const int devID, const int n, const int m, int nLimit, int * gridSize, int * blockSize);

    /* get device ids for the given device information */
    int GetDeviceIDs(char * devInfo, int * devIDs);

    /* show id sequence */
    void ShowDeviceIDs(char * devInfo, char * msg);

    /* show device information */
    void ShowDevInfo();

    /* get the device information in string */
    char * GetDevString(int devID);
};

/* managing the devices */
extern XDevManager GDevs;

/* keep the device config */

#define ProtectCudaDev(devID, devIDBackup) \
{ \
    cudaGetDevice(&devIDBackup); \
    if(devIDBackup != devID) \
        cudaSetDevice(devID); \
} \

#define BacktoCudaDev(devID, devIDBackup) \
{ \
    if(devIDBackup != devID) \
        cudaSetDevice(devIDBackup); \
} \

#define CheckDev(a, b) \
{ \
    if((a < 0 && b >= 0) || (a >= 0 && b < 0)){ \
        fprintf(stderr, "[ERROR] (%s line %d): we must run the code on the same device (%d vs %d)\n", __FILENAME__, __LINE__, a, b); \
        throw; \
    } \
    else if (a >= 0 && b >= 0 && a != b) { \
        fprintf(stderr, "[ERROR] (%s line %d): we must run the code on the same device (%d vs %d)\n", __FILENAME__, __LINE__, a, b); \
        throw; \
    } \
} \

} /* end of the nts (NiuTrans.Tensor) namespace */

#endif
