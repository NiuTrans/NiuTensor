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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2016-5-25
 *
 */

#ifndef __XMEM_H__
#define __XMEM_H__

#include <stdio.h>
#include <stdlib.h>

#ifdef CUDA_BLAS
#define USE_CUDA
#endif

#ifdef USE_CUDA
// the CUDA stuff
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
#endif

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#elif WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

/* the nts (NiuTrans.Tensor) namespace */
namespace nts{

typedef unsigned long long MTYPE;
typedef long long int      MTYPEINT;
typedef long long          INT_64;

//#define CUDA_PITCH 256
#define CUDA_PITCH 1
#define CUDA_HOST_MALLOC 1
#define MY_PITCH CUDA_PITCH
#define BUF_PITCH 256
#define MIN_BLOCK_SIZE_FOR_MEMPOOL 256 * 1024 * 1024
#define MIN_BLOCK_NUM_FOR_MEMPOOL 1024
#define MAX_CPU_MEM_NUM 16
#define MAX_GPU_MEM_NUM 16

/* 
mode of runnig a memory pool 
- UNI_FREE: free all memory space when the memory allocation is no use
- FREE_ON_THE_FLY: run in normal "malloc" and "free" ways
*/
enum MEMPOOL_MODE {UNI_FREE, FREE_ON_THE_FLY};
    
struct MPieceNode;

/* header of a memory piece (FREE_ON_THE_FLY) */
struct MHeader
{
    /* state of the memory piece 
       1: free
       2: in use
    */
    int state;

    /* size of the allocated memory */
    MTYPE size;

    /* pointer to the header of the previous memory piece */
    MHeader * pre;

    /* pointer to the header of the next memory piece */
    MHeader * next;

    /* id of the memory block */
    int blockID;
    
    /* pointer to the index node */
    MPieceNode * indexNode;
};

/* index of memory piece */
struct MPieceNode
{
    /* size of the memory piece */
    MTYPE size;

    /* previous node */
    MPieceNode * pre;

    /* next node */
    MPieceNode * next;

    /* pointer to the head of a memory piece */
    void * p;
    
    /* pointer to the head of memory that is returned back to the user */
    void * pReal;

    /* header of the memory piece */
    MHeader head;
};

/* memory block */
struct XMemBlock
{
    /* pointer to where to start */
    void * mem;

    /* size of the block */
    MTYPE size;

    /* size of the used memory in this block */
    MTYPE used;

    /* desired size of the block */
    MTYPE sizeDesired;
    
    /* first head of the block */
    MHeader * head;
};

/* 
memory pool.
Basically a memory pool runs in two ways. It can be an naive implementation (uni-free mode)
that allocates the memory in a continuous manner and free the used memory space
in the end - the user does not need to call the "memfree" function
when a small piece of memory is not used any more. Instead we call the "go back"
function to the initial state when all memory space in the memory pool is not in use.
Another way (free on-the-fly mode) is to allocate and free the memory space as in standard 
"malloc" and "free" manners. Here we do it on a pre-allocated memory block. This mode is 
more flexible but relatively slower than the uni-free mode.
*/
class XMem
{
public:
    /* 
    device id 
    <0:  CPU memory
    >=0: GPU device ID
    */
    int devID;

    /* mode of running the memory pool */
    MEMPOOL_MODE mode;

    /* signature */
    MTYPE signature;

    /* indicates whether the memory allocation is static */
    bool isStatic;

    /* memory blocks */
    XMemBlock * blocks;

    /* number of memory blocks */
    int blockNum;

    /* max size of a block */
    MTYPE maxBlockSize;

    /* total size of all memory blocks */
    MTYPE totalBlockSize;

    /* total size of used memory */
    MTYPE totalUsed;

    /* current mem block that is using */
    XMemBlock * curBlock;

    /* id of the current mem block */
    int curBlockID;

    /* id of the final mem block that is used */
    int finalBlockID;

    /* pointer to the buffer used to store temp data */
    void * buf;

    /* size of the buffer */
    MTYPE bufSize;

    /* size of the used buffer in this block */
    MTYPE bufUsed;

    /* name of the memory pool */
    char * name;

    /* pin for the memory. It is used for recording the starting address
       before we apply for a large amount of the memory. Then we can go
       back to this point for memory release. */
    int curBlockPin;
    MTYPE curUsedPin;
    MTYPE bufUsedPin;

    /* indicates whether the memory pool is initialized */
    bool isInitialized;

#ifdef USE_CUDA
    /* handle used for cublas */
    cublasHandle_t cublasHandle;

    /* random number generator for cuda code */
    curandGenerator_t randGen;
#endif

public:
    /* index of the free memory pieces */
    MPieceNode * memIndex;
    
    /* for double buffering */
    MPieceNode * memIndex2;

    /* maximum number of index nodes */
    INT_64 nodeNum;

    /* count of the used nodes */
    INT_64 nodeNumUsed;

    /* minimal size allocation for each index entry */
    MTYPE * minSizeIndex;

    /* number of the index entries */
    int indexEntryNum;

    /* index offset */
    int indexOffset;

    /* indicates whether we merge free memory pieces on the fly */
    bool mergeFreeOTF;

public:

    /* constructor */
    XMem();

    /* constructor */
    XMem(int myDevID,
         MEMPOOL_MODE myMode = UNI_FREE,
         MTYPE myBlockSize = MIN_BLOCK_SIZE_FOR_MEMPOOL, 
         int myBlockNum = MIN_BLOCK_NUM_FOR_MEMPOOL, 
         MTYPE myBufSize = 0);

    /* deconstructor */
    ~XMem();

    /* initialize it */
    void Initialize(int myDevID, MEMPOOL_MODE myMode, MTYPE myBlockSize, int myBlockNum, MTYPE myBufSize);

    /* free memory */
    void Free();

    /* free a piece of memory */
    void Free(int myDevID, void * mem);

    /* get signature */
    MTYPE GetSignature();

    /* use string as the name of the memory pool */
    void SetName(const char * myName);

    /* switch to the device we want to work */
    void SetDevice(int myDevID);

    /* switch to the device (with fast cuda execution mode) we want to work */
    void SetDeviceFast(int myDevID);

    /* run in static mode */
    void SetStaticMode(bool myIsStatic);

    /* specify if the memory pool is used for tensor computation (rather
       than storage */
    void SetComputationMode(bool myIsForComputation);

    /* initialize the index */
    void SetIndex(INT_64 size, MTYPE minSizeFirst = 256, int minSizeNum = 20);

    /* get device id */
    int GetDevID();

    /* set desired memory block size */
    void SetDesiredSize(int myDevID, int blockID, MTYPE mySize);

    /* require a piece of memory */
    void * Alloc(MTYPE mySize);

    /* require a piece of memory */
    void * Alloc(int myDevID, MTYPE mySize);

    /* require a piece of memory in a dynamic manner */
    void * AllocDynamic(int myDevID, MTYPE mySize);

    /* require a piece of memory with fixed size (if possible) */
    void * AllocStatic(int myDevID, MTYPE mySize);

    /* require a piece of memory that is not in the memory pool */
    void * AllocGlobal(int myDevID, MTYPE mySize);

    /* get the available size of the memory that can be used */
    MTYPE GetAvailableSize(int myDevID);

    /* require a piece of memory in the buffer */
    void * AllocBuf(int myDevID, MTYPE mySize, int pitch = BUF_PITCH);

    /* release a piece of memory */
    void Release(void * p, MTYPE size, MTYPE code);

    /* release a piece of memory */
    void Release(int myDevID, void * p, MTYPE size);

    /* release a piece of memory in the buffer */
    void ReleaseBuf(int myDevID, MTYPE mySize, int pitch = BUF_PITCH);

    /* release a piece of memory that is not in the memory pool */
    void ReleaseGlobal(int myDevID, void * p);

    /* allocate a piece of memory as "malloc" */
    void * AllocStandard(int myDevID, MTYPE mySize, bool myIsRebuiltIndex = false);

    /* find the highest set bit (or most significant set bit) in an integer-64 */
    int GetMSB(MTYPE mySize);

    /* find the index entry for allocation query */
    int FindIndexEntry(MTYPE mySize);

    /* remove an index node for available memory pieces */
    void RemoveIndexNode(MPieceNode * node, MPieceNode * entry = NULL);

    /* add an index node for available memory pieces */
    void AddFreeIndexNode(MPieceNode * node, MPieceNode * entry = NULL);
    
    /* remove an index node for memory pieces in use */
    void RemoveAllocIndexNode(MPieceNode * node, MPieceNode * entry = NULL);
    
    /* add an index node for available memory pieces */
    void AddAllocIndexNode(MPieceNode * node, MPieceNode * entry = NULL);

    /* release a piece of memory as "free" */
    void ReleaseStandard(int myDevID, void * p, MTYPE size);

    /* rebuild index to merge small fragments of memory and free the block with no use */
    void RebuildIndex();

    /* reset buffer */
    void Reset(int myDevID);

    /* get pitch for aligned memory */
    MTYPE GetPitch(int myDevID, MTYPE baseAddress, MTYPE mySize);

    /* get pitched address for aligned memory */
    void * GetPitchedAddress(void * address, MTYPE pitch);

    /* get current address (for use) */
    void * GetAddress();

    /* clear it */
    void Clear();

    /* clear the buffer */
    void ClearBuf();

    /* clear the memory pool and the buffer */
    void ClearAll();

    /* set a variable to the input value */
    static
    void Copy(void * tgt, void * src, int size, XMem * tgtMem = NULL, XMem * srcMem = NULL);

    /* set a float-typed variable to the input value */
    static
    void CopyFloat(float * tgt, float * src, XMem * tgtMem = NULL, XMem * srcMem = NULL);

    /* set a variable to 0 */
    static
    void SetZero(void * tgt, MTYPE size, XMem * tgtMem = NULL);

    /* record the pin point */
    void SetPin();

    /* go back to the pin point */
    void BackToPin();

    /* record the pin point for buffer */
    void SetPinBuf();

    /* go back to the pin point for buffer */
    void BackToPinBuf();

    /* transform a size into a number (in million) */
    static
    MTYPE GetMemSize(const char * size);

    /* transform a size into a number (in Bytes) */
    static
    MTYPE GetMemSizeInBytes(const char * size);

    /* create a new cublas handle */
    void CreateBLASHandle();

    /* show profile of the memory pool */
    void ShowMemUsage(FILE * file);

#ifdef USE_CUDA
    /* get the handle of cublas */
    cublasHandle_t * GetCublasHandle();
#endif

};

/*
a class for the management of memory
*/
class XMemManager
{
private:
    /* cpu memory pool information */
    XMem CPUMems[MAX_CPU_MEM_NUM];

    /* number of cpu memory pools */
    int nCPUMem;

    /* gpu memory pool information */
    XMem GPUMems[MAX_GPU_MEM_NUM];

    /* number of gpu memory pools */
    int nGPUMem;

public:
    /* constructor */
    XMemManager();

    /* de-constructor */
    ~XMemManager();

    /* get memory size */
    MTYPE GetAvailableMemory();

    /* get GPU memory size */
    MTYPE GetAvailableGPUMemory(int devID);

    /* get buffer size */
    void GetBufferSize(MTYPE freeMem, MTYPE * myBufSize);

    /* initialize it and set the global memory information */
    void Initialize();

    /* free it */
    void Free();

    /* get global memory pool */
    XMem * GetMem(const int devID);

    /* get global memory size */
    int GetMemSize(const int devID, MTYPE * myBlockSize, int * myBlockNum, MTYPE * myBufSize);

    /* show memory information */
    void ShowMemInfo();
};

/* managing the memories */
extern XMemManager GMems;



extern XMem * GMem;

extern int testxmemid;
extern void * recordp;

} /* end of the nts (NiuTrans.Tensor) namespace */

#endif
