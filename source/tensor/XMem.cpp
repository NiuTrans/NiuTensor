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

#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "XGlobal.h"
#include "XUtility.h"
#include "XMem.h"

/* the nts (NiuTrans.Tensor) namespace */
namespace nts{
    
int testxmemid = 0;
void * recordp = NULL;

/*
for managing the memories
*/
XMemManager GMems;

XMem * GMem;

/* constructor */
XMem::XMem()
{
    memset(this, 0, sizeof(XMem));
    devID = -1;
    mode = UNI_FREE;
    curBlockPin = -1;
    indexOffset = -1;
    name = new char[64];
    strcpy(name, "xmem");
    signature = 0;
    mergeFreeOTF = true;
    isInitialized = false;
}

/* 
constructor 
>> myDevID - device id 
             -1:  CPU memory
             >=0: GPU device ID
>> myMode - mode of running the memory pool
            UNI_FREE: free all the space at the end of using the memory pool
            FREE_ON_THE_FLY: normal "malloc" and "free" mode
>> myBlockSize - size of a memory block
>> myBlockNum  - number of memory blocks
>> myBufSize - size of buffer
*/
XMem::XMem(int myDevID, MEMPOOL_MODE myMode, MTYPE myBlockSize, int myBlockNum, MTYPE myBufSize)
{
    memset(this, 0, sizeof(XMem));
    curBlockPin = -1;
    indexOffset = -1;
    name = new char[64];
    strcpy(name, "xmem");
    signature = 0;
    mergeFreeOTF = true;
    Initialize(myDevID, myMode, myBlockSize, myBlockNum, myBufSize);
}

/* deconstructor */
XMem::~XMem()
{
#ifdef USE_CUDA
    int devIDBackup = -1;
    cudaGetDevice(&devIDBackup);
    SetDevice(devID);

    if(devID >= 0 && cublasHandle != NULL)
        cublasDestroy(cublasHandle);
    curandDestroyGenerator(randGen);

    SetDevice(devIDBackup);
#endif
    Free();
    delete[] name;
    delete[] memIndex;
    delete[] memIndex2;
    delete[] minSizeIndex;
}

/* 
initialize it 
>> myDevID - device id 
             -1:  CPU memory
             >=0: GPU device ID
>> myMode - mode of running the memory pool
            UNI_FREE: free all the space at the end of using the memory pool
            FREE_ON_THE_FLY: normal "malloc" and "free" mode
>> myBlockSize - size of a memory block
>> myBlockNum  - number of memory blocks
>> myBufSize - size of buffer
*/
void XMem::Initialize(int myDevID, MEMPOOL_MODE myMode, MTYPE myBlockSize, int myBlockNum, MTYPE myBufSize)
{
    Free();

    CheckNTErrors((myBlockSize > 0 && myBlockNum > 0), "Illegal member block settings!");

    devID = myDevID;
    mode = myMode;
    maxBlockSize = myBlockSize;
    blockNum = myBlockNum;

    blocks = new XMemBlock[blockNum];
    for(int i = 0; i < blockNum; i++){
        blocks[i].mem = NULL;
        blocks[i].size = maxBlockSize;
        blocks[i].sizeDesired = maxBlockSize;
        blocks[i].used = 0;
    }

    curBlock = blocks;
    curBlockID = 0;
    finalBlockID = 0;

    if(myDevID < 0){
        buf = new char[(unsigned int)myBufSize];
    }
    else{
#ifdef USE_CUDA
        int devIDBackup = -1;
        cudaGetDevice(&devIDBackup);
        SetDevice(myDevID);

        CheckNTErrors(cudaMalloc((void **)&buf, myBufSize) == cudaSuccess, "Cannot allocate the memory.");
        CheckNTErrors(cudaMemset(buf, 0, myBufSize) == cudaSuccess, "Cannot update the memory.");
        CheckNTErrors(curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT) == CURAND_STATUS_SUCCESS, "Cannot make the cuda random number generator!");
        CheckNTErrors(curandSetPseudoRandomGeneratorSeed(randGen, (unsigned)time(NULL)) == CURAND_STATUS_SUCCESS, "Cannot generate the seed!");

        SetDevice(devIDBackup);

        /* create the cublas handle */
        SetComputationMode(true);
#else
        ShowNTErrors("Please specify USE_CUDA for compiling this program.");
#endif
    }

    bufSize = myBufSize;

#ifdef SMALL_DEVICE
    if (myMode == FREE_ON_THE_FLY)
        SetIndex(50000);
#else
    if (myMode == FREE_ON_THE_FLY)
        SetIndex(MILLION);
#endif

    signature++;
    isInitialized = true;
}

/* free memory */
void XMem::Free()
{
    for(int i = 0; i < blockNum; i++){
        Free(devID, blocks[i].mem);
    }
    delete[] blocks;
    blocks = NULL;

    Free(devID, buf);
    buf = NULL;
    bufSize = 0;
    bufUsed = 0;

    devID = -1;
}

/* 
free a piece of memory 
>> myDevID - device id(-1: CPU memory, >=0: GPU device ID)
>> mem - address of the memory block to release
*/
void XMem::Free(int myDevID, void * mem)
{
    if(mem == NULL)
        return;

    /* on CPUs */
    if(myDevID < 0){
        delete[] (char*)mem;
    }
    /* on GPUs */
    else{
#ifdef USE_CUDA
        int devIDBackup = -1;
        cudaGetDevice(&devIDBackup);
        SetDevice(myDevID);

        cudaError_t error = cudaFree((char*)mem);
        if(error != cudaSuccess){
            ShowNTErrors("Cannot free the memory.");
        }

        SetDevice(devIDBackup);
#else
        ShowNTErrors("Please specify USE_CUDA for compiling this program.");
#endif
    }
}

/*
get the signature
<< return - the signature
*/
MTYPE XMem::GetSignature()
{
    return signature;
}

/* 
set the name of the memory pool 
>> myName - name of the memory pool
*/
void XMem::SetName(const char * myName)
{
    delete[] name;
    name = new char[(int)strlen(myName) + 1];
    strcpy(name, myName);
}

/* 
switch to the device where we intend to work 
>> myDevID - device id(-1: CPU memory, >=0: GPU device ID)
*/
void XMem::SetDevice(int myDevID)
{
    if(myDevID < 0)
        return;

#ifdef USE_CUDA
    cudaError_t error = cudaSetDevice(myDevID);

    if (error != cudaSuccess){
        fprintf(stderr, "Error! Calling cudaSetDevice(%d) fails(%d:%s)\n", myDevID, error, cudaGetErrorString(error));
        exit(1);
    }

#else
    ShowNTErrors("Please specify USE_CUDA for compiling this program.");
#endif
}

/* 
switch to the device (with fast cuda execution mode) we intend to work on
>> myDevID - device id(-1: CPU memory, >=0: GPU device ID)
*/
void XMem::SetDeviceFast(int myDevID)
{
    SetDevice(myDevID);
#ifdef USE_CUDA
    cudaError_t error = cudaSetDeviceFlags(cudaDeviceScheduleSpin|cudaDeviceLmemResizeToMax);
    if (error != cudaSuccess){
        fprintf(stderr, "Error! Calling cudaSetDeviceFlags(%d) fails(%d:%s)\n", myDevID, error, cudaGetErrorString(error));
        exit(1);
    }
#endif
}

/* 
run in the static mode
>> myIsStatic - specify if the memory allocation is static
*/
void XMem::SetStaticMode(bool myIsStatic)
{
    isStatic = myIsStatic;
}

/* 
specify if the memory pool is used for tensor computation (rather
than storage 
>> myIsForComputation - specify if the memory pool is used in computation (if
                        so we need to create some handles for calling the BLAS interfaces)
*/
void XMem::SetComputationMode(bool myIsForComputation)
{
#ifdef USE_CUDA
    int devIDBackup = -1;
    cudaGetDevice(&devIDBackup);
    SetDevice(devID);

    if(!myIsForComputation && devID >= 0 && cublasHandle != NULL)
        cublasDestroy(cublasHandle);
    if(myIsForComputation)
        CheckNTErrors((enum curandStatus)cublasCreate(&cublasHandle) == CURAND_STATUS_SUCCESS, 
                      "Cannot create the cublas handle.");

    SetDevice(devIDBackup);
#endif
}

/*
initialize the index
>> indexSize - size of the index
>> minSizeFirst - minimal size allocation for the first entry
>> minSizeNum - number of minimal-size index nodes
*/
void XMem::SetIndex(INT_64 indexSize, MTYPE minSizeFirst, int minSizeNum)
{
    delete[] memIndex;
    delete[] memIndex2;
    delete[] minSizeIndex;

    nodeNum = indexSize;
    nodeNumUsed = minSizeNum * 2;
    indexEntryNum = minSizeNum;
    
    memIndex = new MPieceNode[nodeNum];
    memset(memIndex, 0, sizeof(MPieceNode) * nodeNum);
    
    memIndex2 = new MPieceNode[nodeNum];
    memset(memIndex2, 0, sizeof(MPieceNode) * nodeNum);

    minSizeIndex = new MTYPE[indexEntryNum];
    memset(minSizeIndex, 0, sizeof(MTYPE) * indexEntryNum);

    minSizeIndex[0] = minSizeFirst;
    for(int i = 1; i < indexEntryNum; i++)
        minSizeIndex[i] = minSizeIndex[i - 1] * 2;

    indexOffset = GetMSB(minSizeFirst);
}

/* get device id */
int XMem::GetDevID()
{
    return devID;
}

/* set desired memory block size */
void XMem::SetDesiredSize(int myDevID, int blockID, MTYPE mySize)
{
    CheckNTErrors((blockID >= 0 && blockID < blockNum), "Illegal block id!");
    CheckNTErrors((mySize > 0), "Illegal block size!");
    CheckNTErrors((blocks[blockID].mem == NULL), "Cannot reset a memory block that is being used!");

    blocks[blockID].sizeDesired = mySize;
    blocks[blockID].size = mySize;
}

/* 
require a piece of memory 
>> mySize - size of the require memory
*/
void * XMem::Alloc(MTYPE mySize)
{
    return Alloc(devID, mySize);
}

/* 
require a piece of memory 
>> myDevID - device id(-1: CPU memory, >=0: GPU device ID)
>> mySize - size of the require memory
*/
void * XMem::Alloc(int myDevID, MTYPE mySize)
{
    if(mode == FREE_ON_THE_FLY)
        return AllocStandard(myDevID, mySize);
    else if(isStatic)
        return AllocStatic(myDevID, mySize);
    else
        return AllocDynamic(myDevID, mySize);
}

/* 
require a piece of memory in a dynamic manner 
>> myDevID - device id(-1: CPU memory, >=0: GPU device ID)
>> mySize - size of the require memory
*/
void * XMem::AllocDynamic(int myDevID, MTYPE mySize)
{
    int ID;
    XMemBlock * b = NULL;
    bool firstHit = false;

    for (ID = curBlockID; ID < blockNum; ID++) {
        b = blocks + ID;
        if (!firstHit && b->size > b->used) {
            firstHit = true;
            curBlockID = ID;
            curBlock = blocks + curBlockID;
        }
        if (b->size >= b->used + mySize)
            break;
    }

    CheckNTErrors((curBlockID < blockNum), "No enough memory blocks.");
    CheckNTErrors((ID < blockNum), "Cannot find a available memory block. Please use a larger memory pool.");
    CheckNTErrors((b->size - b->used >= mySize), "Cannot allocate the memory. Please use a larger memory block!");

    if (ID > finalBlockID)
        finalBlockID = ID;

    char * mem = NULL;
    char * required = NULL;
    int backOffset = 0;

    /* allocate the memory */
    if (b->mem == NULL && b->used == 0) {
        /* on CPUs */
        if (myDevID < 0) {
            mem = new char[(unsigned int)b->size + 2 * CUDA_PITCH];
            memset(mem, 0, (unsigned int)b->size + 2 * CUDA_PITCH);
        }
        /* on GPUs */
        else {
#ifdef USE_CUDA
            int devIDBackup = -1;
            cudaGetDevice(&devIDBackup);
            SetDevice(myDevID);
            cudaError_t e = cudaMalloc((void **)&mem, b->size + 2 * CUDA_PITCH);
            if (e != cudaSuccess) {
                ShowNTErrors("Cannot allocate the memory.");
            }
            CheckNTErrors(cudaMemset(mem, 0, b->size + 2 * CUDA_PITCH) == cudaSuccess, "Cannot update the memory.");
            SetDevice(devIDBackup);
#else
            ShowNTErrors("Please specify USE_CUDA for compiling this program.");

#endif
        }

        b->mem = mem;
    }

#ifdef USE_CUDA
    if (myDevID >= 0) {
        long long address = (long long)((char*)b->mem + b->used);
        int offset = address % CUDA_PITCH;
        backOffset = offset > 0 ? CUDA_PITCH - offset : 0;
    }
#endif

    required = (char*)b->mem + b->used + backOffset;
    b->used += mySize + backOffset;

#ifdef USE_CUDA
    if (myDevID >= 0) {
        CheckNTErrors(((long long)required % CUDA_PITCH == 0), "The GPU memory is not aligned.");
    }
#endif

    CheckNTErrors((b->size + 2 * CUDA_PITCH >= b->used), "Something is wrong with the memory block.");

    return required;
}

/* 
required a piece of memory with fixed size (if possible) 
>> myDevID - device id(-1: CPU memory, >=0: GPU device ID)
>> mySize - size of the require memory
*/
void * XMem::AllocStatic(int myDevID, MTYPE mySize)
{
    for(int ID = curBlockID; ID < blockNum; ID++){
        XMemBlock * b = blocks + ID;
        if(b->mem == NULL){
            CheckNTErrors((mySize > 0), "Illegal required memory block size!");
            CheckNTErrors((b->mem == NULL), "Incorrect memory allocation!");
            b->size = mySize;
            return AllocDynamic(myDevID, mySize);
        }
        else if(b->mem != NULL && b->size > b->used + mySize)
            return AllocDynamic(myDevID, mySize);
    }

    ShowNTErrors("Cannot find a valid memory block!");

    return NULL;
}

/* 
require a piece of memory that is not in the memory pool 
>> myDevID - device id(-1: CPU memory, >=0: GPU device ID)
>> mySize - size of the require memory
*/
void * XMem::AllocGlobal(int myDevID, MTYPE mySize)
{
    return XMemAllocOnDev(myDevID, (unsigned int)mySize);
}

/* get the available size of the memory that can be used */
MTYPE XMem::GetAvailableSize(int myDevID)
{
    return curBlock->size - curBlock->used;
}

/* 
require a piece of memory in the buffer
>> myDevID - device id(-1: CPU memory, >=0: GPU device ID)
>> mySize - size of the require memory
>> pitch - pitch for aligned memory 
<< return - the head pointer of the required memory
*/
void * XMem::AllocBuf(int myDevID, MTYPE mySize, int pitch)
{
    MTYPE backOffset = 0;

    if(pitch > 1){
        MTYPE address = (MTYPE)((char*)buf + bufUsed);
        int offset  = address % pitch;
        backOffset = offset > 0 ? pitch - offset : 0;
    }

    if((bufSize - bufUsed < mySize)){
        XPRINT1(0, stderr, "Cannot allocate the memory (%s). Please specify a larger buffer in XMem!", name);
        exit(1);
    }

    char * required = (char*)buf + bufUsed + backOffset;
    bufUsed += mySize + backOffset;

    CheckNTErrors((bufSize >= bufUsed), "Something is wrong with the memory block.");

    return required;
}

/* 
release a piece of memory 
>> p - pointer to the memory piece we intend to release
>> size - size of the memory piece to release
>> code - code the memory 
*/
void XMem::Release(void * p, MTYPE size, MTYPE code)
{
    if(code == signature)
        Release(devID, p, size);
}

/* 
release a piece of memory 
>> myDevID - device id
>> p - pointer to the memory piece we intend to release
>> size - size of the memory piece to release
*/
void XMem::Release(int myDevID, void * p, MTYPE size)
{
    if(mode == FREE_ON_THE_FLY)
        ReleaseStandard(myDevID, p, size);
}

/* 
release a piece of memory in the buffer
>> myDevID - device id(-1: CPU memory, >=0: GPU device ID)
>> mySize - size of the require memory
>> pitch - pitch for aligned memory 
*/
void XMem::ReleaseBuf(int myDevID, MTYPE mySize, int pitch)
{
    CheckNTErrors((bufUsed >= mySize), 
                  "Cannot allocate the memory. Please specify a larger buffer in XMem!");

    MTYPE backOffset = 0;

    if(pitch > 1){
        MTYPE address = (MTYPE)((char*)buf + (bufUsed - mySize));
        backOffset  = address % pitch;
    }

    bufUsed -= (mySize + backOffset);
}

/* 
free a piece of memory that is not in the memory pool 
>> myDevID - device id(-1: CPU memory, >=0: GPU device ID)
>> p - the pointer to the address of the memory we intend to free
*/
void XMem::ReleaseGlobal(int myDevID, void * p)
{
    XMemFreeOnDev(myDevID, p);
}

/* 
allocate a piece of memory as "malloc" 
>> myDevID - device id(-1: CPU memory, >=0: GPU device ID)
>> mySize - size of the require memory
>> myIsRebuiltIndex - indicates whether the index has been just rebuilt
<< return - index
*/
void * XMem::AllocStandard(int myDevID, MTYPE mySize, bool myIsRebuiltIndex)
{
    CheckNTErrors(memIndex != NULL, "The index of the memory pool is not initialized!");

    if(mySize <= minSizeIndex[0])
        mySize = minSizeIndex[0];

    int index = FindIndexEntry(mySize);
    MPieceNode * entry = NULL;
    MPieceNode * node = NULL;
    MPieceNode * hit = NULL;
    void * result = NULL;

    /* search for the memory piece avialable for the allocation */
    for(int i = index; i <= indexEntryNum; i++){
        if(i == indexEntryNum){
            entry = memIndex + index;
            CheckNTErrors(mySize >= minSizeIndex[index], "Wrong index!");
        }
        else
            entry = memIndex + i;
        
        node = entry->next;
        while(node != NULL){
            if(node->size == 0){
                MPieceNode * next = node->next;
                RemoveIndexNode(node, entry);
                node = next;
            }
            else{
                if(node->size >= mySize){
                    hit = node;
                    break;
                }
                node = node->next;
            }
        }

        if(hit != NULL)
            break;
    }
    
    /* if a free memory piece is found, we allocate the memory on it. */
    if(hit != NULL){
        MHeader * head = &hit->head;
        char * beg = (char*)GetPitchedAddress((char*)hit->p, MY_PITCH);
        char * end = (char*)beg + mySize;
        MTYPE needed = end - (char*)hit->p;
        MTYPE remaining = head->size - needed;
        
        if(remaining >= minSizeIndex[0]){

            /* make a new index node */
            MPieceNode * newNode = memIndex + nodeNumUsed++;
            newNode->head.indexNode = newNode;
            newNode->p = end;
            newNode->pReal = NULL;
            newNode->size = (char*)end + remaining -
                            (char*)GetPitchedAddress((char*)end, MY_PITCH);
            
            AddFreeIndexNode(newNode);
            
            /* connections for headers */
            MHeader &cur = hit->head;
            MHeader &next = newNode->head;
            next.pre = &cur;
            next.next = cur.next;
            cur.next = &next;
            cur.size = needed;
            
            if(next.next != NULL)
                next.next->pre = &next;
            
            next.state = 1;
            next.size = remaining;
            next.blockID = cur.blockID;
        }
        
        hit->size = mySize;
        hit->head.state = 2;
        hit->pReal = beg;
        blocks[hit->head.blockID].used += head->size;
        
        RemoveIndexNode(hit);
        AddAllocIndexNode(hit);
        
        result = beg;
    }
    else{
        /* if no free memory piece is available, we rebuild the index and merge small fragments
           to make bigger free memory pieces. */
        if(!myIsRebuiltIndex){
            RebuildIndex();
            result = AllocStandard(myDevID, mySize, true);
        }
        /* if there is still no available memory piece, we have to obtain a new block of memory. */
        else{
            int bi;
            for(bi = 0; bi < blockNum; bi++){
                XMemBlock * block = blocks + bi;
                if (block->mem != NULL && (block->head != NULL || block->size < mySize + 2 * MY_PITCH))
                    continue;
                
                if (block->mem == NULL) {
                    block->size = MAX(block->sizeDesired, mySize + 2 * MY_PITCH);
                    if (myDevID < 0) {
                        block->mem = new char[block->size];
                        memset(block->mem, 0, block->size);
                    }
                    else {
#ifdef USE_CUDA
                        int devIDBackup = -1;
                        cudaGetDevice(&devIDBackup);
                        SetDevice(myDevID);
                        cudaError_t e = cudaMalloc((void **)&block->mem, block->size);
                        if (e != cudaSuccess) {
                            ShowNTErrors("Cannot allocate the memory.");
                        }
                        CheckNTErrors(cudaMemset(block->mem, 0, block->size) == cudaSuccess, "Cannot update the memory.");
                        SetDevice(devIDBackup);
#else
                        ShowNTErrors("Please specify USE_CUDA for compiling this program.");
#endif
                    }
                }

                curBlockID = MAX(curBlockID, bi);
                    
                /* make a new index node */
                MPieceNode * newNode = memIndex + nodeNumUsed++;
                newNode->head.indexNode = newNode;
                newNode->p = block->mem;
                newNode->pReal = NULL;
                //newNode->size = (char*)block->mem + block->size -
                //                (char*)GetPitchedAddress(block->mem, MY_PITCH);
                newNode->size = mySize;
                    
                AddFreeIndexNode(newNode);
                    
                MHeader &header = newNode->head;
                header.state = 1;
                header.size = block->size;
                header.pre = NULL;
                header.next = NULL;
                header.blockID = bi;
                    
                block->head = &header;
                block->used = 0;
                    
                result = AllocStandard(myDevID, mySize, myIsRebuiltIndex);
                break;
            }
            CheckNTErrors(bi < blockNum, "No enough memory is available!");
        }
    }

    /* if all index nodes are used, we rebuild the index to release the nodes that are free */
    if(nodeNumUsed == nodeNum){
        RebuildIndex();
        CheckNTErrors(nodeNumUsed < nodeNum, "No enough index nodes for the memory pool!");
    }

    /*if(testxmemid == 30){
        recordp = result;
    }

    if(curBlockID >= 25){
        MHeader * head = blocks[25].head;
        while(head != NULL){
            fprintf(stderr, "head: %ld %ld\n", head->indexNode->pReal, head->indexNode->size);
            head = head->next;
        }
    }

    if(testxmemid == 32){
        int nnn = 0;
    }

    if(recordp != NULL){
        MTYPE size = mySize;
        if(size <= minSizeIndex[0])
            size = minSizeIndex[0];
    
        MPieceNode * entry = NULL;
        MPieceNode * node = NULL;
        MPieceNode * hit = NULL;
        MPieceNode * last = NULL;
    
        entry = memIndex + indexEntryNum + FindIndexEntry(size);
    
        last = entry;
        node = entry->next;
    
        while(node != NULL){
            CheckNTErrors(node->pre == last, "Something is wrong!");
            CheckNTErrors(last->next == node, "Something is wrong!");
            CheckNTErrors(node->head.state == 2, "Something is wrong!");
            last = node;
        
            if(node->size == 0){
                MPieceNode * next = node->next;
                RemoveFreeIndexNode(node, entry);
                node = next;
                ShowNTErrors("Something is wrong!");
            }
            else{
                CheckNTErrors(node->pReal != NULL, "Illegal pointer!");
                if(node->pReal == recordp){
                    hit = node;
                    break;
                }
                node = node->next;
            }
        }

        if(hit == NULL){
            int nnn = 0;
        }
    }*/

    return result;
}

/* 
find the highest set bit (or most significant set bit) in an integer-64 
>> mySize - required size
<< return - the position of MSB
*/
int XMem::GetMSB(MTYPE mySize)
{
    MTYPE value = mySize;

    int result = 0;
    if(value){
        if(0xFFFFFFFF00000000&value){value>>=(1<<5); result|=(1<<5);}
        if(0x00000000FFFF0000&value){value>>=(1<<4); result|=(1<<4);}
        if(0x000000000000FF00&value){value>>=(1<<3); result|=(1<<3);}
        if(0x00000000000000F0&value){value>>=(1<<2); result|=(1<<2);}
        if(0x000000000000000C&value){value>>=(1<<1); result|=(1<<1);}
        if(0x0000000000000002&value){result|=(1<<0);}
    }
    else
        result = -1;

    return result;
}

/* 
find the index entry for allocation query 
>> mySize - required size
<< return - index
*/
int XMem::FindIndexEntry(MTYPE mySize)
{
    CheckNTErrors(minSizeIndex != NULL && indexOffset >= 0, 
                 "The index of the memory pool is not initialized!");

    if(mySize <= minSizeIndex[0])
        mySize = minSizeIndex[0];

    int index = GetMSB(mySize) - indexOffset;

    if(index >= indexEntryNum)
        index = indexEntryNum - 1;

    return index;
}

/* 
remove an index node
>> node - node to remove
>> - the entry of the list that keeps the node
*/
void XMem::RemoveIndexNode(MPieceNode * node, MPieceNode * entry)
{
    MPieceNode * pre = node->pre;
    MPieceNode * next = node->next;
    

    CheckNTErrors(pre != NULL, "cannot free the entry node!");

    pre->next = next;
    if(next != NULL)
        next->pre = pre;
    
    node->pre = NULL;
    node->next = NULL;
}

/* 
add an index node for available memory pieces
>> node - node to add
>> entry - the entry of the list to append the node
*/
void XMem::AddFreeIndexNode(MPieceNode * node, MPieceNode * entry)
{
    MPieceNode * entryForMe = entry != NULL ? entry :
                              memIndex + FindIndexEntry(node->size);

    /*MPieceNode * backup = entryForMe->next;

    while(backup != NULL && backup->head.size < node->head.size){
        backup = backup->next;
        entryForMe = entryForMe->next;
    }

    entryForMe->next = node;
    node->pre = entryForMe;
    node->next = backup;
    if(backup != NULL)
        backup->pre = node;*/
    
    MPieceNode * backup = entryForMe->next;
    entryForMe->next = node;
    node->pre = entryForMe;
    node->next = backup;
    if(backup != NULL)
        backup->pre = node;

    CheckNTErrors(node != node->next, "Something wrong with the index node!");
    CheckNTErrors(node != node->pre,  "Something wrong with the index node!");
}
    
/*
remove an index node for memory pieces in use
>> node - node to remove
>> - the entry of the list that keeps the node
*/
void XMem::RemoveAllocIndexNode(MPieceNode * node, MPieceNode * entry)
{
    RemoveIndexNode(node, entry);
}

/*
add an index node for memory pieces in use
>> node - node to add
>> entry - the entry of the list to append the node
*/
void XMem::AddAllocIndexNode(MPieceNode * node, MPieceNode * entry)
{
    MPieceNode * entryForMe = entry != NULL ? entry :
                              memIndex + indexEntryNum + FindIndexEntry(node->size);
    
    MPieceNode * backup = entryForMe->next;
    entryForMe->next = node;
    node->pre = entryForMe;
    node->next = backup;
    if(backup != NULL)
        backup->pre = node;
    
    CheckNTErrors(node != node->next, "Something wrong with the index node!");
    CheckNTErrors(node != node->pre,  "Something wrong with the index node!");
}

/* 
release a piece of memory as "free" 
>> myDevID - device id(-1: CPU memory, >=0: GPU device ID)
>> p - the pointer to the address of the memory we intend to free
>> size - size of the memory piece to release
*/
void XMem::ReleaseStandard(int myDevID, void * p, MTYPE size)
{
    if(p == NULL)
        return;
    
    if(size <= minSizeIndex[0])
        size = minSizeIndex[0];
    
    MPieceNode * entry = NULL;
    MPieceNode * node = NULL;
    MPieceNode * hit = NULL;
    MPieceNode * last = NULL;
    
    entry = memIndex + indexEntryNum + FindIndexEntry(size);
    
    last = entry;
    node = entry->next;
    
    while(node != NULL){
        CheckNTErrors(node->pre == last, "Something is wrong!");
        CheckNTErrors(last->next == node, "Something is wrong!");
        CheckNTErrors(node->head.state == 2, "Something is wrong!");
        last = node;
        
        if(node->size == 0){
            MPieceNode * next = node->next;
            RemoveIndexNode(node, entry);
            node = next;
            ShowNTErrors("Something is wrong!");
        }
        else{
            CheckNTErrors(node->pReal != NULL, "Illegal pointer!");
            if(node->pReal == p){
                hit = node;
                break;
            }
            node = node->next;
        }
    }
    
    CheckNTErrors(hit != NULL, "No header is found!");
    
    hit->head.state = 1;
    
    RemoveAllocIndexNode(hit);

    MTYPE usedSize = (char*)hit->p + hit->head.size - (char*)GetPitchedAddress((char*)hit->p, MY_PITCH);
    blocks[hit->head.blockID].used -= usedSize;

    if(mergeFreeOTF){
        MHeader * head = &hit->head;
        MHeader * pre = head->pre;
        MHeader * next = head->next;
        bool mergeLeft = false;
        bool mergeRight = false;

        CheckNTErrors(head != pre, "wrong list of memory headers");
        CheckNTErrors(head != next, "wrong list of memory headers");

        if(pre != NULL && pre->state == 1 && pre->blockID == head->blockID){
            mergeLeft = true;
            head->pre = pre->pre;
            if(head->pre != NULL)
                head->pre->next = head;
            hit->p = pre->indexNode->p;
            hit->head.size += pre->size;
            RemoveAllocIndexNode(pre->indexNode);

            if(pre == blocks[head->blockID].head)
                blocks[head->blockID].head = head;
        }

        if(next != NULL && next->state == 1 && next->blockID == head->blockID){
            mergeRight = true;
            head->next = next->next;
            if(head->next != NULL)
                head->next->pre = head;
            hit->head.size += next->size;
            RemoveAllocIndexNode(next->indexNode);
        }

        if(!mergeLeft && !mergeRight){
            hit->size = usedSize;
        }
        else{
            hit->size = (char*)hit->p + hit->head.size - (char*)GetPitchedAddress((char*)hit->p, MY_PITCH);
        }
    }
    else{
        hit->size = usedSize;
    }

    AddFreeIndexNode(hit);
}

/* rebuild index to merge small fragments of memory and free the block with no use */
void XMem::RebuildIndex()
{
    int nodeNumUsed2 = indexEntryNum * 2;
    memset(memIndex2, 0, sizeof(MPieceNode) * indexEntryNum * 2);

    for(int bi = 0; bi <= curBlockID; bi++){
        XMemBlock * block = blocks + bi;
        if(block->mem == NULL || block->head == NULL)
            continue;

        MHeader * head = block->head;
        CheckNTErrors(head->size <= block->size, "Illegal memory block!");
        
        block->head = NULL;
        block->used = 0;

        /* if the block is not used, we delete it */
        if(head->state == 1 && head->size == block->size){
            if(devID < 0){
                delete[] (char*)block->mem;
            }
            else{
#ifdef USE_CUDA
                int devIDBackup = -1;
                cudaGetDevice(&devIDBackup);
                SetDevice(devID);
                CheckNTErrors(cudaFree((char*)block->mem) == cudaSuccess, "Cannot free the memory.");
                SetDevice(devIDBackup);
#else
                ShowNTErrors("Please specify USE_CUDA for compiling this program.");
#endif 
            }

            block->size = 0;
            block->mem = NULL;
        }
        else{
            /* if the block is in use, we build the index */
            int pieceCount = 0;
            MTYPE size = 0;
            MHeader * newLast = NULL;
            while(head != NULL){
                MHeader * next = head->next;
                if(head->state == 1){
                    while(next != NULL && next->state == 1){
                        head->size += next->size;
                        next = next->next;
                    }
                    head->next = next;
                }
                
                MPieceNode * node = head->indexNode;
                void * p = node->p;
                
                /* make a new index node */
                MPieceNode * newNode = memIndex2 + nodeNumUsed2++;
                newNode->p = p;
                
                if(head->state == 1){
                    newNode->size = (char*)p + head->size -
                                    (head->state == 1 ? (char*)GetPitchedAddress((char*)p, MY_PITCH) : (char*)head->indexNode->pReal);
                }
                else
                    newNode->size = node->size;
                
                newNode->pre = NULL;
                newNode->next = NULL;
                
                CheckNTErrors(newNode->size > 0, "Illegal index node!");
                
                MHeader * newHeader = &newNode->head;
                
                newHeader->indexNode = newNode;
                newHeader->pre = newLast;
                newHeader->next = NULL;
                newHeader->blockID = bi;
                newHeader->size = head->size;
                newHeader->state = head->state;
                
                if(newLast != NULL)
                    newLast->next = newHeader;
                
                if(head->state == 1){
                    newNode->pReal = NULL;
                    MPieceNode * entry = memIndex2 + FindIndexEntry(newNode->size);
                    AddFreeIndexNode(newNode, entry);
                }
                else{
                    newNode->pReal = head->indexNode->pReal;
                    MPieceNode * entry = memIndex2 + indexEntryNum + FindIndexEntry(newNode->size);
                    AddAllocIndexNode(newNode, entry);
                    block->used += head->size;
                }
                
                if(newLast == NULL)
                    block->head = newHeader;
                
                pieceCount++;
                size += head->size;
                CheckNTErrors(size <= block->size, "Illegal block size!");
                
                newLast = newHeader;
                head = next;
            }
        }
    }
    
    MPieceNode * backup = memIndex2;
    memIndex2 = memIndex;
    memIndex = backup;    
    nodeNumUsed = nodeNumUsed2;
}

/* 
reset the memory pool  
>> myDevID - device id(-1: CPU memory, >=0: GPU device ID)
*/
void XMem::Reset(int myDevID)
{
    for(int i = 0; i <= curBlockID; i++){
        if(devID >= 0){
#ifdef USE_CUDA
            CheckNTErrors(cudaFree(blocks[i].mem) == cudaSuccess, "Cannot free the memory.");
#else
            ShowNTErrors("We need cuda code here!");
#endif
        }
        else
            delete[] (char*)blocks[i].mem;

        blocks[i].mem = NULL;
        blocks[i].used = 0;
        blocks[i].size = blocks[i].sizeDesired;
    }

    curBlockID = 0;
    curBlock = blocks;
    curBlock->used = 0;
    finalBlockID = 0;
    bufUsed = 0;
}

/* 
get pitch for aligned memory
>> baseAddress - where the allocated memory starts
>> mySize - size of the require memory
<< return - the actual size required for aligned memory
*/
MTYPE XMem::GetPitch(int myDevID, MTYPE baseAddress, MTYPE mySize)
{
    long long address = baseAddress + mySize;
    int offset  = address % CUDA_PITCH;
    int backOffset = offset > 0 ? CUDA_PITCH - offset : 0;
    return mySize + backOffset;
}

/* 
get pitched address for aligned memory 
>> address - the starting address
>> pitch - as it is
<< return - pitched address
*/
void * XMem::GetPitchedAddress(void * address, MTYPE pitch)
{
    MTYPE p = (MTYPE)address;
    MTYPE offset  = p % pitch;
    MTYPE backOffset = offset > 0 ? pitch - offset : 0;
    return (char*)address + backOffset;
}

/* get current address (for use) */
void * XMem::GetAddress()
{
    if(curBlock->mem == NULL)
        Alloc(devID, 0);

    return curBlock->mem;
}

/* clear it */
void XMem::Clear()
{
    if (mode == UNI_FREE) {
        for (int i = 0; i < blockNum; i++)
            blocks[i].used = 0;
        curBlock = blocks;
        curBlockID = 0;
    }
    else if (mode == FREE_ON_THE_FLY) {
        nodeNumUsed = indexEntryNum * 2;
        memset(memIndex, 0, sizeof(MPieceNode) * indexEntryNum * 2);
        for (int i = 0; i <= curBlockID; i++) {
            blocks[i].head = NULL;
            blocks[i].used = 0;
            if (i > 0) {
                blocks[i].size = blocks[i].sizeDesired;
                Free(devID, blocks[i].mem);
                blocks[i].mem = NULL;
            }
        }
        curBlock = blocks;
        curBlockID = 0;
    }
    else {
        ShowNTErrors("Something is wrong!");
    }

    signature++;
}

/* clear the buffer */
void XMem::ClearBuf()
{
    bufUsed = 0;
}

/* clear the memory pool and the buffer */
void XMem::ClearAll()
{
    Clear();
    ClearBuf();
}

/* 
set a variable to the input value
>> tgt - where we put the value
>> src - where the value is from
>> size - data size, e.g., for a float, it is sizeof(float)
>> tgtMem - the memory pool used by the target variable
>> srcMem - the memory pool used by the source variable
>>
*/
void XMem::Copy(void * tgt, void * src, int size, XMem * tgtMem, XMem * srcMem)
{
    if(srcMem == NULL || srcMem->devID < 0){
        if(tgtMem == NULL || tgtMem->devID < 0)  // host (CPU memory)  -> host (CPU memory)
            memcpy(tgt, src, size);
#ifdef USE_CUDA
        else                                     // device (GPU memory) -> host (CPU memory)
            cudaMemcpyFromSymbol(tgt, src, size);
#endif
    }
#ifdef USE_CUDA
    else{
        if(tgtMem == NULL || tgtMem->devID < 0)  // host (CPU memory)  -> device (GPU memory)
            cudaMemcpyToSymbol(tgt, src, size);
        else                                     // device (GPU memory) -> device (GPU memory)
            cudaMemcpy(tgt, src, size, cudaMemcpyDeviceToDevice);
    }
#endif
}

/* 
set a float-typed variable to the input value 
>> tgt - where we put the value
>> src - where the value is from
>> tgtMem - the memory pool used by the target variable
>> srcMem - the memory pool used by the source variable
*/
void XMem::CopyFloat(float * tgt, float * src, XMem * tgtMem, XMem * srcMem)
{
    XMem::Copy(tgt, src, sizeof(float), tgtMem, srcMem);
}

/* 
set a variable to 0 
>> tgt - where the variable is placed
>> size - data size
>> tgtMem - the memory pool used by the variable
*/
void XMem::SetZero(void * tgt, MTYPE size, XMem * tgtMem)
{
    if(tgtMem == 0 || tgtMem->devID < 0)
        memset(tgt, 0, (unsigned int)size);
#ifdef USE_CUDA
    else
        cudaMemset(tgt, 0, size);
#endif
}

/* record the pin point */
void XMem::SetPin()
{
    CheckNTErrors((finalBlockID == curBlockID), "Cannot set pin for the memory pool. Please used a larger size of the first block!");

    curBlockPin = curBlockID;
    curUsedPin = curBlock->used;
}

/* go back to the pin point */
void XMem::BackToPin()
{
    if(curBlockPin < 0)
        return;

    for(int i = curBlockPin + 1; i <= finalBlockID; i++){

        if(devID >= 0){
#ifdef USE_CUDA
            CheckNTErrors(cudaFree(blocks[i].mem) == cudaSuccess, "Cannot free the memory.");
#else
            ShowNTErrors("We need cuda code here!");
#endif
        }
        else
            delete[] (char*)blocks[i].mem;

        blocks[i].mem = NULL;
        blocks[i].used = 0;
        blocks[i].size = blocks[i].sizeDesired;
    }

    curBlockID = curBlockPin;
    curBlock = blocks + curBlockID;
    curBlock->used = curUsedPin;
    finalBlockID = curBlockID;
}

/* record the pin point for buffer */
void XMem::SetPinBuf()
{
    bufUsedPin = bufUsed;
}

/* go back to the pin point */
void XMem::BackToPinBuf()
{
    bufUsed = bufUsedPin;
}

/* transform a size into a number (in million) */
MTYPE XMem::GetMemSize(const char * size)
{
    char * s = new char[strlen(size) + 1];
    strcpy(s, size);

    ToLowercase(s);

    int len = (int)strlen(s);

    bool ok = false;
    float num = 0;

    if(s[len-2] == 'm' && s[len-1] == 'b'){
        s[len-2] = 0;
        num = (float)atof(s);
        ok = true;
    }
    if(s[len-2] == 'g' && s[len-1] == 'b'){
        s[len-2] = 0;
        num = (float)atof(s);
        num *= 1000.0F;
        ok = true;
    }

    delete[] s;

    if(ok)
        return (MTYPE)num;
    else
        return 0;
}

/* transform a size into a number (in Bytes) */
MTYPE XMem::GetMemSizeInBytes(const char * size)
{
    char * s = new char[strlen(size) + 1];
    strcpy(s, size);

    ToLowercase(s);

    int len = (int)strlen(s);

    bool ok = false;
    float num = 0;

    if(s[len-2] == 'm' && s[len-1] == 'b'){
        num = (float)GetMemSize(size) * 1000000;
        ok = true;
    }
    else if(s[len-2] == 'g' && s[len-1] == 'b'){
        num = (float)GetMemSize(size) * 1000000;
        ok = true;
    }
    else{
        num = (float)atof(s);
        ok = true;
    }

    delete[] s;

    if(ok)
        return (MTYPE)num;
    else
        return 0;
}

/* create a new cublas handle */
void XMem::CreateBLASHandle()
{
#ifdef USE_CUDA
    if(cublasHandle != NULL){
        CheckNTErrors(cublasDestroy(cublasHandle) == CUBLAS_STATUS_SUCCESS, 
                      "Cannot destroy the cublas handle.");
    }

    CheckNTErrors((enum curandStatus)cublasCreate(&cublasHandle) == CURAND_STATUS_SUCCESS, 
                  "Cannot create the cublas handle.");
#endif
}

/* show profile of the memory pool */
void XMem::ShowMemUsage(FILE * file)
{
    MTYPE used = 0;
    MTYPE total = 0;

    for(int i = 0; i < blockNum; i++){
        if(blocks[i].mem != NULL){
            used  += blocks[i].used;
            total += blocks[i].size;
        }
    }

    fprintf(file, "mem:%.1fMB used:%.1fMB usage:%.3f\n", 
           (DTYPE)total/MILLION, (DTYPE)used/MILLION, (DTYPE)used/total);
}

#ifdef USE_CUDA

/* get the handle of cublas */
cublasHandle_t * XMem::GetCublasHandle()
{
    return &cublasHandle;
}

#endif

/* constructor */
XMemManager::XMemManager()
{
    Initialize();
}

/* de-constructor */
XMemManager::~XMemManager()
{
}

/* get memory size */
MTYPE XMemManager::GetAvailableMemory()
{
    unsigned long freeMem = 0;
#if __APPLE__
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    unsigned int namelen = sizeof(mib) / sizeof(mib[0]);
    unsigned long long size;
    size_t len = sizeof(size);
    if (sysctl(mib, namelen, &size, &len, NULL, 0) < 0){
        ShowNTErrors("Cannot get memory size on Mac!");
    }
    else{
        return size;
    }
#elif _WIN32
    MEMORYSTATUSEX memoryStatus;
    memoryStatus.dwLength = sizeof(memoryStatus);
    if (GlobalMemoryStatusEx(&memoryStatus)){
        freeMem = memoryStatus.ullAvailPhys;
    }
#else
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    freeMem = pages * page_size;
#endif
    return (MTYPE)freeMem;
}

/* get GPU memory size */
MTYPE XMemManager::GetAvailableGPUMemory(int devID)
{
    size_t freeMem = 0;
    
#ifdef USE_CUDA
    size_t totalMem = 0;
    cudaSetDevice(devID);
    if (cudaMemGetInfo(&freeMem, &totalMem) != cudaSuccess){
        XPRINT(0, stderr, "cannot get GPU memory information.");
        exit(1);
    }
#endif
    return (MTYPE)freeMem;
}

/* get buffer size */
void XMemManager::GetBufferSize(MTYPE freeMem, MTYPE * myBufSize)
{
    *myBufSize = 0;
    if (freeMem >= MILLION * 128){
        *myBufSize = MILLION * 32;
        if (freeMem >= MILLION * 256){
            *myBufSize = MILLION * 64;
            if (freeMem >= MILLION * 512){
                *myBufSize = MILLION * 128;
                if (freeMem >= MILLION * 1024) {
                    *myBufSize = MILLION * 128;
                    if (freeMem >= MILLION * 2048)
                        *myBufSize = MILLION * 128;
                }
            }
        }
    }
} 

/* initialize it and set the global memory information */
void XMemManager::Initialize()
{
    srand((unsigned int)time(NULL));

    Free();
    
    /* CPUs (we actually do not care about how many CPUs are using) */
    nCPUMem = 1;

    /* GPUs */
    nGPUMem = 0;

#ifdef USE_CUDA
    if (cudaGetDeviceCount(&nGPUMem) != cudaSuccess) {
        XPRINT(0, stderr, "cannot get GPU information.");
        exit(1);
    }
#endif

}

/* free it */
void XMemManager::Free()
{
    for (int i = 0; i < MAX_CPU_MEM_NUM; i++)
        CPUMems[i].Free();
    for (int i = 0; i < MAX_GPU_MEM_NUM; i++)
        GPUMems[i].Free();
}

/* get global memory pool */
XMem * XMemManager::GetMem(const int devID)
{
    XMem * mem = NULL;
    if (devID < 0){
        if(!CPUMems[0].isInitialized){
            MTYPE freeMem = GetAvailableMemory();
            MTYPE myBufSize = 0;
            GetBufferSize(freeMem, &myBufSize);
            CPUMems[0].Initialize(-1, FREE_ON_THE_FLY, 
                                  MIN_BLOCK_SIZE_FOR_MEMPOOL, 
                                  MIN_BLOCK_NUM_FOR_MEMPOOL, 
                                  myBufSize);
        }
        mem = CPUMems;
    }
    else{
        if (devID < nGPUMem){
            if(!GPUMems[devID].isInitialized){
                MTYPE freeMem = GetAvailableGPUMemory(devID);
                MTYPE myBufSize = 0;
                GetBufferSize(freeMem, &myBufSize);
                GPUMems[devID].Initialize(devID, FREE_ON_THE_FLY, 
                                          MIN_BLOCK_SIZE_FOR_MEMPOOL, 
                                          MIN_BLOCK_NUM_FOR_MEMPOOL, 
                                          myBufSize);
            }
            mem = GPUMems + devID;
        }
        else{
            XPRINT1(0, stderr, "Cannot get the memory (%d). Please check your device id!", devID);
        }
    }
    
    return mem;
}

/* get global memory size */
int XMemManager::GetMemSize(const int devID, MTYPE * myBlockSize, int * myBlockNum, MTYPE * myBufSize)
{
    XMem * mem = GetMem(devID);
    int result = 0;
    if (mem != NULL){
        *myBlockSize = mem->maxBlockSize;
        *myBlockNum = mem->blockNum;
        *myBufSize = mem->bufSize;
        result = 1;
    }
    return result;
}

/* show memory information */
void XMemManager::ShowMemInfo()
{
    XPRINT(1, stderr, "Memory Information:\n");
    MTYPE myBlockSize, myBufSize;
    int myBlockNum;
    for(int i = 0; i < nCPUMem; i++){
        GetMemSize(-1, &myBlockSize, &myBlockNum, &myBufSize);
        XPRINT3(1, stderr, " - id:-1 CPU, blockSize:%lld, blockNum:%d, bufSize:%lld\n", myBlockSize, myBlockNum, myBufSize);
    }

    for(int i = 0; i < nGPUMem; i++){
        GetMemSize(i, &myBlockSize, &myBlockNum, &myBufSize);
        XPRINT4(1, stderr, " - id:%2d GPU, blockSize:%lld, blockNum:%d, bufSize:%lld\n", i, myBlockSize, myBlockNum, myBufSize);
    }
}

} /* end of the nts (NiuTrans.Tensor) namespace */
