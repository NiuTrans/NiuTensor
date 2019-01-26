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
 * some public functions are defined here
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-04-27
 *
 */

#include "XUtility.h"

#if !defined( WIN32 ) && !defined( _WIN32 )
    #include "sys/time.h"
    #include "time.h"
    #include "iconv.h"
#else
    #include "time.h"
    #include "windows.h"
    #include "process.h"
#endif

namespace nts{ // namespace nts(NiuTrans.Tensor)

/* 
get first digit number
for 1.23 it returns 1.0
for 0.21 it returns 0.1
for -930.00 it returns 100
*/
DTYPE GetFirstDigitNum(DTYPE p)
{
    if(p == 0)
        return 0;
    else if(p < 0)
        p = -p;

    DTYPE t = 1.0;
    DTYPE backup = p;

    if(p == 1.0F)
        return 1.0F;
    else if(p > 1.0F){
        while(p > 1.0F){
            p /= 10;
            t *= 10;
        }
        t /= 10;   
    }
    else{
        while(p < 1.0F){
            p *= 10;
            t /= 10;
        }
    }

    int x = (int)(backup/t);
    if(x < 1)
        x = 1;

    return t * x;
}

bool IsFloatValid(float f)
{
    int * a = (int*)&f;
    return((*a)&0x7f800000)!=0x7f800000;
}

bool IsNAN(float f)
{
    return (f != f);
}

bool IsNAN(double f)
{
    return (f != f);
}

bool IsINF(float f)
{
    return !IsNAN(f) && IsNAN(f - f);
}

bool IsINF(double f)
{
    return !IsNAN(f) && IsNAN(f - f);
}

void ToLowercase(char * str)
{
    int len = (int)strlen(str);
    for(int i = 0; i < len; i++){
        if(str[i] >= 'A' && str[i] <= 'Z')
            str[i] -= ('A' - 'a');
    }
}

char * GetNextWord(char * p)
{
    if(p == NULL)
        return p;

    while(*p == ' ' || *p == '\t')
        p++;

    if(*p == '\r' || *p == '\n' || *p == '\0')
        return NULL;

    return p;
}

void XMemSet(void * p, int value, size_t size)
{
#ifdef USE_CUDA
    cudaMemset(p, value, size);
#else
    memset(p, value, size);
#endif
}

void XMemSet(int devID, void * p, int value, size_t size)
{
    if(devID >= 0){
#ifdef USE_CUDA
        int devIDBackup = 0;
        cudaGetDevice(&devIDBackup);
        cudaSetDevice(devID);
        cudaMemset(p, value, size);
        cudaSetDevice(devIDBackup);
#else
        ShowNTErrors("Please specify USE_CUDA and recompile the code!");
#endif
    }
    else
        memset(p, value, size);

    
}

#ifdef USE_CUDA
cudaMemcpyKind GetMemcpyKind(int devIDFrom, int devIDTo)
{
    if(devIDFrom < 0 && devIDTo < 0)
        return cudaMemcpyHostToHost;
    else if(devIDFrom < 0 && devIDTo >= 0)
        return cudaMemcpyHostToDevice;
    else if(devIDFrom >= 0 && devIDTo < 0)
        return cudaMemcpyDeviceToHost;
    else
        return cudaMemcpyDeviceToDevice;
}
#endif

void XMemCopy(void * t, int devIDT, const void * s, int devIDS, size_t size)
{
    if(t == s)
        return;

    if(devIDT < 0 && devIDS < 0){
        memcpy(t, s, size);
        return;
    }
#ifdef USE_CUDA
    else{
        int devID = devIDT < 0 ? devIDS : devIDT;
        int devIDBackup = 0;
        cudaGetDevice(&devIDBackup);
        cudaSetDevice(devID);

        if(devIDT >= 0 && devIDS < 0){
            cudaError_t error = cudaMemcpy(t, s, size, cudaMemcpyHostToDevice);
            if(error != cudaSuccess){
                ShowNTErrors("cudaMemcpy error (cudaMemcpyHostToDevice)");
            }
        }
        else if(devIDT < 0 && devIDS >= 0){
            cudaError_t error = cudaMemcpy(t, s, size, cudaMemcpyDeviceToHost);
            if(error != cudaSuccess){
                ShowNTErrors("cudaMemcpy error (cudaMemcpyDeviceToHost)");
            }
        }
        else{
            //if(devIDT == devIDS){
                cudaError_t error = cudaMemcpy(t, s, size, cudaMemcpyDeviceToDevice);
                if(error != cudaSuccess){
                    ShowNTErrors("cudaMemcpy error (cudaMemcpyDeviceToDevice)");
                }
            /*}
            else{
                CheckNTErrors((cudaMemcpyPeer(t, devIDT, s, devIDS, size) == cudaSuccess),
                                    "cudaMemcpy error (cudaMemcpyDeviceToDevice)");
            }*/
        }

        cudaSetDevice(devIDBackup);
    }
#else
    ShowNTErrors("Please specify USE_CUDA and recompile the code!");
#endif

}

#ifdef USE_CUDA
void XMemCopyAsync(void * t, int devIDT, const void * s, int devIDS, size_t size, cudaStream_t stream, int streamDevID)
{
    if(t == s)
        return;

    int devIDBackup = -1;
    if(streamDevID >= 0 && (devIDT >= 0 || devIDS >= 0)){
        CheckNTErrors((cudaGetDevice(&devIDBackup) == cudaSuccess), "Cannot get GPU device id!");
        if(streamDevID != devIDBackup)
            CheckNTErrors((cudaSetDevice(streamDevID) == cudaSuccess), "Cannot set GPU device!");
    }

    if(devIDT < 0 && devIDS < 0){
        memcpy(t, s, size);
        return;
    }
    else if(devIDT >= 0 && devIDS < 0){
        cudaError_t error = cudaMemcpyAsync(t, s, size, cudaMemcpyHostToDevice, stream);
        if(error != cudaSuccess){
            ShowNTErrors("cudaMemcpyAsync error (cudaMemcpyHostToDevice)");
        }
    }
    else if(devIDT < 0 && devIDS >= 0){
        cudaError_t error = cudaMemcpyAsync(t, s, size, cudaMemcpyDeviceToHost, stream);
        if(error != cudaSuccess){
            ShowNTErrors("cudaMemcpyAsync error (cudaMemcpyDeviceToHost)");
        }
    }
    else{
        //if(devIDT == devIDS){
            cudaError_t error = cudaMemcpyAsync(t, s, size, cudaMemcpyDeviceToDevice, stream);
            if(error != cudaSuccess){
                ShowNTErrors("cudaMemcpyAsync error (cudaMemcpyDeviceToDevice)");
            }
        //}
        /*else{
            CheckNTErrors((cudaMemcpyPeerAsync(t, devIDT, s, devIDS, size, stream) == cudaSuccess),
                                "cudaMemcpyAsync error (cudaMemcpyDeviceToDevice)");
        }*/
    }

    if(streamDevID >= 0 && (devIDT >= 0 || devIDS >= 0)){
        if(streamDevID != devIDBackup)
            CheckNTErrors((cudaSetDevice(devIDBackup) == cudaSuccess), "Cannot set GPU device!");
    }
}
#else
void XMemCopyAsync(void * t, int devIDT, void * s, int devIDS, size_t size, void * stream, int streamDevID)
{
    XMemCopy(t, devIDT, s, devIDS, size);
}
#endif

void XMemCopy2D(void * t, size_t tPitch, int devIDT, const void * s, size_t sPitch, int devIDS, size_t mSize, int n)
{
    if (t == s)
        return;

    if (devIDT < 0 && devIDS < 0) {
        for(int i = 0; i < n; i++)
            memcpy((char*)t + tPitch * i, (char*)s + sPitch * i, mSize);
        return;
    }
#ifdef USE_CUDA
    else{
        int devID = devIDT < 0 ? devIDS : devIDT;
        int devIDBackup = 0;
        cudaGetDevice(&devIDBackup);
        cudaSetDevice(devID);

        if (devIDT >= 0 && devIDS < 0) {
            cudaError_t error = cudaMemcpy2D(t, tPitch, s, sPitch, mSize, n, cudaMemcpyHostToDevice);
            if(error != cudaSuccess){
                ShowNTErrors("cudaMemcpy2D error (cudaMemcpyHostToDevice)");
            }
        }
        else if (devIDT < 0 && devIDS >= 0) {
            cudaError_t error = cudaMemcpy2D(t, tPitch, s, sPitch, mSize, n, cudaMemcpyDeviceToHost);
            if(error != cudaSuccess){
                ShowNTErrors("cudaMemcpy error (cudaMemcpyDeviceToHost)");
            }
        }
        else {
            cudaError_t error = cudaMemcpy2D(t, tPitch, s, sPitch, mSize, n, cudaMemcpyDeviceToDevice);
            if (error != cudaSuccess) {
                ShowNTErrors("cudaMemcpy error (cudaMemcpyDeviceToDevice)");
            }
        }

        cudaSetDevice(devIDBackup);
    }
#else
    ShowNTErrors("Please specify USE_CUDA and recompile the code!");
#endif
}

void XMemCopy2DAsync(void * t, size_t tPitch, int devIDT, const void * s, size_t sPitch, int devIDS, size_t mSize, int n, XStream * stream)
{
    if (t == s)
        return;

    if (devIDT < 0 && devIDS < 0) {
        for(int i = 0; i < n; i++)
            memcpy((char*)t + tPitch * i, (char*)s + sPitch * i, mSize);
        return;
    }
#ifdef USE_CUDA
    else{
        CheckNTErrors(stream != NULL, "No stream found!");
        cudaStream_t &cstream = stream->stream;
        if (devIDT >= 0 && devIDS < 0) {
            cudaError_t error = cudaMemcpy2DAsync(t, tPitch, s, sPitch, mSize, n, cudaMemcpyHostToDevice, cstream);
            if(error != cudaSuccess){
                ShowNTErrors("cudaMemcpy2D error (cudaMemcpyHostToDevice)");
            }
        }
        else if (devIDT < 0 && devIDS >= 0) {
            cudaError_t error = cudaMemcpy2DAsync(t, tPitch, s, sPitch, mSize, n, cudaMemcpyDeviceToHost, cstream);
            if(error != cudaSuccess){
                ShowNTErrors("cudaMemcpy error (cudaMemcpyDeviceToHost)");
            }
        }
        else {
            cudaError_t error = cudaMemcpy2DAsync(t, tPitch, s, sPitch, mSize, n, cudaMemcpyDeviceToDevice, cstream);
            if (error != cudaSuccess) {
                ShowNTErrors("cudaMemcpy error (cudaMemcpyDeviceToDevice)");
            }
        }
    }
#else
    ShowNTErrors("Please specify USE_CUDA and recompile the code!");
#endif
}

void * XMemAlloc(int devID, size_t size)
{
    void * p = NULL;

    if(devID < 0){
        p = new char[size];
        return p;
    }
    else{
#ifdef USE_CUDA
        int devIDBackup = 0;
        cudaGetDevice(&devIDBackup);
        cudaSetDevice(devID);

        cudaError_t e = cudaMalloc((void **)&p, size);

        if(e != cudaSuccess){
            ShowNTErrors("Cannot allocate the memory in XMemAlloc.");
        }

        cudaSetDevice(devIDBackup);

        return p;
#else
        ShowNTErrors("Please specify USE_CUDA and recompile the code!");
        return NULL;
#endif
    }
}

void * XMemAllocOnDev(int devID, size_t size)
{
    void * p = NULL;

    if(devID < 0){
        p = new char[size];
        return p;
    }
    else{
#ifdef USE_CUDA
        cudaError_t e = cudaMalloc((void **)&p, size);

        if(e != cudaSuccess){
            ShowNTErrors("Cannot allocate the memory in XMemAlloc.");
        }

        return p;
#else
        ShowNTErrors("Please specify USE_CUDA and recompile the code!");
        return NULL;
#endif
    }
}

void XMemFree(int devID, void * p)
{
    if(p == NULL)
        return;

    if(devID < 0){
        delete[] (char*)p;
        return;
    }

#ifdef USE_CUDA
    int devIDBackup = 0;
    cudaGetDevice(&devIDBackup);
    cudaSetDevice(devID);

    cudaError_t e = cudaFree((char*)p);

    if(e != cudaSuccess){
        ShowNTErrors("Cannot free the memory in XMemAlloc.");
    }

    cudaSetDevice(devIDBackup);
#else
    ShowNTErrors("Please specify USE_CUDA and recompile the code!");
#endif
}

void XMemFreeOnDev(int devID, void * p)
{
    if(devID < 0){
        delete[] (char*)p;
        return;
    }

#ifdef USE_CUDA
    cudaError_t e = cudaFree((char*)p);

    if(e != cudaSuccess){
        ShowNTErrors("Cannot free the memory in XMemAlloc.");
    }
#else
    ShowNTErrors("Please specify USE_CUDA and recompile the code!");
#endif
}

/* return the CPU value that is pointed by pointer*/
DTYPE ToCPU(int devID, void * value)
{
    CheckNTErrors(value != NULL, "Empty pointer");

    if(devID < 0)
        return *((DTYPE*)value);
    else{
        DTYPE vCPU;
        XMemCopy(&vCPU, -1, (DTYPE*)value, devID, sizeof(DTYPE));
        return vCPU;
    }
}

/* return the CPU int that is pointed by pointer*/
int ToCPUInt(int devID, void * value)
{
    CheckNTErrors(value != NULL, "Empty pointer");

    if(devID < 0)
        return *((int*)value);
    else{
        int vCPU;
        XMemCopy(&vCPU, -1, (int*)value, devID, sizeof(int));
        return vCPU;
    }
}

/* assign a number to a variable that is kept on a specified device */
bool SetToDevice(int devID, void * p, DTYPE value)
{
    if(p == NULL)
        return false;

    if(devID < 0)
        *(DTYPE*)p = value;
    else{
        XMemCopy(p, devID, &value, -1, sizeof(DTYPE));
    }

    return true;
}

/* assign a integer number to a variable that is kept on a specified device */
bool SetToDeviceInt(int devID, void * p, int value)
{
    if(p == NULL)
        return false;

    if(devID < 0)
        *(int*)p = value;
    else{
        XMemCopy(p, devID, &value, -1, sizeof(int));
    }

    return true;
}

/* get the next number with power of 2 */
unsigned int GetNextPower2(unsigned int n)
{
    unsigned int p = 1;

    if (n && !(n & (n - 1)))
        return n;
 
    while (p < n) 
        p <<= 1;
     
    return p;
}

/* sleep for a while */
void XSleep(int sleepTime)
{
#ifdef  _WIN32
    Sleep((DWORD)sleepTime);
#else
    sleep(sleepTime/1000);
#endif
}

/* get current clock (in ms) */
double GetClock()
{
#ifndef WIN32
    timeval startT;
    gettimeofday (&startT, NULL);
    return (((double)(startT.tv_sec) * 1000000 + (double)(startT.tv_usec))/1000000) * 1000;
#else
    return clock()*1000/CLOCKS_PER_SEC;
#endif
}

/* get current clock (in s) */
double GetClockSec()
{
#ifndef WIN32
    timeval startT;
    gettimeofday(&startT, NULL);
    return ((double)(startT.tv_sec) * 1000000 + (double)(startT.tv_usec))/1000000;
#else
    return clock() / CLOCKS_PER_SEC;
#endif
}

void XShortSort(char * lo, char * hi, int * indexlo, int * indexhi, int width, int stride, int (*comp)(const void *, const void *));
void XSwapForSort(char * a, char * b, int * indexA, int * indexB, int width);

#define REC_SORT_SIZE (8*sizeof(void*) - 2)
#define MIN_QSORT_NUM 8

/* 
quick sorting
>> data - data array to sort
>> index - index of the items
>> num - number of the items that we intend to sort
>> width - width of an item
>> stride - number of the items that we need to go over when we move to the next item
            NOTE: this means that the items may not placed in a continuous memory space
>> comp - the comparison function 
*/
void XQSort(void * data, void * index, int num, int width, int stride, int (*comp)(const void *, const void *))
{
    char *lo, *hi;         // ends of sub-array currently sorting
    int *indexlo, *indexhi;
    char *mid;             // points to middle of subarray
    int *indexmid;
    char *loguy, *higuy;  // traveling pointers for partition step
    int *indexloguy, *indexhiguy;
    int size;             // size of the sub-array
    char *loStack[REC_SORT_SIZE], *hiStack[REC_SORT_SIZE];
    int *indexloStack[REC_SORT_SIZE], *indexhiStack[REC_SORT_SIZE];
    int stackptr;         // stack for saving sub-arraies to be processed

    int realStride = stride * width;

    if(num < 2 || width == 0)
        return;

    stackptr = 0;

    lo = (char*)data;
    hi = (char*)data + realStride * (num - 1);
    indexlo = (int*)index;
    indexhi = index != NULL ? (int*)index + stride * (num - 1) : NULL;

recurse:

    /* number of items to sort */
    size = (int)(hi - lo)/realStride + 1;

    if(size <= MIN_QSORT_NUM)
        XShortSort(lo, hi, indexlo, indexhi, width, stride, comp);
    else {
        mid = lo + (size/2) * realStride;
        indexmid = indexlo + (size/2) * stride;
        
        /* sort the first, last and middle elements into order */
        if(comp(lo, mid) > 0)
            XSwapForSort(lo, mid, indexlo, indexmid, width);
        if(comp(lo, hi) > 0)
            XSwapForSort(lo, hi, indexlo, indexhi, width);
        if(comp(mid, hi) > 0)
            XSwapForSort(mid, hi, indexmid, indexhi, width);
        
        /* traveling pointers for partition step */
        loguy = lo;
        higuy = hi;
        indexloguy = indexlo;
        indexhiguy = indexhi;
        
        for(;;){
            if(index == NULL){
                if(mid > loguy){
                    do{
                        loguy += realStride;
                    }while(loguy < mid && comp(loguy, mid) <= 0);
                }
                if(mid <= loguy){
                    do{
                        loguy += realStride;
                    }while(loguy <= hi && comp(loguy, mid) <= 0);
                }
                do{
                    higuy -= realStride;
                }while(higuy > mid && comp(higuy, mid) > 0);
            }
            else{
                if(mid > loguy){
                    do{
                        loguy += realStride;
                        indexloguy += stride;
                    }while(loguy < mid && comp(loguy, mid) <= 0);
                }
                if(mid <= loguy){
                    do{
                        loguy += realStride;
                        indexloguy += stride;
                    }while(loguy <= hi && comp(loguy, mid) <= 0);
                }
                do{
                    higuy -= realStride;
                    indexhiguy -= stride;
                }while(higuy > mid && comp(higuy, mid) > 0);
            }
        
            if(higuy < loguy)
                break;
        
            XSwapForSort(loguy, higuy, indexloguy, indexhiguy, width);
        
            if(mid == higuy) {
                mid = loguy;
                indexmid = indexloguy;
            }
            /* find adjacent elements equal to the partition element */ 
            higuy += realStride;
            indexhiguy += stride;
        }
        
        if(index == NULL){
            if (mid < higuy){
                do{
                    higuy -= realStride;
                }while(higuy > mid && comp(higuy, mid) == 0);
            }  
            if(mid >= higuy){
                do{
                    higuy -= realStride;
                }while(higuy > lo && comp(higuy, mid) == 0);
            }
        }
        else{
            if (mid < higuy){
                do{
                    higuy -= realStride;
                    indexhiguy -= stride;
                }while(higuy > mid && comp(higuy, mid) == 0);
            }  
            if(mid >= higuy){
                do{
                    higuy -= realStride;
                    indexhiguy -= stride;
                }while(higuy > lo && comp(higuy, mid) == 0);
            }
        }
        
        /* the partition is finished. We sort the subarrays [lo, higuy] and [loguy, hi] now */
        if(higuy - lo >= hi - loguy){
            if(lo < higuy){
                loStack[stackptr] = lo;
                hiStack[stackptr] = higuy;
                indexloStack[stackptr] = indexlo;
                indexhiStack[stackptr] = indexhiguy;
                ++stackptr;
            }
            if(loguy < hi){
                lo = loguy;
                indexlo = indexloguy;
                goto recurse;  // small recursion
            }
        }
        else{
            if(loguy < hi){
                loStack[stackptr] = loguy;
                hiStack[stackptr] = hi;
                indexloStack[stackptr] = indexloguy;
                indexhiStack[stackptr] = indexhi;
                ++stackptr;
            }  
            if(lo < higuy){
                hi = higuy;
                indexhi = indexhiguy;
                goto recurse;  // small recursion
            }
        }  
    }

    /* we have done it. Check if there are any, and do them. */  
    --stackptr;
    if(stackptr >= 0){
        lo = loStack[stackptr];
        hi = hiStack[stackptr];
        indexlo = indexloStack[stackptr];
        indexhi = indexhiStack[stackptr];

        /* subarray */
        goto recurse;          
    }  
    else  
        return;
}

/*
sorting of the array with very few items
>> lo - pointer to the first item
>> hi - pointer to the last item
>> indexlo - pointer to the of lo
>> indexhi - pointer to the of hi
>> width - width of an item
>> stride - number of the items that we need to go over when we move to the next item
            NOTE: this means that the items may not placed in a continuous memory space
>> comp - the comparison function
*/
void XShortSort(char * lo, char * hi, int * indexlo, int * indexhi, int width, int stride, int (*comp)(const void *, const void *))
{
    char * p = NULL;
    char * max = NULL;
    int realStride = stride * width;

    if(indexlo == NULL){
        while (hi > lo) { 
            max = lo;   
            for(p = lo + realStride; p <= hi; p += realStride) {
                if (comp(p, max) > 0) {
                    max = p;
                }
            }
            XSwapForSort(max, hi, NULL, NULL, width);
            hi -= realStride;
        }
    }
    else{
        int * pIndex = NULL;
        int * maxIndex = NULL;
        while (hi > lo) { 
            max = lo;
            maxIndex = indexlo;
            for(p = lo + realStride, pIndex = indexlo + stride; p <= hi; p += realStride, pIndex += stride) {
                if (comp(p, max) > 0) {
                    max = p;
                    maxIndex = pIndex;
                }
            }
            XSwapForSort(max, hi, maxIndex, indexhi, width);
            hi -= realStride;
            indexhi -= stride;
        }
    }
}

/*
swap data items
>> a - the first item
>> b - the second item
>> indexA - index of a
>> indexB - index of b
>> width - width of an item
*/
void XSwapForSort(char * a, char * b, int * indexA, int * indexB, int width)
{
    char tmp;
    if(a != b){
        while(width--){
            tmp = *a;
            *a++ = *b;
            *b++ = tmp;
        }
        if(indexA != indexB){
            int indexTMP;
            indexTMP = *indexA;
            *indexA = *indexB;
            *indexB = indexTMP;
        }
    }
}

/* comparison of XFloat */
int CompXFloat(const void * a, const void * b)
{
    float va = *((float*)a);
    float vb = *((float*)b);
    if(va == vb)
        return 0;
    else if(va > vb)
        return -1;
    else
        return 1;
}

/* reset cleans up all runtime-related resources accociated with the GPUs available 
   for the current process.*/
void ResetGPUDevices()
{
#ifdef USE_CUDA

    cudaThreadExit();
    return;

    /*int devNum = 0;
    cudaGetDeviceCount(&devNum);

    for (int i = 0; i < devNum; i++){
        cudaSetDevice(i);
        cudaDeviceReset();
    }*/
#endif
}

} // namespace nts(NiuTrans.Tensor)
