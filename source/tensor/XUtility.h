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

#include <stdio.h>
#include "XGlobal.h"
#include "XDevice.h"

#ifndef __XUTILITY_H__
#define __XUTILITY_H__

namespace nts{ // namespace nts(NiuTrans.Tensor)

extern DTYPE GetFirstDigitNum(DTYPE p);
extern bool IsFloatValid(float f);
extern bool IsNAN(float f);
extern bool IsNAN(double f);
extern bool IsINF(float f);
extern bool IsINF(double f);
extern void ToLowercase(char * str);
extern char * GetNextWord(char * p);
extern void XMemSet(void * p, int value, size_t size);
extern void XMemSet(int devID, void * p, int value, size_t size);
extern void XMemCopy(void * t, int devIDT, const void * s, int devIDS, size_t size);
extern void XMemCopy2D(void * t, size_t tPitch, int devIDT, const void * s, size_t sPitch, int devIDS, size_t mSize, int n);
extern void XMemCopy2DAsync(void * t, size_t tPitch, int devIDT, const void * s, size_t sPitch, int devIDS, size_t mSize, int n, XStream * stream);
extern void * XMemAlloc(int devID, size_t size);
extern void * XMemAllocOnDev(int devID, size_t size);
extern void XMemFree(int devID, void * p);
extern void XMemFreeOnDev(int devID, void * p);
extern DTYPE ToCPU(int devID, void * value);
extern int ToCPUInt(int devID, void * value);
extern bool SetToDevice(int devID, void * p, DTYPE value);
extern bool SetToDeviceInt(int devID, void * p, int value);
extern unsigned int GetNextPower2(unsigned int n);
extern void XSleep(int sleepTime);
extern double GetClock();
extern double GetClockSec();

extern void XQSort(void * data, void * index, int num, int width, int stride, int (*comp)(const void *, const void *));
extern int CompXFloat(const void * a, const void * b);

#ifdef USE_CUDA
extern void XMemCopyAsync(void * t, int devIDT, const void * s, int devIDS, size_t size, cudaStream_t stream, int streamDevID);
#else
extern void XMemCopyAsync(void * t, int devIDT, const void * s, int devIDS, size_t size, void * stream, int streamDevID);
#endif

extern void ResetGPUDevices();

} // namespace nts(NiuTrans.Tensor)

#endif // __XUTILITY_H__
