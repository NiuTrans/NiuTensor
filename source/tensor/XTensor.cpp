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
 * implementation of tensors used in this work. It it is the basis of XMatrix 
 * and XVector
 *
 *
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2017-07-31
 * $Update by: LI Yinqiao (li.yin.qiao.2012@hotmail.com) 2017-11-18 bug fixes
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdarg.h>
#include <time.h>
#include "XTensor.h"
#include "XCall.h"
#include "XGlobal.h"
#include "XUtility.h"
#include "XDevice.h"
#include "XMem.h"
#include "XHeap.h"
#include "XBLAS.h"
#include "XName.h"
#include "core/shape/MergeBlockLists.h"
#include "core/movement/CopyValues.h"
#include "core/arithmetic/Sum.h"
#include "core/arithmetic/Multiply.h"
#include "core/arithmetic/Sub.h"
#include "core/arithmetic/Div.h"
#include "core/math/ScaleAndShift.h"
#include "core/getandset/SetData.h"
#include "function/Identity.h"
#include "core/CHeader.h"

#ifdef USE_CUDA

// the CUDA stuff
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
#include "core/utilities/FlushToMem.cuh"
#include "core/utilities/SetAscendingOrder.cuh"

#endif

/* the nts (NiuTrans.Tensor) namespace */
namespace nts{

int tensorIDGlobal = 0;
MUTEX_HANDLE tensorMutex;
XTensor NULLTensor;

/* generate a tensor id */
int MakeTensorID()
{
    if(tensorIDGlobal == 0)
        MUTEX_INIT(tensorMutex);

    MUTEX_LOCK(tensorMutex);
    int id = tensorIDGlobal++;    
    MUTEX_UNLOCK(tensorMutex);

    return id;
}

/* constructor */
XTensor::XTensor()
{
    Init();
    SetDataPointer();

    id = MakeTensorID();
    isDefaultDType = true;
    isInGlobalMem  = false;
    isInit = false;
    isTmp =  false;
}

/* constructor */
XTensor::XTensor(const XTensor * reference)
{
    Init();
    SetDataPointer();
    id = MakeTensorID();

    InitTensorV2(this, reference);
}

/* 
constructor 
>> myOrder - order of the tensor
>> myDevID - device id
>> myMem - memory pool used to allocating the data array
*/
XTensor::XTensor(const int myOrder, int myDevID, XMem * myMem)
{
    CheckNTErrors((myOrder > 0), "Illegal tensor order1");

    Init();
    SetDataPointer();

    id = MakeTensorID();
    order = myOrder;
    mem = myMem;
    devID = myMem == NULL ? myDevID : myMem->devID;
}

/* 
constructor 
>> myOrder - order of the tensor
>> myDimSize - size of each dimension
>> myDataType - unit size (e.g., int, float, and double)
>> myDenseRatio - how often an element has a non-zero value
>> myDevID - device id
>> myMem - memory pool used to allocating the data array
*/
XTensor::XTensor(const int myOrder, const int * myDimSize, const TENSOR_DATA_TYPE myDataType,
                 const float myDenseRatio, int myDevID, XMem * myMem)
{
    Init();
    SetDataPointer();

    id = MakeTensorID();
    order = myOrder;
    mem = myMem;
    devID = myMem != NULL ? myMem->devID : myDevID;

    if(order >= 0)
        Resize(myOrder, myDimSize, myDataType, myDenseRatio);
}

/* copy constructor */
XTensor::XTensor(const XTensor &reference)
{
    Init();
    SetDataPointer();
    id = MakeTensorID();
    ShallowCopy(reference);
    data = NULL;
    dataHost = NULL;
    
    if(reference.isTmp){
        devID = reference.devID;
        mem = reference.mem;
        data = reference.data;
        signature = reference.signature;
        
        /* what we really want to do is "reference.data = NULL;"
           As "reference" is constant, we cannot reset "reference.data"
           here. So we save the ADDRESS of "reference.data" in
           "reference.dataP", and do this work by updating "*reference.dataP".
           This is VERY tricky and there might be better solutions :) */
        *reference.dataP = NULL;
    }
    else{
        devID = reference.devID;
        mem = reference.mem;
        InitTensorV2(this, &reference);
        _CopyValues(&reference, this);
    }

    if(reference.isTmp)
        XLink::Replace(&reference, this);
    else{
        CheckNTErrors(outgo.tailNum == 0, "The node has outgoing edge to other nodes!");
        XLink::CopyIncoming(&reference, this);
    }

    isInit = true;
    isTmp  = reference.isTmp;
}

/* copy constructor (with right value reference) */
XTensor::XTensor(const XTensor &&reference)
{
    Init();
    SetDataPointer();
    id = MakeTensorID();
    ShallowCopy(reference);
    data = NULL;
    dataHost = NULL;
    
    devID = reference.devID;
    mem = reference.mem;
    data = reference.data;
    signature = reference.signature;
        
    /* what we really want to do is "reference.data = NULL;"
       As "reference" is constant, we cannot reset "reference.data"
       here. So we save the ADDRESS of "reference.data" in
       "reference.dataP", and do this work by updating "*reference.dataP".
       This is VERY tricky and there might be better solutions :) */
    *reference.dataP = NULL;

    XLink::Replace(&reference, this);

    isInit = true;
    isTmp  = reference.isTmp;
}

/* de-constructor */
XTensor::~XTensor()
{
    /* We make a hard copy of the tensor to keep
       the connectivity of the graph. To kill memory
       leak, we release the data of the new tensor
       when its parent is deleted (see ClearIncoming). */
    if(outgo.tailNum > 0){
        int dims[MAX_TENSOR_DIM_NUM];
        memcpy(dims, dimSize, order * sizeof(int));
        dims[0] = -dims[0];
        
        XTensor * newTensor = new XTensor(order, dims, dataType, denseRatio, devID, mem);
        newTensor->SetTMPFlag();
        newTensor->data = data;
        data = NULL;
        
        XLink::Replace(this, newTensor);
    }
    
    XLink::ClearOutgoing(this);
    XLink::ClearIncoming(this);
    
    DestroyData();

    if(grad != NULL)
        delete grad;
}

/* set the name of the tensor */
void XTensor::SetName(const char * myName)
{
    strcpy(name, myName);
}

/* initialize member variables */
void XTensor::Init()
{
    name[0] = '\0';
    id = -1;
    mem = NULL;
    signature = 0;
    data = NULL;
    dataHost = NULL;
    dataP = NULL;
    devID = -1;
    order = -1;
    memset(dimSize, 0, sizeof(int) * MAX_TENSOR_DIM_NUM);
    dataType = DEFAULT_DTYPE;
    unitSize = sizeof(float);
    unitNum = 0;
    isSparse = false;
    unitNumNonZero = 0;
    denseRatio = 1.0F;
    isShared = false;
    isDefaultDType = true;
    isInGlobalMem = false;
    memset(isAllValued, 0, sizeof(bool) * MAX_TENSOR_DIM_NUM);
    isInit = false;
    isTmp =  false;
    isGrad = false;
    isVar  = false;
    enableGrad = true;
    visitMark = 0;
    grad = NULL;
}

/* delete data arrays */
void XTensor::DestroyData()
{
    if(data != NULL && mem == NULL && !isShared)
        XMemFree(devID, data);
    else if(data != NULL && isInGlobalMem)
        FreeData(this, mem);
    else if(data != NULL)
        mem->Release(data, GetDataSizeInChar(), signature);
    
    data = NULL;

    if(dataHost != NULL)
        delete[] (char*)dataHost;
    dataHost = NULL;
}

/* 
shallow copy of the tensor
Note that we do not copy data array here
>> tensor - the source tensor
*/
void XTensor::ShallowCopy(const XTensor &tensor)
{
    strcpy(name, tensor.name);
    order = tensor.order;
    enableGrad = tensor.enableGrad;
    memcpy(dimSize, tensor.dimSize, sizeof(int) * MAX_TENSOR_DIM_NUM);
    dataType = tensor.dataType;
    unitSize = tensor.unitSize;
    unitNum = tensor.unitNum;
    isSparse = tensor.isSparse;
    unitNumNonZero = tensor.unitNumNonZero;
    denseRatio =  tensor.denseRatio;
    isShared = tensor.isShared;
    isDefaultDType = tensor.isDefaultDType;
    isInGlobalMem = tensor.isInGlobalMem;
    memcpy(isAllValued, tensor.isAllValued, sizeof(bool) * MAX_TENSOR_DIM_NUM);
}

/* overloading of the equal-sign */
XTensor& XTensor::operator= (const XTensor& tensor)
{
    /* we must make a hard copy of the tensor if it is the input
       of another node. */
    if(outgo.tailNum > 0){
        int dims[MAX_TENSOR_DIM_NUM];
        memcpy(dims, dimSize, order * sizeof(int));
        dims[0] = -dims[0];
        
        XTensor * newTensor = new XTensor(order, dims, dataType, denseRatio, devID, mem);
        newTensor->SetTMPFlag();
        newTensor->data = data;
        newTensor->dataHost = dataHost;
        newTensor->signature = tensor.signature;
        
        XLink::Replace(this, newTensor);
        XLink::ClearOutgoing(this);
        XLink::ClearIncoming(this);
        newTensor->ShallowCopy(this);

        data = NULL;
        dataHost = NULL;
    }

    if(false && !tensor.isTmp){
        /* NOTE: this might lead to additional data copy by Mac LLVM compilers */
        /* we make an identity transformation here */
        
        if(outgo.tailNum > 0)
            XLink::ClearOutgoing(this);
        XLink::ClearIncoming(this);
        
        if(!_IsSameShaped(this, &tensor))
            Resize(tensor.order, tensor.dimSize, tensor.dataType, tensor.denseRatio);
        
        _Identity(&tensor, this);
        XLink::MakeLink(&tensor, NULL, this, FUNC_IDENTITY);
    }
    else{
        /* hard copy of the data array */
        int size = unitNum * unitSize;
        if( isInit && !isSparse && !tensor.isSparse &&
            size == tensor.unitNum * tensor.unitSize &&
          ((devID < 0 && tensor.devID < 0) && devID == tensor.devID) &&
            data != NULL)
        {
            XMemCopy(data, devID, tensor.data, tensor.devID, size);
            if(dataHost != NULL && tensor.dataHost != NULL)
                XMemCopy(dataHost, -1, tensor.dataHost, tensor.devID, size);
        }
        else{
            DestroyData();
            if(!isInit){
                devID = tensor.devID;
                mem = tensor.mem;
            }

            Resize(tensor.order, tensor.dimSize, tensor.dataType, tensor.denseRatio);
            _CopyValues(&tensor, this);
        }

        /* copy member variables */
        ShallowCopy(tensor);

        isInit = true;
        isTmp  = false;

        CheckNTErrors(outgo.tailNum == 0, "The node has outgoing edge to other nodes!");

        /* create tensor links for the new tensor */
        XLink::Copy(&tensor, this);
    }

    return *this;
}

/* overloading of the equal-sign (with right value reference) */
XTensor& XTensor::operator= (const XTensor&& tensor)
{
    /* we must make a hard copy of the tensor if it is the input
       of another node. */
    if(outgo.tailNum > 0){
        int dims[MAX_TENSOR_DIM_NUM];
        memcpy(dims, dimSize, order * sizeof(int));
        dims[0] = -dims[0];
        
        XTensor * newTensor = new XTensor(order, dims, dataType, denseRatio, devID, mem);
        newTensor->SetTMPFlag();
        newTensor->data = data;
        newTensor->dataHost = dataHost;
        newTensor->signature = tensor.signature;
        
        XLink::Replace(this, newTensor);
        XLink::ClearOutgoing(this);
        XLink::ClearIncoming(this);
        newTensor->ShallowCopy(this);

        data = NULL;
        dataHost = NULL;
    }
    
    DestroyData();

    ShallowCopy(tensor);
    
    isInit = true;
    devID = tensor.devID;
    mem  = tensor.mem;
    data = tensor.data;
    signature = tensor.signature;
        
    /* what we really want to do is "reference.data = NULL;"
       As "reference" is constant, we cannot reset "reference.data"
       here. So we save the ADDRESS of "reference.data" in
       "reference.dataP", and do this work by updating "*reference.dataP".
       This is VERY tricky and there might be better solutions :) */
    *tensor.dataP = NULL;

    XLink::Copy(&tensor, this);

    return *this;
}

/* overloading of the plus-sign */
XTensor XTensor::operator+ (const XTensor& tensor) const
{
    return Sum(*this, tensor);
}

/* overloading of the plus-sign */
XTensor XTensor::operator+ (const DTYPE shift) const 
{
    return ScaleAndShift(*this, 1, shift);
}

/* overloading of the multiply-sign */
XTensor XTensor::operator* (const XTensor& tensor) const
{
    return Multiply(*this, tensor);
}

/* overloading of the multiply-sign */
XTensor XTensor::operator* (const DTYPE scale) const
{
    return ScaleAndShift(*this, scale, 0);
}

/* overloading of the minus-sign */
XTensor XTensor::operator- (const XTensor& tensor) const
{
    return Sub(*this, tensor);
}

/* overloading of the minus-sign */
XTensor XTensor::operator- (const DTYPE shift) const
{
    return ScaleAndShift(*this, 1, -shift);
}

/* overloading of the minus-sign */
XTensor XTensor::operator- () const
{
    return Negate(*this);
}

/* overloading of the division-sign */
XTensor XTensor::operator/ (const XTensor& tensor) const
{
    return Div(*this, tensor);
}

/* overloading of the division-sign */
XTensor XTensor::operator/ (const DTYPE scale) const
{
    return ScaleAndShift(*this, (DTYPE)1/scale, 0);
}

/* 
linear transformation b = a * \scale + \shift
>> scale - the slope
>> shift - the intercept
*/
XTensor XTensor::Lin(DTYPE scale, DTYPE shift) const
{
    return Linear(*this, scale, shift);
}

/* 
relocate the data on the target device 
>> myDevId - target device id
>> myMem - memory pool on the target device
*/
void XTensor::SetDevice(int myDevId, XMem * myMem)
{
    if (myMem != NULL) {
        FlushToMem(myMem);
        isInGlobalMem = false;
    }
    else {
        myMem = GMems.GetMem(myDevId);
    }
}

bool XTensor::IsReduceShaped(const XTensor * a, const XTensor * b, int dim)
{
    if(a == NULL || b == NULL)
        return false;

    if ((a->order - 1) != b->order)
        return false;

    for (int i = 0; i < b->order; i++) {
        if (i < dim) {
            if (a->dimSize[i] != b->dimSize[i])
                return false;
        }
        else if (i >= dim) {
            if (a->dimSize[i+1] != b->dimSize[i])
                return false;
        }
    }

    if(a->dataType != b->dataType)
        return false;

    if(a->denseRatio != b->denseRatio)
        return false;

    if(a->isSparse != b->isSparse)
        return false;

    return true;
}

/* 
set the size of each dimension 
>> myDimSize - size of each dimension
*/
void XTensor::SetDim(int * myDimSize)
{
    for (int i = 0; i < order; i++) {
        dimSize[i] = myDimSize[i];
    }
}

/* 
get the size of a given dimension 
>> dim - the given dim we are looking at
*/
int XTensor::GetDim(const int dim) const
{
    CheckNTErrors(dim < order, "dimenision is out of range!");
    CheckNTErrors(dim >= -order, "dimenision is out of range!");
    
    int d = dim;
    if(dim < 0)
        d = order + dim;

    return dimSize[d];
}

/* 
reshape the tensor 
>> myOrder - order of the tensor
>> myDimSize - size of each dimension
*/
void XTensor::Reshape(const int myOrder, const int * myDimSize)
{
    int dims[MAX_TENSOR_DIM_NUM];
    int num = 1;

    for(int i = 0; i < myOrder; i++){
        num *= myDimSize[i];
        dims[i] = abs(myDimSize[i]);
    }

    CheckNTErrors(abs(num) == unitNum, "Wrong size found when we reshape the tensor!");

    order = myOrder;
    memcpy(dimSize, dims, sizeof(int) * order);
}

/* 
reshape the tensor into a vector
>> num - number of elements
*/
void XTensor::Reshape(const int num)
{
    int dim = num;
    Reshape(1, &dim);
}

/* 
reshape the tensor into a matrix
>> rowNum - number of rows
>> colNum - number of columns
*/
void XTensor::Reshape(const int rowNum, const int colNum)
{
    int dims[2] = {rowNum, colNum};
    Reshape(2, dims);
}

/*
reshape the tensor by merging two consecutive dimensions
>> i - dimension i
>> j - i + 1
*/
void XTensor::ReshapeMerged(const int i, const int j)
{
    if (i < 0)
        return;

    int di = i;
    int dj = j < 0 ? i + 1 : j;

    CheckNTErrors(di < order, "Wrong dimension index!");


    int dims[MAX_TENSOR_DIM_NUM];

    for (int k = 0; k < di; k++)
        dims[k] = dimSize[k];
    dims[di] = dimSize[di] * dimSize[dj];
    for (int k = dj + 1; k < order; k++)
        dims[k - 1] = dimSize[k];

    Reshape(order - 1, dims);
}

/* return a tensor that datatype is same as the special tensor */
XTensor XTensor::TypeAs(const XTensor input)
{
    return ConvertDataType(*this, input.dataType);
}

/* get the number of items in the data array */
int XTensor::GetSize() const
{
    if(isSparse)
        return unitNumNonZero;
    else
        return unitNum;
}

/* get the size of the memory space used */
int XTensor::GetDataSizeInChar() const
{
    if(isSparse){
        int num = int(unitNum * denseRatio + 1);
        int tupleSize = sizeof(int)+sizeof(DTYPE);
        int size = sizeof(int) + tupleSize*(num);
        return size;
    }
    else{
        return unitNum * unitSize;
    }
}

/* 
get unit size in terms of "dataType" 
>> myDataType - type of unit
<< return - unit size
*/
int XTensor::GetUnitSize(TENSOR_DATA_TYPE myDataType) const
{
    if(myDataType == X_INT)
        return sizeof(int);
    else if(myDataType == X_FLOAT)
        return sizeof(float);
    else if(myDataType == X_DOUBLE)
        return sizeof(double);
    else if(myDataType == X_INT8)
        return 1;
    else if(myDataType == X_FLOAT16)
        return 2;
    return sizeof(float);
}

/* 
get offset (2D) 
>> row - index of demension 0
>> col - index of demension 1
*/
MTYPE XTensor::GetOffset2D(int row, int col) const
{
    CheckNTErrors(order == 2, "Cannot get a 2d cell for a tensor whose order is not 2!");
    CheckNTErrors(row >= 0 && row < dimSize[0], "dimension 0 is out of range!");
    CheckNTErrors(col >= 0 && col < dimSize[1], "dimension 1 is out of range!");

    return row * dimSize[1] + col;
}

/* 
get offset (3D) 
>> d0 - index of demension 0
>> d1 - index of demension 1
>> d2 - index of demension 2
*/
MTYPE XTensor::GetOffset3D(int d0, int d1, int d2) const
{
    CheckNTErrors(order == 3, "Cannot get a 3d cell for a tensor whose order is not 2!");
    CheckNTErrors(d0 >= 0 && d0 < dimSize[0], "dimension 0 is out of range!");
    CheckNTErrors(d1 >= 0 && d1 < dimSize[1], "dimension 1 is out of range!");
    CheckNTErrors(d2 >= 0 && d2 < dimSize[2], "dimension 2 is out of range!");

    return (d0 * dimSize[1] + d1) * dimSize[2] + d2;
}

/* 
a vector with all entries of 0 
>> stream - stream for the job pipeline
*/
void XTensor::SetZeroAll(XStream * stream)
{
    if(data == NULL)
        return;

    if(isSparse){
        if(devID >= 0){
#ifdef USE_CUDA
            int size = sizeof(int) + (sizeof(int)+sizeof(DTYPE)) * unitNumNonZero;
            
            int devIDBackup = 0;
            cudaGetDevice(&devIDBackup);
            cudaSetDevice(devID);

            if(stream == NULL)
                cudaMemset(data, 0, size);
            else
                cudaMemsetAsync(data, 0, size, stream->stream);
            
            cudaSetDevice(devIDBackup);
#endif
        }
        else
            *(int*)data = 0;

        unitNumNonZero = 0; 
    }
    else{
        if(devID >= 0){
#ifdef USE_CUDA
            int devIDBackup = 0;
            cudaGetDevice(&devIDBackup);
            cudaSetDevice(devID);
            
            if(stream == NULL)
                cudaMemset(data, 0, unitNum * unitSize);
            else
                cudaMemsetAsync(data, 0, unitNum * unitSize, stream->stream);
            
            cudaSetDevice(devIDBackup);
#endif
        }
        else
            memset(data, 0, unitNum * unitSize);
    }
}

/*  set the tensor with an data array 
>> d - input data. it must be on CPU
>> num - number of data items
>> beg - where we start the data copy in the data array of the tensor
*/
void XTensor::SetData(const void * d, int num, int beg)
{
    if (data == NULL || d ==NULL)
        return;

    CheckNTErrors(!isSparse, "TODO");
    CheckNTErrors(num <= unitNum - beg, "Illegal size!");

    XMemCopy((char*)data + beg * unitSize, devID, d, -1, num * unitSize);
}

/* generate data items with a uniform distribution in [0, 1] */
void XTensor::Rand(int rNum, int cNum)
{
    _SetDataRand(this, rNum, cNum);
}

/* generate data items with a range by start, end and the step
>> start - the begin of the array
>> end - the end of the array (not included self)
>> step - the step of two items
*/
void XTensor::Range(DTYPE lower, DTYPE upper, DTYPE step)
{
    _SetDataRange(this, lower, upper, step);
}

/* 
set the tensor items by a uniform distribution in range [lower, upper]
>> lower - lower value of the range
>> upper - upper value of the range
*/
void XTensor::SetDataRand(DTYPE lower, DTYPE upper)
{
    // TODO: GPU code!!!!!!!

    if (data == NULL)
        return;

    // srand((unsigned)time(0));
    DTYPE variance = upper - lower;
    void * d = NULL;
    if (dataType == X_FLOAT) {
        d = new float[unitNum];
        for (int i = 0; i < unitNum; i++) {
            DTYPE value = lower + variance * (float)rand() / RAND_MAX;
            *((float*)d + i) = value;
        }
    }
    else if (dataType == X_DOUBLE) {
        d = new double[unitNum];
        for (int i = 0; i < unitNum; i++) {
            *((double*)d + i) = lower + variance * rand() / RAND_MAX;
        }
    }
    else {
        ShowNTErrors("Data type must be X_FLOAT or X_Double!");
    }

    SetData(d, unitNum);
    
    if (dataType == X_FLOAT) {
        delete[] (float*)d;
    }
    else {
        delete[] (double*)d;
    }
}

/* a gauss distribution (Box-Muller method) */
double GaussRand(DTYPE mean, DTYPE standardDeviation)
{
    // TODO: GPU code!!!!!!!

    static double u, v;
    static int phase = 0;
    double z;
    double pi = 3.141592654;

    if (phase == 0){
        u = (rand() + 1.0) / (RAND_MAX + 1.0);
        v = (rand() + 1.0) / (RAND_MAX + 1.0);
        z = sqrt(-2.0 * log(u))* sin(2.0 * pi * v);
    }
    else{
        z = sqrt(-2.0 * log(u)) * cos(2.0 * pi * v);
    }

    phase = 1 - phase;
    return mean + (z * standardDeviation);
}

/* 
set the tensor items by a normal distribution
>> mean - mean or expectation of the distribution
>> standardDeviation - standard deviation of the distribution
*/
void XTensor::SetDataRandn(DTYPE mean, DTYPE standardDeviation)
{
    // TODO: cuda code!!!!!!!

    if (data == NULL)
        return;

    // srand((unsigned)time(0));
    void * d = NULL;
    if (dataType == X_FLOAT) {
        d = new float[unitNum];
        for (int i = 0; i < unitNum; i++) {
            *((float*)d + i) = (float)GaussRand(mean, standardDeviation);
        }
    }
    else if (dataType == X_DOUBLE) {
        d = new double[unitNum];
        for (int i = 0; i < unitNum; i++) {
            *((double*)d + i) = GaussRand(mean, standardDeviation);
        }
    }
    else {
        ShowNTErrors("Data type must be X_FLOAT or X_Double!");
    }

    SetData(d, unitNum);

    if (dataType == X_FLOAT) {
        delete[] (float*)d;
    }
    else {
        delete[] (double*)d;
    }
}

/* 
set tensor items with an array of offsets 
>> offsets - offset for each data item
>> value - value for the data items
>> num - number of the data items
*/
void XTensor::SetDataBatched(MTYPE * offsets, DTYPE value, int num)
{
    _SetDataWithOffset(this, offsets, value, num);
}

/* 
set tensor items with an array of values 
>> offsets - offset for each data item
>> values - value for each data item
>> num - number of the data items
*/
void XTensor::SetDataBatchedWithValues(MTYPE * offsets, void * values, int num)
{
    _SetDataWithOffsetAndValue(this, offsets, values, num);
}

/* set the pointer to "data" */
void XTensor::SetDataPointer()
{
    dataP = &data;
}

/* 
get the value of a cell with the index 
>> index - index of each dimension
>> size - size of the index
<< return - cell value
*/
DTYPE XTensor::Get(int index[], int size) const
{
    CheckNTErrors(dataType == DEFAULT_DTYPE, "The tensor is not in the default type.");

    return ToCPU(devID, GetCell(index, size));
}
    
/*
get the value of a cell with its offset
>> offset - offset in the array
<< return - cell value
*/
DTYPE XTensor::Get(int offset) const
{
    CheckNTErrors(dataType == DEFAULT_DTYPE, "The tensor is not in the default type.");
    CheckNTErrors(offset >= 0 && offset < unitNum, "Invalid index!");
    CheckNTErrors(data != NULL, "Cannot use an uninitialized tensor!");
    CheckNTErrors(denseRatio == 1.0F, "Only dense tensors are supported in Get(offset).");
    
    DTYPE * address = (DTYPE*)data + offset;
    
    return ToCPU(devID, address);
}

/* 
get the pointer to a cell
>> index - index of each dimension
>> size - size of index
<< return - pointer to the cell
*/
void * XTensor::GetCell(int index[], int size) const
{
    CheckNTErrors((size == order), "Illegal index!");

    int offset = index[0];
    for(int i = 1; i < size; ++i){
        CheckNTErrors((index[i] < dimSize[i]), "Index is out of range!");
        offset = offset * dimSize[i] + index[i];
    }
    
    if(isSparse){
        DTYPE value;
        void * p;
        if(BinarySearch(offset, value, p))
            return (char*)p + sizeof(int);
        else
            return NULL;
    }
    else{
        return ((char*)data) + offset * unitSize;
    }
}

/*
get the value of a cell in a 1d tensor in default type
>> i - idex
<< return - value of cell(i) in float
*/
DTYPE XTensor::Get1D(int i) const
{
    CheckNTErrors((order == 1), "Cannot get a 2d cell for a tensor whose order is not 2!");
    CheckNTErrors((i >= 0 && i < dimSize[0]), "dimension 0 is out of range!");
    CheckNTErrors((dataType == DEFAULT_DTYPE), "The tensor is not in default type.");
    
    int dimSize[1] = {i};
    void * value = GetCell(dimSize, 1);
    
    return ToCPU(devID, value);
}
    
/* 
get the value of a cell in a 2d tensor in default type
>> ni - row index
>> mi - column index
<< return - value of cell(ni, mi) in float
*/
DTYPE XTensor::Get2D(int ni, int mi) const
{
    CheckNTErrors((order == 2), "Cannot get a 2d cell for a tensor whose order is not 2!");
    CheckNTErrors((ni >= 0 && ni < dimSize[0]), "dimension 0 is out of range!");
    CheckNTErrors((mi >= 0 && mi < dimSize[1]), "dimension 1 is out of range!");
    CheckNTErrors((dataType == DEFAULT_DTYPE), "The tensor is not in default type.");

    int dims[2] = {ni, mi};
    void * value = GetCell(dims, 2);
    
    return ToCPU(devID, value);
}

/* 
get the value of a cell in a 3d tensor 
>> d0 - index of dimension 0
>> d1 - index of dimension 1
>> d2 - index of dimension 2
*/
DTYPE XTensor::Get3D(int d0, int d1, int d2) const
{
    CheckNTErrors((order == 3), "Cannot get a 2d cell for a tensor whose order is not 2!");
    CheckNTErrors((d0 >= 0 && d0 < dimSize[0]), "dimension 0 is out of range!");
    CheckNTErrors((d1 >= 0 && d1 < dimSize[1]), "dimension 1 is out of range!");
    CheckNTErrors((d2 >= 0 && d2 < dimSize[2]), "dimension 2 is out of range!");
    CheckNTErrors((dataType == DEFAULT_DTYPE), "The tensor is not in default type.");

    int dims[3] = {d0, d1, d2};
    void * value = GetCell(dims, 3);
    
    return ToCPU(devID, value);
}
    
/*
get the int value of a cell by its offset
>> offset - offset of the item
*/
int XTensor::GetInt(int offset) const
{
    CheckNTErrors(dataType == X_INT, "The tensor is not in the integer type.");
    CheckNTErrors(offset >= 0 && offset < unitNum, "Invalid index!");
    CheckNTErrors(data != NULL, "Cannot use an uninitialized tensor!");
    CheckNTErrors(denseRatio == 1.0F, "Only dense tensors are supported in Get(offset).");
    
    int * address = (int*)data + offset;
    
    return ToCPUInt(devID, address);
}

/*
get the value of a cell in a 1d tensor in int type
>> i - index
<< return - value of cell(i) in int
*/
int XTensor::Get1DInt(int i) const
{
    CheckNTErrors(order == 1, "Cannot get a 2d cell for a tensor whose order is not 2!");
    CheckNTErrors(i >= 0 && i < dimSize[0], "dimension 0 is out of range!");
    CheckNTErrors(dataType == X_INT, "The tensor is not in int type.");
    
    int dimSize[1] = {i};
    void * value = GetCell(dimSize, 1);
    
    return ToCPUInt(devID, value);
}
    
/* 
get the value of a cell in a 2d tensor in int type
>> ni - row index
>> mi - column index
<< return - value of cell(ni, mi) in int
*/
 int XTensor::Get2DInt(int ni, int mi) const
{
    CheckNTErrors(order == 2, "Cannot get a 2d cell for a tensor whose order is not 2!");
    CheckNTErrors(ni >= 0 && ni < dimSize[0], "dimension 0 is out of range!");
    CheckNTErrors(mi >= 0 && mi < dimSize[1], "dimension 1 is out of range!");
    CheckNTErrors(dataType == X_INT, "The tensor is not in default type.");

    int dims[2] = {ni, mi};
    void * value = GetCell(dims, 2);
    
    return ToCPUInt(devID, value);
}

/* 
get the value of a cell in a 3d tensor in int type
>> d0 - index of dimension 0
>> d1 - index of dimension 1
>> d2 - index of dimension 2
<< return - value of cell(d0, d1, d2) in int
*/
int XTensor::Get3DInt(int d0, int d1, int d2) const
{
    CheckNTErrors(order == 3, "Cannot get a 2d cell for a tensor whose order is not 2!");
    CheckNTErrors(d0 >= 0 && d0 < dimSize[0], "dimension 0 is out of range!");
    CheckNTErrors(d1 >= 0 && d1 < dimSize[1], "dimension 1 is out of range!");
    CheckNTErrors(d2 >= 0 && d2 < dimSize[2], "dimension 2 is out of range!");
    CheckNTErrors(dataType == X_INT, "The tensor is not in default type.");

    int dims[3] = {d0, d1, d2};
    void * value = GetCell(dims, 3);
    
    return ToCPUInt(devID, value);
}

/* 
get the value of a cell in the sparse tensor 
>> i - i-th tuple in the tuple list of the sparse tensor
<< return - value of the tuple
*/
DTYPE XTensor::GetInSparse(int i) const
{
    CheckNTErrors(i >= 0 && i < unitNum, "Index is out of range!");
    CheckNTErrors(dataType == DEFAULT_DTYPE, "The tensor is not in default type.");

    char * d = (char*)data + sizeof(int);
    DTYPE * value = (DTYPE*)(d + (sizeof(int) + sizeof(DTYPE)) * i + sizeof(int));

    return ToCPU(devID, value);
}

/* 
get the key value of a tuple in a sparse tensor 
>> i - i-th tuple in the tuple list of the sparse tensor
<< return - key of the tuple
*/
int XTensor::GetKeyInSparse(int i) const
{
    CheckNTErrors(i >= 0 && i < unitNum, "Index is out of range!");
    CheckNTErrors(dataType == DEFAULT_DTYPE, "The tensor is not in default type.");

    char * d = (char*)data + sizeof(int);
    int * key = (int*)(d + (sizeof(int) + sizeof(DTYPE)) * i);
    
    return ToCPUInt(devID, key);
}

/* 
set the value of a cell 
>> value - value we tend to set
>> index - index of the cell for each dimension
>> size - size of the index
*/
bool XTensor::Set(DTYPE value, int index[], int size)
{
    CheckNTErrors(dataType == DEFAULT_DTYPE, "The tensor is not in default type.");

    return SetToDevice(devID, GetCell(index, size), value);
}

/*
set the value of a cell with its offset in the array
>> value - the value we intend to set
>> offset - the offset in the array
*/
bool XTensor::Set(DTYPE value, int offset)
{
    CheckNTErrors(offset >= 0 && offset < unitNum, "Invalid index!");
    CheckNTErrors(data != NULL, "Cannot use an uninitialized tensor!");

    DTYPE * d = (DTYPE*)data + offset;

    return SetToDevice(devID, d, value);
}

/* 
set the value of a cell in a 1d tensor 
>> value - value we tend to set
>> i - item offset
<< return - succeeded or not
*/
bool XTensor::Set1D(DTYPE value, int i)
{
    CheckNTErrors(order == 1, "Cannot get a 2d cell for a tensor whose order is not 2!");
    CheckNTErrors(i >= 0 && i < dimSize[0], "dimension 0 is out of range!");
    CheckNTErrors(dataType == DEFAULT_DTYPE, "The tensor is not in default type.");

    int dims[1] = {i};

    return SetToDevice(devID, GetCell(dims, 1), value);
}

/* 
set the value of a cell in a 2d tensor in default type
>> value - value we tend to set
>> ni - row index
>> mi - column index
<< return - succeeded or not
*/
bool XTensor::Set2D(DTYPE value, int ni, int mi)
{
    CheckNTErrors(order == 2, "Cannot get a 2d cell for a tensor whose order is not 2!");
    CheckNTErrors(ni >= 0 && ni < dimSize[0], "dimension 0 is out of range!");
    CheckNTErrors(mi >= 0 && mi < dimSize[1], "dimension 1 is out of range!");
    CheckNTErrors(dataType == DEFAULT_DTYPE, "The tensor is not in default type.");

    int dims[2] = {ni, mi};

    return SetToDevice(devID, GetCell(dims, 2), value);
}

/* 
set the value of a cell in a 3d tensor in default type
>> value - value we tend to set
>> d0 - index of demension 0
>> d1 - index of demension 1
>> d2 - index of demension 2
<< return - succeeded or not
*/
bool XTensor::Set3D(DTYPE value, int d0, int d1, int d2)
{
    CheckNTErrors(order == 3, "Cannot get a 2d cell for a tensor whose order is not 2!");
    CheckNTErrors(d0 >= 0 && d0 < dimSize[0], "dimension 0 is out of range!");
    CheckNTErrors(d1 >= 0 && d1 < dimSize[1], "dimension 1 is out of range!");
    CheckNTErrors(d2 >= 0 && d2 < dimSize[2], "dimension 2 is out of range!");
    CheckNTErrors(dataType == DEFAULT_DTYPE, "The tensor is not in default type.");

    int dims[3] = {d0, d1, d2};

    return SetToDevice(devID, GetCell(dims, 3), value);
}
    
/*
set the integer value of a cell by its offset
>> value - value we tend to set to the item
>> offset - offset of the item
*/
bool XTensor::SetInt(int value, int offset)
{
    CheckNTErrors(offset >= 0 && offset < unitNum, "Invalid index!");
    CheckNTErrors(data != NULL, "Cannot use an uninitialized tensor!");
    
    int * d = (int*)data + offset;
    
    return SetToDeviceInt(devID, d, value);
}


/* 
set the integer value of a cell 
>> value - value we tend to set
>> index - index of the cell for each dimension
>> size - size of the index
<< return - succeeded or not
*/
bool XTensor::SetInt(int value, int index[], int size)
{
    CheckNTErrors(dataType == X_INT, "The tensor is not in integer type.");

    return SetToDeviceInt(devID, GetCell(index, size), value);
}

/* 
set the integer value of a cell in a 1d tensor 
>> value - value we tend to set
>> i - item offset
<< return - succeeded or not
*/
bool XTensor::Set1DInt(int value, int i)
{
    CheckNTErrors(order == 1, "Cannot get a 2d cell for a tensor whose order is not 2!");
    CheckNTErrors(i >= 0 && i < dimSize[0], "dimension 0 is out of range!");
    CheckNTErrors(dataType == X_INT, "The tensor is not in integer type.");

    int dims[1] = {i};

    return SetToDeviceInt(devID, GetCell(dims, 1), value);
}

/* 
set the integer value of a cell in a 2d tensor in default type
>> value - value we tend to set
>> ni - row index
>> mi - column index
<< return - succeeded or not
*/
bool XTensor::Set2DInt(int value, int ni, int mi)
{
    CheckNTErrors(order == 2, "Cannot get a 2d cell for a tensor whose order is not 2!");
    CheckNTErrors(ni >= 0 && ni < dimSize[0], "dimension 0 is out of range!");
    CheckNTErrors(mi >= 0 && mi < dimSize[1], "dimension 1 is out of range!");
    CheckNTErrors(dataType == X_INT, "The tensor is not in integer type.");

    int dims[2] = {ni, mi};

    return SetToDeviceInt(devID, GetCell(dims, 2), value);
}

/* 
set the integer value of a cell in a 3d tensor in default type
>> value - value we tend to set
>> d0 - index of demension 0
>> d1 - index of demension 1
>> d2 - index of demension 2
<< return - succeeded or not
*/
bool XTensor::Set3DInt(int value, int d0, int d1, int d2)
{
    CheckNTErrors(order == 3, "Cannot get a 2d cell for a tensor whose order is not 2!");
    CheckNTErrors(d0 >= 0 && d0 < dimSize[0], "dimension 0 is out of range!");
    CheckNTErrors(d1 >= 0 && d1 < dimSize[1], "dimension 1 is out of range!");
    CheckNTErrors(d2 >= 0 && d2 < dimSize[2], "dimension 2 is out of range!");
    CheckNTErrors((dataType == X_INT), "The tensor is not in integer type.");

    int dims[3] = {d0, d1, d2};

    return SetToDeviceInt(devID, GetCell(dims, 3), value);
}

/* 
increase the value of a cell in a 2d tensor
>> value - value we tend to set
>> ni - row index
>> mi - column index
<< return - succeeded or not
*/
 bool XTensor::Add2D(DTYPE value, int ni, int mi)
{
    CheckNTErrors(ni >= 0 && ni < dimSize[0], "the row index is out of range!");
    CheckNTErrors(mi >= 0 && mi < dimSize[1], "the column index is out of range!");
    CheckNTErrors(dataType == DEFAULT_DTYPE, "The tensor is not in default type.");
    CheckNTErrors(isSparse == false, "TODO!");

    if(devID < 0){
        DTYPE * p = (DTYPE*)data + ni * dimSize[1] + mi;

        CheckNTErrors((p != NULL), "No data array is found!");    

        *p = *p + value;
    
        return true;
    }
    else{
        int dims[2] = {ni, mi};
        return SetToDevice(devID, GetCell(dims, 2), Get2D(ni, mi) + value);
    }
}

/* get the number of non-zero elements (in a sparse tensor) */
int XTensor::GetNonzeroSize() const
{
    if(!isSparse){
        XPRINT(1, stderr, "WARNING! Counting non-zero elements in a dense tensor might be slow!\n");
        CheckNTErrors(devID < 0, "TODO");
        if(dataType == DEFAULT_DTYPE){
            int count = 0;
            for(int i = 0; i < unitNum; i++){
                DTYPE value = *(DTYPE*)((char*)data + i * sizeof(DTYPE));
                if(value == 0)
                    count++;
            }
            return count;
        }
        else{
            ShowNTErrors("TODO!");
            return -1;
        }
    }
    else{
        /* return the head of the tuple list */
        return unitNumNonZero;
    }
}

/* 
set the tensor as "temporary" 
>> myIsTMP - the flag
*/
void XTensor::SetTMPFlag(bool myIsTmp)
{
    isTmp = myIsTmp;
}

/* 
set the tensor as "keep-gradient" 
>> myIsGrad - the flag
*/
void XTensor::SetGradFlag(bool myIsGrad)
{
    isGrad = myIsGrad;
}

/* 
set the tensor as "variable" 
>> myIsVar - the flag
*/
void XTensor::SetVarFlag(bool myIsVar)
{
    isVar = myIsVar;
    if(isVar)
        SetGradFlag(true);
}

/* 
resize a tensor with a specified tensor size
>> myOrder - order of the tensor
>> myDimSize - the size of each dimension
>> myDataType - unit size (e.g., int, float, and double)
>> myDenseRatio - how often an element has non-zero value
<< return - succeeded or not
*/
bool XTensor::Resize(const int myOrder, const int * myDimSize, 
                     const TENSOR_DATA_TYPE myDataType, const float myDenseRatio)
{
    /* free old mem */
    if(data != NULL){
        if (mem == NULL)
            XMemFree(devID, data);
        else
            mem->Release(data, GetDataSizeInChar(), signature);
    }

    signature = mem != NULL ? mem->GetSignature() : 0;
    
    order = myOrder;
    unitNum = 1;
    unitNumNonZero = 0;
    isInit = true;

    bool filledData = true;
    bool zeroData = false;
    for(int i = 0; i < order; i++){
        dimSize[i] = abs(myDimSize[i]);
        if(myDimSize[i] < 0)
            filledData = false;
        if(myDimSize[i] == 0)
            zeroData = true;
        unitNum *= dimSize[i];
    }

    data = NULL;
    denseRatio = myDenseRatio;
    isSparse = denseRatio < 1.0F ? true : false;
    dataType = myDataType;
    unitSize = GetUnitSize(dataType);

    if(myDataType != DEFAULT_DTYPE)
        isDefaultDType = false;
    else
        isDefaultDType = true;

    if(zeroData){
        unitNum = 0;
        return false;
    }

    if(isSparse){
        /*
        for sparse matrices, we use a list of tuple (key, value), 
        ordered by key. Take a (2-dimensional) matrix as an example, 
        we have key = m * i + j;
        The data array is
        ---------
        | 0 | 3 |
        ---------
        | 5 | 0 |
        ---------
        we have
        2
        (0, 1, 3)
        (1, 0, 5)
        where the first number (2) indicates the number of elements.
        */
        
        int num = int(unitNum * denseRatio + 1);
        int tupleSize = sizeof(int)+sizeof(DTYPE);
        int size = sizeof(int) + tupleSize*(num);
        
        if(filledData){
            int * d = NULL;

            if(mem == NULL){
                d = new int[size];
                memset(d, 0, size);
            }
            else{
                d = (int*)mem->Alloc(mem->devID, size);
            }

            if(d == NULL)
                return false;

#if !defined(UNSAFE_BUT_FAST_MEM)
            XMem::SetZero(d, sizeof(int), mem);
#endif
            data = d;
        }
        return true;
    }
    else{
        if(filledData){
            /* allocate the new one */
            if(mem == NULL){
                data = XMemAlloc(devID, unitNum * unitSize); 
#if defined(UNSAFE_BUT_FAST_MEM)
                XMemSet(devID, data, 0, unitNum * unitSize);
#endif
            }
            else
                data = (void*)mem->Alloc(mem->devID, unitNum * unitSize);

            if(data == NULL)
                return false;
        }

#if !defined(UNSAFE_BUT_FAST_MEM)
        if(data != NULL)
            XMem::SetZero(data, unitNum * unitSize, mem);
#endif
        return true;
    }
}

/* 
resize a tensor by another
>> myTensor - tensor for reference
*/
bool XTensor::Resize(const XTensor * myTensor)
{
    denseRatio = myTensor->denseRatio;
    TENSOR_DATA_TYPE myDataType = myTensor->dataType;

    if(myDataType != DEFAULT_DTYPE)
        isDefaultDType = false;
    else
        isDefaultDType = true;

    return Resize(myTensor->order, myTensor->dimSize, myDataType, denseRatio);
}

/* 
binary search to find an element in a sparse tensor
>> key - for search
>> value - value for return
>> position - the position of the tuple.
              it is the previous one if there is no hit
<< return - found it or not?
*/
bool XTensor::BinarySearch(int key, DTYPE &value, void * &position) const
{
    CheckNTErrors((isSparse), "A sparse tensor is required!");
    CheckNTErrors((dataType == DEFAULT_DTYPE), "The tensor is not in the default type.");

    int * d = (int*)data;

    if(key < 0 || *d == 0){
        value = 0;
        position = NULL;
        return false;
    }

    int low = 0;  
    int high = *d - 1;  
    int last = -1;
    bool ok = false;
    int * k = NULL;
    int headSize = sizeof(int);
    int tupleSize = sizeof(int)+sizeof(DTYPE);
    char * p = (char*)data + headSize;

    while (low <= high){  
        int mid = low + (high-low)/2;
        k = (int*)(p + tupleSize * mid);
        if (*k == key){
            ok = true;
            high = mid -1;
            break;
        }  
        else if(*k > key){
            high = mid -1;
        }
        else{
            low = mid +1;
            last = mid;
        }
    }  

    if(ok){
        DTYPE * p = (DTYPE*)((char*)k + sizeof(int));
        value = *p;
        position = k;
        return true;
    }
    else{
        value = 0;
        if(last == -1)
            position = NULL;
        else
            position = (char*)data + headSize + tupleSize * last;
        return false;
    }
}

/* 
dump data to a file 
>> file - where to domp the data
>> label - label of the tensor
>> n - number of items to dump
>> beg - the first item id
>> verbose - verbose level
*/
void XTensor::Dump(FILE * file, const char * label, const int n, const int beg, const int verbose)
{
    if (verbose > verboseLevel)
        return;

    void * d = data;
    bool isNewData = false;

#ifdef USE_CUDA
    if (devID >= 0) {
        CudaGPUToCPUFlush(this);
        d = dataHost;
        isNewData = true;
    }
#endif

    if (d == NULL) {
        if (isSparse) {
            int num = 0;
            for (int i = 0; i < order; i++)
                num *= dimSize[i];
            num = int(num * denseRatio + 1);
            int tupleSize = sizeof(int) + sizeof(DTYPE);
            int size = sizeof(int) + tupleSize*(num);

            d = new char[size];
            memset(d, 0, size);
        }
        else {
            d = new char[unitNum * unitSize];
            memset(d, 0, unitNum * unitSize);
        }
        isNewData = true;
    }

    if (label != NULL)
        fprintf(file, "%s ", label);
    
    if(isInit){
        fprintf(file, "order=%d dimsize=", order);
        for (int i = 0; i < order; i++) {
            fprintf(file, "%d", dimSize[i]);
            if (i < order - 1)
                fprintf(file, ",");
        }
    }
    else{
        fprintf(file, "order=-1 dimsize=-1");
    }

    fprintf(file, " dtype=%s dense=%f\n", GetDataTypeName(dataType), denseRatio);

    if(!isInit){
        fprintf(file, "NULL");
    }
    if (!isSparse) {
        if (dataType == DEFAULT_DTYPE) {
            int end = MIN(n > 0 ? beg + n : beg + unitNum, unitNum);
            for(int i = beg; i < end; i++){
                DTYPE f = ((DTYPE*)d)[i];
                if(i == beg)
                    fprintf(file, "%e", f);
                else
                    fprintf(file, " %e", f);

            }
        }
        else if (dataType == X_INT) {
            int end = MIN(n > 0 ? beg + n : beg + unitNum, unitNum);
            for(int i = beg; i < end; i++){
                int f = ((int*)d)[i];
                if(i == beg)
                    fprintf(file, "%d", f);
                else
                    fprintf(file, " %d", f);
            }
        }
        else
            ShowNTErrors("TODO!");
    }
    else {
        int num = this->unitNumNonZero > 0 ? *(int*)d : 0;
        if (beg + n > 0)
            num = MIN(num, beg + n);
        fprintf(file, "%d ", num);
        for (int i = beg; i < num; i++) {
            int key = GetKeyInSparse(i);
            DTYPE value = GetInSparse(i);
            fprintf(file, "[%d]%e ", key, value);
        }
    }
    fprintf(file, "\n");

    if (isNewData) {
        delete[](char*)d;
#ifdef USE_CUDA
        if (devID >= 0)
            dataHost = NULL;
#endif
    }
}

/* 
dump data to a file
>> tensor - the tensor for dumping
>> file - where to domp the data
>> label - label of the tensor
>> n - number of the items to dump
>> beg - the first item id
>> verbose - verbose level
*/
void XTensor::Dump(const XTensor * tensor, FILE * file, const char * label, const int n, const int beg, const int verbose)
{
    XTensor a(tensor->order, tensor->dimSize, tensor->dataType, tensor->denseRatio, tensor->devID, tensor->mem);
    _CopyValues(tensor, &a);
    a.Dump(file, label, n, beg, verbose);
}

/* 
dump data to a binary file 
>> file - where to dump the data
*/
void XTensor::BinaryDump(FILE* file)
{
    XTensor tmp;
    InitTensorOnCPU(&tmp, this);
    _CopyValues(this, &tmp);

    switch (dataType) {
    case X_INT: {
        fwrite(tmp.data, sizeof(int), unitNum, file);
    }
    default: {
        fwrite(tmp.data, sizeof(float), unitNum, file);
    }
    }
}

/* 
read data from a file
>> file - where to load the data
>> label - label of the tensor
*/
void XTensor::Read(FILE * file, const char * label)
{
    char typeName[32] = "";
    char dimSizeName[128] = "";
    int dimNum;
    int dims[MAX_TENSOR_DIM_NUM];
    float dRatio;

    int head = (int)strlen(label);
    if (label != NULL) {
        for (int i = 0; i < head; i++) {
            char c = fgetc(file);
            CheckNTErrors(c == label[i], "Incorrect tensor label!");
        }
    }

    fgetc(file);

    if (fscanf(file, "order=%d dimsize=%s dtype=%s dense=%f",
                      &dimNum, dimSizeName, typeName, &dRatio) < 4) {
        ShowNTErrors("Incorrect format when reading the tensor!");
    }

    char c;
    
    do {
        c = fgetc(file);
    } while (c != '\n' && c != EOF);

    isSparse = dRatio < 1.0F ? true : false;

    int o = 0;
    bool sameSize = true;
    char * p = dimSizeName;
    while (*p != 0) {
        while (*p == ' ' || *p == '\t')
            p++;
        int dsize = 0;
        while (*p != ',' && *p != ' ' && *p != '\t' && *p != '\0') {
            CheckNTErrors(*p >= '0' && *p <= '9', "Incorrect number format!");
            dsize = dsize * 10 + (*p - '0');
            p++;
        }
        p++;
        dims[o++] = dsize;
        if (dims[o - 1] != dimSize[o - 1])
            sameSize = false;
    }

    CheckNTErrors(o == dimNum, "Incorrect dimension number!");
    for (int i = 0; i < o; i++) {
        CheckNTErrors(dimSize[i] == dims[i], "Incorrect dimension size!");
    }

    if (!sameSize || dRatio > denseRatio || GetDataType(typeName) != dataType)
        Resize(dimNum, dims, GetDataType(typeName), dRatio);

    void * dataBuf = XMemAlloc(-1, GetDataSizeInChar());
    void * dataBackup = data;
    data = dataBuf;

    if (!isSparse) {
        if (dataType == DEFAULT_DTYPE) {
            for (int i = 0; i < unitNum; i++) {
                DTYPE * f = ((DTYPE*)data) + i;
                if (fscanf(file, "%e", f) < 1) {
                    ShowNTErrors("Incorrect tensor format!");
                }
            }
        }
        else {
            ShowNTErrors("TODO!");
        }
    }
    else {
        int num = 0;
        if (fscanf(file, "%d", &num) < 1) {
            ShowNTErrors("Incorrect tensor format!");
        }

        for (int i = 0; i < num; i++) {
            int key;
            DTYPE value;
            if (fscanf(file, "[%d]%e", &key, &value) < 3) {
                ShowNTErrors("Incorrect sparse tensor format!");
            }

            int ds[MAX_TENSOR_DIM_NUM];
            for (int i = 0; i < order; i++) {
                ds[i] = key % dimSize[i];
                key /= dimSize[i];
            }
            Set(value, ds);
        }
    }


    do {
        c = fgetc(file);
    } while (c != '\n' && c != EOF);

    XMemCopy(dataBackup, devID, data, -1, GetDataSizeInChar());
    data = dataBackup;

    delete[](char*)dataBuf;
}

/* 
read data from a binary file
>>> file - the file stream pointer
>>> offset - the distance from the start to this tensor
*/
void XTensor::BinaryRead(FILE* file, size_t offset)
{
    fseek(file, offset, 0);
    switch (dataType) {
    case X_INT: {
        int * d = new int[unitNum];
        fread(d, sizeof(int), unitNum, file);
        SetData(d, unitNum);
        delete[] d;
    }
    default: {
        float * d = new float[unitNum];
        fread(d, sizeof(float), unitNum, file);
        SetData(d, unitNum);
        delete[] d;
    }
    }
}

/*
flush the data to the target device
>> targetMem - memory pool on the target device
*/
void XTensor::FlushToMem(XMem * targetMem)
{
    if (targetMem == NULL)
        return;

    if (targetMem->devID >= 0) {
#ifdef USE_CUDA
        if (devID < 0) {
            TensorList l(1);
            l.Add(this);
            CudaCPUToGPUFlush(&l, targetMem->devID, targetMem);
        }
        else if (mem != targetMem) {
            void * tmpData = targetMem->Alloc(targetMem->devID, GetDataSizeInChar());
            XMemCopy(tmpData, targetMem->devID, data, devID, GetDataSizeInChar());
            data = tmpData;
            mem = targetMem;
            devID = mem->devID;
        }
#else
        ShowNTErrors("Recompile the code with USE_CUDA!");
#endif
    }
    else {
        if (devID >= 0) {
#ifdef USE_CUDA
            CudaGPUToCPUFlush(this);
            mem = targetMem;
            devID = mem->devID;
#else
            ShowNTErrors("Recompile the code with USE_CUDA!");
#endif
        }
    }
}

/*
allocate the memory space of the tensor (in the global memory) 
>> tensor - the tensor we intend to process
>> myMem - the memory pool we are using
>> useBuf - indicates whether we use the buffer in the memory pool
*/
void XTensor::AllocateData(XTensor * tensor, XMem * myMem, bool useBuf)
{
    if(tensor == NULL)
        return;

    if(myMem == NULL){
        if(tensor->data != NULL)
            FreeData(tensor, NULL, false);
        tensor->data = XMemAlloc(tensor->devID, tensor->GetDataSizeInChar());
        tensor->isInGlobalMem = true;
    }
    else{
        CheckNTErrors((tensor->data == NULL), "Cannot renew the space for the tensor");
        if(useBuf){
            tensor->data = myMem->AllocBuf(tensor->devID, tensor->GetDataSizeInChar());
            tensor->isInGlobalMem = false;
        }
        else{
            tensor->data = myMem->AllocGlobal(tensor->devID, tensor->GetDataSizeInChar());
            tensor->isInGlobalMem = true;
        }
    }

    tensor->signature = 0;
}

/* 
free the memory space of the tensor (in the global memory) 
>> tensor - the tensor we intend to process
>> myMem - the memory pool we are using
>> useBuf - indicates whether we use the buffer in the memory pool
*/
void XTensor::FreeData(XTensor * tensor, XMem * myMem, bool useBuf)
{
    if(tensor == NULL)
        return;

    if(myMem == NULL){
        XMemFree(tensor->devID, tensor->data);
    }
    else{
        if(tensor->isInGlobalMem)
            myMem->ReleaseGlobal(tensor->devID, tensor->data);
        else
            myMem->ReleaseBuf(tensor->devID, tensor->GetDataSizeInChar());
    }

    tensor->data = NULL;
    tensor->isInGlobalMem = false;
}

/* overloading of the plus-sign */
XTensor operator+ (const DTYPE shift, const XTensor &tensor) 
{
    return ScaleAndShift(tensor, 1, shift);
}

/* overloading of the minus-sign */
XTensor  operator- (const DTYPE shift, const XTensor &tensor)
{
    return ScaleAndShift(tensor, 1, -shift);
}

/* overloading of the multiply-sign */
XTensor  operator* (const DTYPE scale, const XTensor &tensor)
{
    return ScaleAndShift(tensor, scale, 0);
}

/* overloading of the division-sign */
XTensor  operator/ (const DTYPE scale, const XTensor &tensor)
{
    return ScaleAndShift(tensor, (DTYPE)1/scale, 0);
}

} /* end of the nts (NiuTrans.Tensor) namespace */
