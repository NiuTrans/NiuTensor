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

/* 
constructor 
>> myOrder - order of the tensor
>> myMem - memory pool used to allocating the data array
*/
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

    InitTensor(this, reference);
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
>> myDimSize - the size of each dimension
>> myDataType - unit size (e.g., int, float, and double)
>> myDenseRatio - how often an element has non-zero value
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
           As "reference" is constant, we cannot reset reference.data
           here. So we save the ADDRESS of reference.data in
           reference.dataP, and do this work by updating "*reference.dataP".
           This is VERY trick and might not be the best solution :) */
        *reference.dataP = NULL;
    }
    else{
        devID = reference.devID;
        mem = reference.mem;
        InitTensor(this, &reference);
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

/* initialize member variables */
void XTensor::Init()
{
    id = -1;
    mem = NULL;
    signature = 0;
    data = NULL;
    dataHost = NULL;
    dataP = NULL;
    devID = -1;
    order = -1;
    memset(dimSize, 0, sizeof(int) * MAX_TENSOR_DIM_NUM);
    memset(dimSizeRDI, 0, sizeof(int) * MAX_TENSOR_DIM_NUM);
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
shallow copy of tensor
Note that we do not copy data array here
>> tensor - the source tensor
*/
void XTensor::ShallowCopy(const XTensor &tensor)
{
    order = tensor.order;
    memcpy(dimSize, tensor.dimSize, sizeof(int) * MAX_TENSOR_DIM_NUM);
    memcpy(dimSizeRDI, tensor.dimSizeRDI, sizeof(int) * MAX_TENSOR_DIM_NUM);
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
        /* NOTE: this might lead to additional data copy on Mac machines */
        /* we make an identity transformation here */
        
        if(outgo.tailNum > 0)
            XLink::ClearOutgoing(this);
        XLink::ClearIncoming(this);
        
        if(!IsSameShaped(this, &tensor))
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
        XLink::Replace(&tensor, this);
    }

    return *this;
}

/* overloading of the plus-sign */
XTensor XTensor::operator+ (const XTensor& tensor)
{
    return Sum(*this, tensor);
}

/* overloading of the plus-sign */
XTensor XTensor::operator+ (const DTYPE shift)
{
    return ScaleAndShift(*this, 1, shift);
}

/* overloading of the multiply-sign */
XTensor XTensor::operator* (const XTensor& tensor)
{
    return Multiply(*this, tensor);
}

/* overloading of the multiply-sign */
XTensor XTensor::operator* (const DTYPE scale)
{
    return ScaleAndShift(*this, scale, 0);
}

/* overloading of the minus-sign */
XTensor XTensor::operator- (const XTensor& tensor)
{
    return Sub(*this, tensor);
}

/* overloading of the minus-sign */
XTensor XTensor::operator- (const DTYPE shift)
{
    return ScaleAndShift(*this, 1, -shift);
}

/* overloading of the division-sign */
XTensor XTensor::operator/ (const XTensor& tensor)
{
    return Div(*this, tensor);
}

/* overloading of the division-sign */
XTensor XTensor::operator/ (const DTYPE scale)
{
    return ScaleAndShift(*this, (DTYPE)1/scale, 0);
}

/* 
linear transformation b = a * \scale + \shift
>> scale - the slope
>> shift - the intercept
*/
XTensor XTensor::Lin(DTYPE scale, DTYPE shift)
{
    return Linear(*this, scale, shift);
}

/* 
judge whether the two matrices are in the same type and size 
>> a - input tensor
>> b - anther tensor to compare with
<< return - whether the two input tensors are identical
*/
bool XTensor::IsSameShaped(const XTensor * a, const XTensor * b)
{
    if(a == NULL || b == NULL)
        return false;

    if(a->order != b->order)
        return false;

    for(int i = 0; i < a->order; i++){
        if(a->dimSizeRDI[i] != b->dimSizeRDI[i])
            return false;
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
judge whether the three matrices are in the same type and size 
>> a - input tensor
>> b - anther tensor to compare with
>> c - a tensor again
<< return - whether the two input tensors are identical
*/
bool XTensor::IsSameShaped(const XTensor * a, const XTensor * b, const XTensor * c)
{
    return IsSameShaped(a, b) && IsSameShaped(a, c);
}

/* 
set the size of each dimension 
>> myDimSize - size of each dimension
*/
void XTensor::SetDim(int * myDimSize)
{
    for (int i = 0; i < order; i++) {
        dimSize[i] = myDimSize[i];
        dimSizeRDI[order - i - 1] = myDimSize[i];
    }
}

/* 
get the size of a given dimension 
>> dim - the given dim we are looking at
*/
int XTensor::GetDim(const int dim) const
{
    CheckNTErrors(dim < order, "dimenision is out of range!");
    
    int d = dim;
    if(dim < 0)
        d = order - 1;

    return dimSize[d];
}

/* 
reshape the tensor 
>> myOrder - order of the tensor
>> myDimSize - the size of each dimension
*/
void XTensor::Reshape(const int myOrder, const int * myDimSize)
{
    int dims[MAX_TENSOR_DIM_NUM];
    int dimsRDI[MAX_TENSOR_DIM_NUM];
    int num = 1;

    for(int i = 0; i < myOrder; i++){
        num *= myDimSize[i];
        dims[i] = abs(myDimSize[i]);
        dimsRDI[myOrder - i - 1] = dims[i];
    }

    CheckNTErrors(abs(num) == unitNum, "Wrong size found when we reshape the tensor!");

    order = myOrder;
    memcpy(dimSize, dims, sizeof(int) * order);
    memcpy(dimSizeRDI, dimsRDI, sizeof(int) * order);
}

/* 
reshape the tensor to a vector 
>> num - number of elements
*/
void XTensor::Reshape(const int num)
{
    int dim = num;
    Reshape(1, &dim);
}

/* 
reshape the tensor to a matrix 
>> rowNum - number of rows
>> colNum - number of columns
*/
void XTensor::Reshape(const int rowNum, const int colNum)
{
    int dims[2] = {rowNum, colNum};
    Reshape(2, dims);
}

/* get the number of items in the data array */
int XTensor::GetSize() const
{
    if(isSparse)
        return unitNumNonZero;
    else
        return unitNum;
}

/* get size of the memory used */
int XTensor::GetDataSizeInChar()
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
int XTensor::GetUnitSize(TENSOR_DATA_TYPE myDataType)
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
MTYPE XTensor::GetOffset2D(int row, int col)
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
MTYPE XTensor::GetOffset3D(int d0, int d1, int d2)
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
>> beg - where we start this in the data array of the tensor
*/
void XTensor::SetData(const void * d, int num, int beg)
{
    if (data == NULL || d ==NULL)
        return;

    CheckNTErrors(!isSparse, "TODO");
    CheckNTErrors(num <= unitNum - beg, "Illegal size!");

    XMemCopy((char*)data + beg * unitSize, devID, d, -1, num * unitSize);
}

/* 
set the tensor items by a uniform distribution in range [lower, upper]
>> lower - lower value of the range
>> upper - upper value of the range
*/
void XTensor::SetDataRand(DTYPE lower, DTYPE upper)
{
    // TODO: cuda code!!!!!!!

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
    // TODO: cuda code!!!!!!!

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
>> value - value for data items
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

/* check whether the data array is the same as the answer
>> d - input data. it must be on CPU
>> num - number of data items
>> beg - where we start this in the data array of the tensor
*/
bool XTensor::CheckData(const void * d, int num, int beg)
{
    if (data == NULL || d == NULL)
        return false;

    CheckNTErrors(!isSparse, "TODO");
    CheckNTErrors(num == unitNum - beg, "Illegal size!");

    if (devID < 0) {
        return !memcmp(data, d, num * unitSize);
    }
#ifdef USE_CUDA
    else {
        char * copy = new char[num * unitSize];
        XMemCopy(copy, -1, data, devID, num * unitSize);
        int cmpResult = memcmp(copy, d, num * unitSize);
        bool result = (cmpResult == 0) ? true : false;
        delete[] copy;
        return result;
    }
#endif
    return true;
}
    
/* set the pointer to "data" */
void XTensor::SetDataPointer()
{
    dataP = &data;
}

/* compare two number */
bool IsFloatEqual(DTYPE a, DTYPE b, float absError, float relError)
{
    if(a == b)
        return true;
    if(fabs(a - b) < absError)
        return true;
    if(fabs(a) < fabs(b))
        return (fabs((a - b) / b) < relError) ? true : false;
    else
        return (fabs((a - b) / a) < relError) ? true : false;
}

/* check whether the data array is the same as the answer */
bool XTensor::CheckData(const void * d, int num, float tolerance, int beg)
{
    if (data == NULL || d == NULL)
        return false;

    CheckNTErrors(!isSparse, "TODO");
    CheckNTErrors(num == unitNum - beg, "Illegal size!");

    DTYPE * valuePrt = (DTYPE*)data;
    DTYPE value = 0;
    DTYPE * answerPrt = (DTYPE*)d;
    for (int i = beg; i < num; i++) {
        value = ToCPU(devID, valuePrt);
        if(IsFloatEqual(value, *answerPrt, tolerance, 1e-4F) == false)
            return false;
        valuePrt++;
        answerPrt++;
    }
    return true;
}

/* 
set the cell to the ascending order along a given dimension 
>> dim - the dimension specified
*/
void XTensor::SetAscendingOrder(int dim)
{
    CheckNTErrors((dim >= 0 && dim < order), "Wrong dimension specified!");
    CheckNTErrors((dataType == X_INT), "TODO!");

    int dimRDI = order - dim - 1;
    if(devID >= 0){
#ifdef USE_CUDA
        CudaSetAscendingOrder(this, dim);
#else
        ShowNTErrors("Plesae specify USE_CUDA and recompile the code!");
#endif
    }
    else{
        int stride = 1;
        int strideNum = dimSizeRDI[dimRDI];
        for(int i = 0; i < dimRDI; i++)
            stride *= dimSizeRDI[i];

        int blockNum = 1;
        for(int i = dimRDI + 1; i < order; i++)
            blockNum *= dimSizeRDI[i];

        for(int k = 0; k < blockNum; k++){
            for(int j = 0; j < strideNum; j++){
                int * d = (int*)data + stride * strideNum * k + stride * j;
                for(int i = 0; i < stride; i++)
                    d[i] = j;
            }
        }
    }
}

/* 
get the value of a cell with the index 
>> index - index of each dimension
>> size - size of index
<< return - cell value
*/
DTYPE XTensor::Get(int index[], int size)
{
    CheckNTErrors((dataType == DEFAULT_DTYPE), "The tensor is not in default type.");

    return ToCPU(devID, GetCell(index, size));
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

    int * indexRDI = new int[size];
    for (int i = 0; i < size; i++)
        indexRDI[size - i - 1] = index[i];

    int offset = indexRDI[size - 1];
    for(int i = size - 2; i >= 0; i--){
        CheckNTErrors((indexRDI[i] < dimSizeRDI[i]), "Index is out of range!");
        offset = offset * dimSizeRDI[i] + indexRDI[i];
    }
    
    delete[] indexRDI;

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
DTYPE XTensor::Get1D(int i)
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
DTYPE XTensor::Get3D(int d0, int d1, int d2)
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
get the value of a cell in a 1d tensor in int type
>> i - index
<< return - value of cell(i) in int
*/
int XTensor::Get1DInt(int i)
{
    CheckNTErrors((order == 1), "Cannot get a 2d cell for a tensor whose order is not 2!");
    CheckNTErrors((i >= 0 && i < dimSize[0]), "dimension 0 is out of range!");
    CheckNTErrors((dataType == X_INT), "The tensor is not in int type.");
    
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
 int XTensor::Get2DInt(int ni, int mi)
{
    CheckNTErrors((order == 2), "Cannot get a 2d cell for a tensor whose order is not 2!");
    CheckNTErrors((ni >= 0 && ni < dimSize[0]), "dimension 0 is out of range!");
    CheckNTErrors((mi >= 0 && mi < dimSize[1]), "dimension 1 is out of range!");
    CheckNTErrors((dataType == X_INT), "The tensor is not in default type.");

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
int XTensor::Get3DInt(int d0, int d1, int d2)
{
    CheckNTErrors((order == 3), "Cannot get a 2d cell for a tensor whose order is not 2!");
    CheckNTErrors((d0 >= 0 && d0 < dimSize[0]), "dimension 0 is out of range!");
    CheckNTErrors((d1 >= 0 && d1 < dimSize[1]), "dimension 1 is out of range!");
    CheckNTErrors((d2 >= 0 && d2 < dimSize[2]), "dimension 2 is out of range!");
    CheckNTErrors((dataType == X_INT), "The tensor is not in default type.");

    int dims[3] = {d0, d1, d2};
    void * value = GetCell(dims, 3);
    
    return ToCPUInt(devID, value);
}

/* 
get the value of a cell in the sparse tensor 
>> i - i-th tuple in the tuple list of the sparse tensor
<< return - value of the tuple
*/
DTYPE XTensor::GetInSparse(int i)
{
    CheckNTErrors((i >= 0 && i < unitNum), "Index is out of range!");
    CheckNTErrors((dataType == DEFAULT_DTYPE), "The tensor is not in default type.");

    char * d = (char*)data + sizeof(int);
    DTYPE * value = (DTYPE*)(d + (sizeof(int) + sizeof(DTYPE)) * i + sizeof(int));

    return ToCPU(devID, value);
}

/* 
get the key value of a tuple in a sparse tensor 
>> i - i-th tuple in the tuple list of the sparse tensor
<< return - key of the tuple
*/
int XTensor::GetKeyInSparse(int i)
{
    CheckNTErrors((i >= 0 && i < unitNum), "Index is out of range!");
    CheckNTErrors((dataType == DEFAULT_DTYPE), "The tensor is not in default type.");

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
	CheckNTErrors((dataType == DEFAULT_DTYPE), "The tensor is not in default type.");

    return SetToDevice(devID, GetCell(index, size), value);
}

/* 
set the value of a cell in a 1d tensor 
>> value - value we tend to set
>> i - item offset
<< return - succeeded or not
*/
bool XTensor::Set1D(DTYPE value, int i)
{
    CheckNTErrors((order == 1), "Cannot get a 2d cell for a tensor whose order is not 2!");
    CheckNTErrors((i >= 0 && i < dimSize[0]), "dimension 0 is out of range!");
    CheckNTErrors((dataType == DEFAULT_DTYPE), "The tensor is not in default type.");

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
    CheckNTErrors((order == 2), "Cannot get a 2d cell for a tensor whose order is not 2!");
    CheckNTErrors((ni >= 0 && ni < dimSize[0]), "dimension 0 is out of range!");
    CheckNTErrors((mi >= 0 && mi < dimSize[1]), "dimension 1 is out of range!");
    CheckNTErrors((dataType == DEFAULT_DTYPE), "The tensor is not in default type.");

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
set the integer value of a cell 
>> value - value we tend to set
>> index - index of the cell for each dimension
>> size - size of the index
<< return - succeeded or not
*/
bool XTensor::SetInt(int value, int index[], int size)
{
    CheckNTErrors((dataType == X_INT), "The tensor is not in integer type.");

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
    CheckNTErrors((order == 1), "Cannot get a 2d cell for a tensor whose order is not 2!");
    CheckNTErrors((i >= 0 && i < dimSize[0]), "dimension 0 is out of range!");
    CheckNTErrors((dataType == X_INT), "The tensor is not in integer type.");

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
    CheckNTErrors((order == 2), "Cannot get a 2d cell for a tensor whose order is not 2!");
    CheckNTErrors((ni >= 0 && ni < dimSize[0]), "dimension 0 is out of range!");
    CheckNTErrors((mi >= 0 && mi < dimSize[1]), "dimension 1 is out of range!");
    CheckNTErrors((dataType == X_INT), "The tensor is not in integer type.");

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
    CheckNTErrors((ni >= 0 && ni < dimSize[0]), "the row index is out of range!");
    CheckNTErrors((mi >= 0 && mi < dimSize[1]), "the column index is out of range!");
    CheckNTErrors((dataType == DEFAULT_DTYPE), "The tensor is not in default type.");
    CheckNTErrors((isSparse == false), "TODO!");

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
int XTensor::GetNonzeroSize()
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
        dimSizeRDI[order - i - 1] = dimSize[i];
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
resize a tensor by another one 
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
<< return - find it or not?
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
                num *= dimSizeRDI[i];
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
>> tensor - tensor whose data is dumped
>> file - where to domp the data
>> label - label of the tensor
>> n - number of items to dump
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
                ds[i] = key % dimSizeRDI[i];
                key /= dimSizeRDI[i];
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
            XList l(1);
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
>> useBuf - use the buffer in the memory pool
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
>> useBuf - use the buffer in the memory pool
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

/*************************************************
* we define the "new and delete" functions below
*/

/* 
initialize a tensor 
>> tensor - the tensor we intend to initialize
>> myOrder - order of the tensor
>> myDimSize - the size of each dimension
>> myDataType - unit size (e.g., int, float, and double)
>> myDenseRatio - how often an element has non-zero value
>> myDevID - when myMem is NULL, myDevID specifies the device 
             on which we allocate the data on site
>> myMem - memory pool used to allocating the data array
           myMem = NULL means that the tensor is allocated on
           the device dynamically, rather than on the memory pool
*/

void InitTensor(XTensor * tensor,
                const int myOrder, const int * myDimSize, const TENSOR_DATA_TYPE myDataType,
                const float myDenseRatio, const int myDevID, XMem * myMem)
{
    if(myMem != NULL && tensor->mem == NULL){
        tensor->mem = myMem;
        tensor->devID = myMem->devID;
    }

    if(tensor->mem != NULL){
        tensor->Resize(myOrder, myDimSize, myDataType, myDenseRatio);
    }
    else{
        int dims[MAX_TENSOR_DIM_NUM];
        memcpy(dims, myDimSize, sizeof(int) * myOrder);

        bool allocated = true;
        for (int i = 0; i < myOrder; i++) {
            if (dims[i] < 0)
                allocated = false;
        }

        dims[0] = -abs(dims[0]);

        if (myDevID == CURRENT_GPU)
            tensor->devID = XDevice::GetGPUDevice();
        else
            tensor->devID = myDevID;

        tensor->Resize(myOrder, dims, myDataType, myDenseRatio);

        if(allocated)
            XTensor::AllocateData(tensor);
    }
}

/* 
initialize a dense tensor 
>> tensor - the tensor we intend to initialize
>> num - number of elements
>> myDataType - unit size (e.g., int, float, and double) 
>> myDevID - when myMem is NULL, myDevID specifies the device 
             on which we allocate the data on site
>> myMem - memory pool used to allocating the data array
           myMem = NULL means that the tensor is allocated on
           the device dynamically, rather than on the memory pool
*/

void InitTensor1D(XTensor * tensor, const int num,
                  const TENSOR_DATA_TYPE myDataType, const int myDevID, XMem * myMem)
{
    int dims[1];
    dims[0] = num;

    InitTensor(tensor, 1, dims, myDataType, 1.0F, myDevID, myMem);
}

/* 
initialize a dense matrix 
>> tensor - the tensor we intend to initialize
>> rowNum - number of rows
>> colNum - number of columns
>> myDataType - unit size (e.g., int, float, and double) 
>> myDevID - when myMem is NULL, myDevID specifies the device 
             on which we allocate the data on site
>> myMem - memory pool used to allocating the data array
           myMem = NULL means that the tensor is allocated on
           the device dynamically, rather than on the memory pool
*/

void InitTensor2D(XTensor * tensor, const int rowNum, const int colNum,
                  const TENSOR_DATA_TYPE myDataType, const int myDevID, XMem * myMem)
{
    int dims[2];
    dims[0] = rowNum;
    dims[1] = colNum;

    InitTensor(tensor, 2, dims, myDataType, 1.0F, myDevID, myMem);
}

/* 
initialize a dense 3d tensor 
>> tensor - the tensor we intend to initialize
>> d0 - size of dimension 0
>> d1 - size of dimension 1
>> d2 - size of dimension 2
>> myDataType - unit size (e.g., int, float, and double) 
>> myDevID - when myMem is NULL, myDevID specifies the device 
             on which we allocate the data on site
>> myMem - memory pool used to allocating the data array
           myMem = NULL means that the tensor is allocated on
           the device dynamically, rather than on the memory pool
*/

void InitTensor3D(XTensor * tensor, const int d0, const int d1, const int d2, 
                  const TENSOR_DATA_TYPE myDataType, const int myDevID, XMem * myMem)
{
    int dims[3];
    dims[0] = d0;
    dims[1] = d1;
    dims[2] = d2;

    InitTensor(tensor, 3, dims, myDataType, 1.0F, myDevID, myMem);
}
    
/*
initialize a dense 4d tensor
>> tensor - the tensor we intend to initialize
>> d0 - size of dimension 0
>> d1 - size of dimension 1
>> d2 - size of dimension 2
>> d3 - size of dimension 3
>> myDataType - unit size (e.g., int, float, and double)
>> myDevID - when myMem is NULL, myDevID specifies the device
             on which we allocate the data on site
>> myMem - memory pool used to allocating the data array
           myMem = NULL means that the tensor is allocated on
           the device dynamically, rather than on the memory pool
*/

void InitTensor4D(XTensor * tensor, const int d0, const int d1, const int d2, const int d3,
                  const TENSOR_DATA_TYPE myDataType, const int myDevID, XMem * myMem)
{
    int dims[4];
    dims[0] = d0;
    dims[1] = d1;
    dims[2] = d2;
    dims[3] = d3;
    
    InitTensor(tensor, 4, dims, myDataType, 1.0F, myDevID, myMem);
}
    
/*
initialize a dense 5d tensor
>> tensor - the tensor we intend to initialize
>> d0 - size of dimension 0
>> d1 - size of dimension 1
>> d2 - size of dimension 2
>> d3 - size of dimension 3
>> d4 - size of dimension 4
>> myDataType - unit size (e.g., int, float, and double)
>> myDevID - when myMem is NULL, myDevID specifies the device
             on which we allocate the data on site
>> myMem - memory pool used to allocating the data array
           myMem = NULL means that the tensor is allocated on
           the device dynamically, rather than on the memory pool
*/

void InitTensor5D(XTensor * tensor, const int d0, const int d1, const int d2, const int d3, const int d4,
                  const TENSOR_DATA_TYPE myDataType, const int myDevID, XMem * myMem)
{
    int dims[5];
    dims[0] = d0;
    dims[1] = d1;
    dims[2] = d2;
    dims[3] = d3;
    dims[4] = d4;
    
    InitTensor(tensor, 5, dims, myDataType, 1.0F, myDevID, myMem);
}

/* 
initialize a tensor with a reference tensor 
>> tensor - the tensor we intend to initialize
>> reference - the reference tensor
*/
void InitTensor(XTensor * tensor, const XTensor * reference)
{
    if(reference->order < 0)
        return;

    InitTensor(tensor, reference->order, reference->dimSize, 
               reference->dataType, reference->denseRatio, 
               reference->devID, reference->mem);
}

/* 
generate a XTensor 
>> myOrder - order of the tensor
>> myDimSize - the size of each dimension
>> myDataType - unit size (e.g., int, float, and double)
>> myDenseRatio - how often an element has non-zero value
>> myDevID - when myMem is NULL, myDevID specifies the device 
             on which we allocate the data on site
>> myMem - memory pool used to allocating the data array
           myMem = NULL means that the tensor is allocated on
           the device dynamically, rather than on the memory pool.
*/

XTensor * NewTensor(const int myOrder, const int * myDimSize, const TENSOR_DATA_TYPE myDataType,
                    const float myDenseRatio, const int myDevID, XMem * myMem)
{
    if(myMem != NULL)
        return new XTensor(myOrder, myDimSize, myDataType, myDenseRatio, myDevID, myMem);
    else{
        XTensor * tensor = new XTensor();
        InitTensor(tensor, myOrder, myDimSize, myDataType, myDenseRatio, myDevID, myMem);
        return tensor;
    }
}

/*
generate a XTensor which allocates data on the buffer 
>> myOrder - order of the tensor
>> myDimSize - the size of each dimension
>> myMem - memory pool used to allocating the data array.
           we actually allocate the data on the buffer associated with
           the memory pool
>> devID - device id
>> myDataType - unit size (e.g., int, float, and double)
>> myDenseRatio - how often an element has non-zero value

*/
XTensor * NewTensorBuf(const int myOrder, const int * myDimSize,
                       const TENSOR_DATA_TYPE myDataType, const float myDenseRatio,
                       const int devID, XMem * myMem)
{
    int dims[MAX_TENSOR_DIM_NUM];
    memcpy(dims, myDimSize, sizeof(int) * myOrder);

    dims[0] = -abs(dims[0]);

    XTensor * tensor = NewTensor(myOrder, dims, myDataType, myDenseRatio, devID, myMem);

    if (tensor->unitNum * tensor->unitSize == 176657664) {
        tensor->Dump(stderr, "", 200);
    }
    if(myMem != NULL)
        tensor->data = myMem->AllocBuf(myMem->devID, tensor->unitNum * tensor->unitSize);
    else
        tensor->data = XMemAlloc(devID, tensor->unitNum * tensor->unitSize);

    return tensor;
}

/* 
generate a XTensor which allocates data on the buffer 
>> reference - reference tensor
>> devID - device id
>> myMem - memory pool used to allocating the data array.
           we actually allocate the data on the buffer associated with
           the memory pool
*/
XTensor * NewTensorBuf(const XTensor * reference, int devID, XMem * myMem)
{
    return NewTensorBuf(reference->order, reference->dimSize, 
                        reference->dataType, reference->denseRatio,
                        devID, myMem);
}

/* 
generate a dense vector 
>> num - number of entries
>> myDataType - unit size (e.g., int, float, and double) 
>> myDevID - when myMem is NULL, myDevID specifies the device 
             on which we allocate the data on site
>> myMem - memory pool used to allocating the data array
           myMem = NULL means that the tensor is allocated on
           the device dynamically, rather than on the memory pool.
*/

XTensor * NewTensor1D(const int num, 
                      const TENSOR_DATA_TYPE myDataType, const int myDevID, XMem * myMem)
{
    int dims[1];
    dims[0] = num;

    return NewTensor(1, dims, myDataType, 1.0F, myDevID, myMem);
}

/* 
generate a dense matrix
>> rowNum - number of rows
>> colNum - number of colums
>> myDataType - unit size (e.g., int, float, and double) 
>> myDevID - when myMem is NULL, myDevID specifies the device 
             on which we allocate the data on site
>> myMem - memory pool used to allocating the data array
           myMem = NULL means that the tensor is allocated on
           the device dynamically, rather than on the memory pool.
*/

XTensor * NewTensor2D(const int rowNum, const int colNum,
                      const TENSOR_DATA_TYPE myDataType, const int myDevID, XMem * myMem)
{
    int dims[2];
    dims[0] = rowNum;
    dims[1] = colNum;

    return NewTensor(2, dims, myDataType, 1.0F, myDevID, myMem);
}

/* 
generate a dense 3d tensor 
>> d0 - size of dimension 0
>> d1 - size of dimension 1
>> d2 - size of dimension 2
>> myDataType - unit size (e.g., int, float, and double) 
>> myDevID - when myMem is NULL, myDevID specifies the device 
             on which we allocate the data on site
>> myMem - memory pool used to allocating the data array
           myMem = NULL means that the tensor is allocated on
           the device dynamically, rather than on the memory pool.
*/

XTensor * NewTensor3D(const int d0, const int d1, const int d2,
                      const TENSOR_DATA_TYPE myDataType, const int myDevID, XMem * myMem)
{
    int dims[3];
    dims[0] = d0;
    dims[1] = d1;
    dims[2] = d2;

    return NewTensor(3, dims, myDataType, 1.0F, myDevID, myMem);
}

/* 
generate a dense 4d tensor 
>> d0 - size of dimension 0
>> d1 - size of dimension 1
>> d2 - size of dimension 2
>> d3 - size of dimension 3
>> myDataType - unit size (e.g., int, float, and double) 
>> myDevID - when myMem is NULL, myDevID specifies the device 
             on which we allocate the data on site
>> myMem - memory pool used to allocating the data array
           myMem = NULL means that the tensor is allocated on
           the device dynamically, rather than on the memory pool.
*/

XTensor * NewTensor4D(const int d0, const int d1, const int d2, const int d3,
                      const TENSOR_DATA_TYPE myDataType, const int myDevID, XMem * myMem)
{
    int dims[4];
    dims[0] = d0;
    dims[1] = d1;
    dims[2] = d2;
    dims[3] = d3;

    return NewTensor(4, dims, myDataType, 1.0F, myDevID, myMem);
}

/* 
generate a dense 5d tensor 
>> d0 - size of dimension 0
>> d1 - size of dimension 1
>> d2 - size of dimension 2
>> d3 - size of dimension 3
>> d4 - size of dimension 4
>> myDataType - unit size (e.g., int, float, and double) 
>> myDevID - when myMem is NULL, myDevID specifies the device 
             on which we allocate the data on site
>> myMem - memory pool used to allocating the data array
           myMem = NULL means that the tensor is allocated on
           the device dynamically, rather than on the memory pool.
*/

XTensor * NewTensor5D(const int d0, const int d1, const int d2, const int d3, const int d4,
                      const TENSOR_DATA_TYPE myDataType, const int myDevID, XMem * myMem)
{
    int dims[5];
    dims[0] = d0;
    dims[1] = d1;
    dims[2] = d2;
    dims[3] = d3;
    dims[4] = d4;

    return NewTensor(5, dims, myDataType, 1.0F, myDevID, myMem);
}

/* 
generate a copy of XTensor 
>> a - the tensor we copy from
>> isFilledData - indicates whether we allocate the data for
                  the newly-generated tensor
*/
XTensor * NewTensor(const XTensor * a, bool isFilledData)
{
    int dims[MAX_TENSOR_DIM_NUM];
    
    CheckNTErrors((a != NULL), "Empty input!");
    
    memset(dims, 0, sizeof(int) * MAX_TENSOR_DIM_NUM);

    if(a->order > 0)
        memcpy(dims, a->dimSize, sizeof(int) * a->order);

    if(!isFilledData)
        dims[0] = -dims[0];

    XTensor * newTensor = new XTensor(a->order, dims,
                                      a->dataType, a->denseRatio,
                                      a->devID, a->mem);

    return newTensor;

}

/* 
free the data space of a given tensor 
>> tensor - pointer to the tensor
*/
void DelTensor(XTensor * tensor)
{
    delete tensor;
}

/* 
free the data space of a given tensor (on the buffer)
>> tensor - pointer to the tensor
*/
void DelTensorBuf(XTensor * tensor)
{
    if(tensor->mem != NULL)
        tensor->mem->ReleaseBuf(tensor->devID, tensor->unitNum * tensor->unitSize);
    else
        XMemFree(tensor->devID, tensor->data);
    tensor->data = NULL;
    delete tensor;
}

} /* end of the nts (NiuTrans.Tensor) namespace */
