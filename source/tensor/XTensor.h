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
 * the tensor class
 *
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2017-07-31
 * I'm working while most of the students are enjoying their holidays :(
 * $Updated by: LI Yinqiao (li.yin.qiao.2012@hotmail.com) 2017-11-18 bug fixes
 *
 */

#ifndef __XTENSOR_H__
#define __XTENSOR_H__

#include <math.h>
#include "XGlobal.h"
#include "XMem.h"
#include "XPRunner.h"
#include "XStream.h"
#include "XHeap.h"
#include "XList.h"
#include "XDataType.h"
#include "XMem.h"
#include "XLink.h"
#include "XCall.h"

/* the nts (NiuTrans.Tensor) namespace */
namespace nts{

/* cross reference */
struct XLink;

/* define the maximum number of dimensions in a tensor */
#define MAX_TENSOR_DIM_NUM 8
#define USE_BATCHED_STRIDED_MAT_MUL
#define MIN_TENSOR_SPLIT_NUM 0
#define MIN_TENSOR_SPLIT_LIST_NUM 1024
#define MIN_TENSOR_MERGE_NUM 0
#define MIN_TENSOR_MERGE_LIST_NUM 1024
#define MIN_TENSOR_CAT_NUM 8
#define MAX_TENSOR_NAME_SIZE 32

/* computation flags */
#define UNSAFE_BUT_FAST_MEM
#define FAST_MATRIX

/* XTensor is a class to do everything a tensor can do :) */
struct XTensor
{
public:
    /* name */
    char name[MAX_TENSOR_NAME_SIZE];

    /* id */
    int id;

    /* memory pool */
    XMem * mem;

    /* signature of the memory pool */
    MTYPE signature;

    /* data array to keep the elements */
    void * data;

    /* copy of data on the host memory. It is only activated 
       when the tensor is operated on GPUs */
    void * dataHost;
    
    /* a pointer to data (i.e., a pointer to the address of "data".
       This is for reset "data" when XTensor is used as a const variable. */
    void ** dataP;

    /* 
    device id 
    <0:  CPU memory
    >=0: GPU device ID
    */
    int devID;

    /* order of the tensor. e.g., a matrix (a 2-dimensional array) 
       is a 2nd-order tensor */
    int order;

    /* size of each dimension */
    int dimSize[MAX_TENSOR_DIM_NUM];

    /* data unit - data type for every cell */
    TENSOR_DATA_TYPE dataType;

    /* size of tensor unit, e.g., sizeof(int) */
    int unitSize;

    /* number of units */
    int unitNum;

    /*
    if it is a sparse matrix
    dense matrix:  there are n * m entries - i.e.,
                   the size of "data" is n * m
    sparse matrix: number of entries depends on how
                   many entries are non-zero
    */
    bool isSparse;

    /* nubmer of non-zero items in a sparse matrix */
    int unitNumNonZero;

    /*
    denseRatio - how dense the matrix is
    denseRatio = 1: a dense matrix
    denseRatio < 1: how often an element has a non-zero value
    */
    float denseRatio;

    /* indicates whether the data array is shared with other tensors */
    bool isShared;

    /* indicates whether the date type used in this tensor is in default type (i.e., DTYPE) */
    bool isDefaultDType;

    /* indicates whether the data is allocated in the global memory rather than a memory pool */
    bool isInGlobalMem;

    /* indicates whether the SPARSE tensor has non-zero values for all entries alone each dimension */
    bool isAllValued[MAX_TENSOR_DIM_NUM];

    /* indicates whether the tensor is initialized or not */
    bool isInit;

    /* indicates whether the tensor is created temporarily */
    bool isTmp;

    /* indicates whether the tensor keeps the gradient when used as model parameters */
    bool isGrad;

    /* indicates whether the gradient of the tensor should be computed */
    bool enableGrad;

    /* indicates whether the tensor is used as paramters (or variables) */
    bool isVar;

    /* mark for traversing the gragh */
    unsigned int visitMark;

    /* gradient (for back-propagation) */
    XTensor * grad;
    
    /*
    the link used to form networks. Note that when we compute on tensors, we actually create a
    network where nodes are tensors and edges the connections among them. Each connection is
    a hyperedge whose head is the output tensor and tails are input tensors. E.g,
    c = a + b
    represents a network with three nodes (a, b and c) and a hyperedge that links a and b (tails) to c (head).
    Here "income" keeps which nodes (tensors) are used to form the current node (tensor).
    */
    XLink income;
    
    /* It keeps which nodes (tensors) we go to from the current node (tensor). */
    XLink outgo;

    /********************
     XTensor untilities
    ********************/
    
    /* constructor */
    XTensor();

    /* constructor */
    XTensor(const XTensor * reference);

    /* constructor */
    XTensor(const int myOrder, int myDevID, XMem * myMem);

    /* constructor */
    XTensor(const int myOrder, const int * myDimSize, const TENSOR_DATA_TYPE myDataType, 
            const float myDenseRatio, int myDevID, XMem * myMem);

    /* copy constructor */
    XTensor(const XTensor &reference);

    /* copy constructor (with right value reference) */
    XTensor(const XTensor &&reference);

    /* de-constructor */
    ~XTensor();

    /* set the name of the tensor */
    void SetName(const char * myName);

    /* initialize member variables */
    void Init();

    /* delete data arrays */
    void DestroyData();

    /* shallow copy of tensor */
    void ShallowCopy(const XTensor &tensor);

    /* overloading of the equal-sign */
    XTensor& operator= (const XTensor &tensor);

    /* overloading of the equal-sign (with right value reference) */
    XTensor& operator= (const XTensor &&tensor);

    /* overloading of the plus-sign */
    XTensor  operator+ (const XTensor &tensor) const;
    
    /* overloading of the plus-sign */
    XTensor  operator+ (const DTYPE shift) const;

    /* overloading of the multiply-sign */
    XTensor  operator* (const XTensor &tensor) const;
    
    /* overloading of the multiply-sign */
    XTensor  operator* (const DTYPE scale) const;

    /* overloading of the minus-sign */
    XTensor  operator- (const XTensor &tensor) const;
    
    /* overloading of the minus-sign */
    XTensor  operator- (const DTYPE shift) const;

    /* overloading of the minus-sign */
    XTensor  operator- () const;

    /* overloading of the division-sign */
    XTensor  operator/ (const XTensor &tensor) const;
    
    /* overloading of the division-sign */
    XTensor  operator/ (const DTYPE scale) const;

    /* linear transformation */
    XTensor Lin(DTYPE scale, DTYPE shift = 0) const;

    /* relocate the data on the target device */
    void SetDevice(int myDevId, XMem * myMem = NULL);

    /* judge whether b is the reduced shape of a ?? */
    static
    bool IsReduceShaped(const XTensor * a, const XTensor * b, int dim);

    /* set the size of each dimension */
    void SetDim(int * myDimSize);

    /* get the size of a given dimension */
    int GetDim(const int dim) const;

    /* reshape the tensor */
    void Reshape(const int order, const int * myDimSize);

    /* reshape the tensor to a vector */
    void Reshape(const int num);

    /* reshape the tensor to a matrix */
    void Reshape(const int rowNum, const int colNum);

    /* reshape the tensor by merging two consecutive dimensions */
    void ReshapeMerged(const int i, const int j = -1);

    /* return a tensor that datatype is same as the special tensor */
    XTensor TypeAs(const XTensor input);

    /* get the number of items in the data array */
    int GetSize() const;

    /* get size of the memory used */
    int GetDataSizeInChar() const;

    /* get unit size in terms of "dataType" */
    int GetUnitSize(TENSOR_DATA_TYPE myDataType) const;

    /* get offset (2D) */
    MTYPE GetOffset2D(int row, int col) const;

    /* get offset (3D) */
    MTYPE GetOffset3D(int d0, int d1, int d2) const;

    /* a tensor with all entries of 0 */
    void SetZeroAll(XStream * stream = NULL);

    /* set the tensor with an data array */
    void SetData(const void * d, int num, int beg = 0);

    /* generate data items with a uniform distribution in [0, 1] */
    void Rand(int rNum, int cNum);

    /* generate data items with a range by start, end and the step */
    void Range(DTYPE lower, DTYPE upper, DTYPE step);

    /* set tensor items by a uniform distribution */
    void SetDataRand(DTYPE lower = 0.0F, DTYPE upper = 1.0F);

    /* set tensor items by a normal distribution */
    void SetDataRandn(DTYPE mean, DTYPE standardDeviation);

    /* set tensor items with an array of offsets */
    void SetDataBatched(MTYPE * offsets, DTYPE value, int num);

    /* set tensor items with an array of values */
    void SetDataBatchedWithValues(MTYPE * offsets, void * values, int num);
    
    /* set the pointer to "data" */
    void SetDataPointer();

    /* get the value of a cell with the index */
    DTYPE Get(int index[], int size = -1) const;
    
    /* get the value of a cell with the offset */
    DTYPE Get(int offset) const;

    /* get the pointer to a cell */
    void * GetCell(int index[], int size = -1) const;

    /* get the default type value of a cell in a 1d tensor */
    DTYPE Get1D(int i) const;

    /* get the default type value of a cell in a 2d tensor */
    DTYPE Get2D(int ni, int mi) const;
    
    /* get the default type value of a cell in a 3d tensor */
    DTYPE Get3D(int d0, int d1, int d2) const;

    /* get the int value of a cell by its offset */
    int GetInt(int offset) const;
    
    /* get the int value of a cell in a 1d tensor */
    int Get1DInt(int i) const;

    /* get the int value of a cell in a 2d tensor */
    int Get2DInt(int ni, int mi) const;
    
    /* get the int value of a cell in a 3d tensor */
    int Get3DInt(int d0, int d1, int d2) const;

    /* get the value of a cell in a sparse tensor */
    DTYPE GetInSparse(int i) const;

    /* get the key value of a tuple in a sparse tensor */
    int GetKeyInSparse(int i) const;

    /* set the value of a cell */
    bool Set(DTYPE value, int index[], int size = -1);

    /* set the value of a cell with its offset in the array */
    bool Set(DTYPE value, int offset);

    /* set the value of a cell in a 1d tensor */
    bool Set1D(DTYPE value, int i);

    /* set the value of a cell in a 2d tensor */
    bool Set2D(DTYPE value, int ni, int mi);

    /* set the value of a cell in a 3d tensor */
    bool Set3D(DTYPE value, int d0, int d1, int d2);
    
    /* set the integer value of a cell by its offset */
    bool SetInt(int value, int offset);
    
    /* set the integer value of a cell */
    bool SetInt(int value, int index[], int size = -1);

    /* set the integer value of a cell in a 1d tensor */
    bool Set1DInt(int value, int i);

    /* set the integer value of a cell in a 2d tensor */
    bool Set2DInt(int value, int ni, int mi);

    /* set the integer value of a cell in a 3d tensor */
    bool Set3DInt(int value, int d0, int d1, int d2);

    /* increase the value of a cell in a 2d */
    bool Add2D(DTYPE value, int ni, int mi);

    /* get the number of non-zero elements (in a sparse tensor) */
    int GetNonzeroSize() const;

    /* set the tensor as "temporary" */
    void SetTMPFlag(bool myIsTmp = true);

    /* set the tensor as "keep-gradient" */
    void SetGradFlag(bool myIsGrad = true);

    /* set the tensor as "variable" */
    void SetVarFlag(bool myIsVar = true);

    /* resize a tensor with a specified tensor size */
    bool Resize(const int myOrder, const int * myDimSize,
                const TENSOR_DATA_TYPE myDataType = DEFAULT_DTYPE,
                const float myDenseRatio = 1.0F);

    /* resize a tensor by another one */
    bool Resize(const XTensor * myTensor);

    /* binary search to find an element in a sparse matrix*/
    bool BinarySearch(int key, DTYPE &value, void * &position) const;

    /* dump data to a file */
    void Dump(FILE * file, const char * label = NULL, const int n = -1, const int beg = 0, const int verbose = 0);

    /* dump data to a file */
    static
    void Dump(const XTensor * tensor, FILE * file, const char * label = NULL, const int n = -1, const int beg = 0, const int verbose = 0);

    /* dump data to a binary file */
    void BinaryDump(FILE * file);

    /* read data from a file */
    void Read(FILE * file, const char * label = NULL);

    /* read data from a binary file */
    void BinaryRead(FILE * file, size_t offset);

    /* flush the data to the target device */
    void FlushToMem(XMem * targetMem);

    /* allocate the memory space of the tensor (in the global memory) */
    static
    void AllocateData(XTensor * tensor, XMem * myMem = NULL, bool useBuf = false);

    /* free the memory space of the tensor (in the global memory) */
    static
    void FreeData(XTensor * tensor, XMem * myMem = NULL, bool useBuf = false);
};

/* we make a unique id for every tensor */
extern int tensorIDGlobal;
extern MUTEX_HANDLE tensorMutex;
extern XTensor NULLTensor;
extern int MakeTensorID();


/* overloading of the plus-sign */
XTensor  operator+ (const DTYPE shift, const XTensor &tensor);

/* overloading of the minus-sign */
XTensor  operator- (const DTYPE shift, const XTensor &tensor);

/* overloading of the multiply-sign */
XTensor  operator* (const DTYPE scale, const XTensor &tensor);

/* overloading of the division-sign */
XTensor  operator/ (const DTYPE scale, const XTensor &tensor);

} /* end of the nts (NiuTrans.Tensor) namespace */

#endif
