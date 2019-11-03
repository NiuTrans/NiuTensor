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
* $Created by: LI Yinqiao (email: li.yin.qiao.2012@hotmail.com) 2019-10-21
*/

#include "XTensor.h"
#include "XCall.h"
#include "XDevice.h"
#include "XUtility.h"


namespace nts { // namespace nts(NiuTrans.Tensor)

/*************************************************
* we define the "new and delete" functions below
*/

/* 
initialize a tensor V2
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

void InitTensorV2(XTensor * tensor,
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
>> myOrder - order of the tensor
>> myDimSize - the size of each dimension
>> myDataType - unit size (e.g., int, float, and double)
>> myDenseRatio - how often an element has non-zero value
>> myDevID - when myMem is NULL, myDevID specifies the device 
             on which we allocate the data on site
*/

void InitTensor(XTensor * tensor,
                const int myOrder, const int * myDimSize, const TENSOR_DATA_TYPE myDataType,
                const int myDevID, const bool isEnableGrad)
{
    if (tensor->mem == NULL) {
        XMem * myMem = GMems.GetMem(myDevID);
        tensor->mem = myMem;
        tensor->devID = myMem->devID;
    }
    if(tensor->mem != NULL){
        tensor->Resize(myOrder, myDimSize, myDataType, 1.0F);
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

        tensor->Resize(myOrder, dims, myDataType, 1.0F);

        if(allocated)
            XTensor::AllocateData(tensor);
    }
    tensor->enableGrad = isEnableGrad;
}

/* 
initialize a dense tensor V2 
>> tensor - the tensor we intend to initialize
>> num - number of elements
>> myDataType - unit size (e.g., int, float, and double) 
>> myDevID - when myMem is NULL, myDevID specifies the device 
             on which we allocate the data on site
>> myMem - memory pool used to allocating the data array
           myMem = NULL means that the tensor is allocated on
           the device dynamically, rather than on the memory pool
*/

void InitTensor1DV2(XTensor * tensor, const int num,
                  const TENSOR_DATA_TYPE myDataType, const int myDevID, XMem * myMem)
{
    int dims[1];
    dims[0] = num;

    InitTensorV2(tensor, 1, dims, myDataType, 1.0F, myDevID, myMem);
}

/* 
initialize a dense tensor 
>> tensor - the tensor we intend to initialize
>> num - number of elements
>> myDataType - unit size (e.g., int, float, and double) 
>> myDevID - when myMem is NULL, myDevID specifies the device 
             on which we allocate the data on site
*/

void InitTensor1D(XTensor * tensor, const int num,
                  const TENSOR_DATA_TYPE myDataType, const int myDevID, const bool isEnableGrad)
{
    int dims[1];
    dims[0] = num;

    InitTensor(tensor, 1, dims, myDataType, myDevID, isEnableGrad);
}

/* 
initialize a dense matrix V2 
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

void InitTensor2DV2(XTensor * tensor, const int rowNum, const int colNum,
                  const TENSOR_DATA_TYPE myDataType, const int myDevID, XMem * myMem)
{
    int dims[2];
    dims[0] = rowNum;
    dims[1] = colNum;

    InitTensorV2(tensor, 2, dims, myDataType, 1.0F, myDevID, myMem);
}

/* 
initialize a dense matrix 
>> tensor - the tensor we intend to initialize
>> rowNum - number of rows
>> colNum - number of columns
>> myDataType - unit size (e.g., int, float, and double) 
>> myDevID - when myMem is NULL, myDevID specifies the device 
             on which we allocate the data on site
*/

void InitTensor2D(XTensor * tensor, const int rowNum, const int colNum,
                  const TENSOR_DATA_TYPE myDataType, const int myDevID, const bool isEnableGrad)
{
    int dims[2];
    dims[0] = rowNum;
    dims[1] = colNum;

    InitTensor(tensor, 2, dims, myDataType, myDevID, isEnableGrad);
}

/* 
initialize a dense 3d tensor V2 
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

void InitTensor3DV2(XTensor * tensor, const int d0, const int d1, const int d2, 
                  const TENSOR_DATA_TYPE myDataType, const int myDevID, XMem * myMem)
{
    int dims[3];
    dims[0] = d0;
    dims[1] = d1;
    dims[2] = d2;

    InitTensorV2(tensor, 3, dims, myDataType, 1.0F, myDevID, myMem);
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
*/

void InitTensor3D(XTensor * tensor, const int d0, const int d1, const int d2, 
                  const TENSOR_DATA_TYPE myDataType, const int myDevID, const bool isEnableGrad)
{
    int dims[3];
    dims[0] = d0;
    dims[1] = d1;
    dims[2] = d2;

    InitTensor(tensor, 3, dims, myDataType, myDevID, isEnableGrad);
}
    
/*
initialize a dense 4d tensor V2 
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

void InitTensor4DV2(XTensor * tensor, const int d0, const int d1, const int d2, const int d3,
                  const TENSOR_DATA_TYPE myDataType, const int myDevID, XMem * myMem)
{
    int dims[4];
    dims[0] = d0;
    dims[1] = d1;
    dims[2] = d2;
    dims[3] = d3;
    
    InitTensorV2(tensor, 4, dims, myDataType, 1.0F, myDevID, myMem);
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
*/

void InitTensor4D(XTensor * tensor, const int d0, const int d1, const int d2, const int d3,
                  const TENSOR_DATA_TYPE myDataType, const int myDevID, const bool isEnableGrad)
{
    int dims[4];
    dims[0] = d0;
    dims[1] = d1;
    dims[2] = d2;
    dims[3] = d3;
    
    InitTensor(tensor, 4, dims, myDataType, myDevID, isEnableGrad);
}

/*
initialize a dense 5d tensor V2
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

void InitTensor5DV2(XTensor * tensor, const int d0, const int d1, const int d2, const int d3, const int d4,
                  const TENSOR_DATA_TYPE myDataType, const int myDevID, XMem * myMem)
{
    int dims[5];
    dims[0] = d0;
    dims[1] = d1;
    dims[2] = d2;
    dims[3] = d3;
    dims[4] = d4;
    
    InitTensorV2(tensor, 5, dims, myDataType, 1.0F, myDevID, myMem);
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
*/

void InitTensor5D(XTensor * tensor, const int d0, const int d1, const int d2, const int d3, const int d4,
                    const TENSOR_DATA_TYPE myDataType, const int myDevID, const bool isEnableGrad)
{
    int dims[5];
    dims[0] = d0;
    dims[1] = d1;
    dims[2] = d2;
    dims[3] = d3;
    dims[4] = d4;
    
    InitTensor(tensor, 5, dims, myDataType, myDevID, isEnableGrad);
}

/* 
initialize a tensor with a reference tensor V2 
>> tensor - the tensor we intend to initialize
>> reference - the reference tensor
*/
void InitTensorV2(XTensor * tensor, const XTensor * reference)
{
    if(reference->order < 0)
        return;

    tensor->enableGrad = reference->enableGrad;
    InitTensorV2(tensor, reference->order, reference->dimSize, 
               reference->dataType, reference->denseRatio, 
               reference->devID, reference->mem);
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

    tensor->enableGrad = reference->enableGrad;
    InitTensor(tensor, reference->order, reference->dimSize, 
               reference->dataType, reference->devID);
}
    
/*
initialize a tensor on the CPU with a reference tensor
>> tensor - the tensor we intend to initialize
>> reference - the reference tensor
*/
void InitTensorOnCPU(XTensor * tensor, const XTensor * reference)
{
    if(reference->order < 0)
        return;
    
    tensor->enableGrad = reference->enableGrad;
    InitTensor(tensor, reference->order, reference->dimSize,
               reference->dataType, -1);
}
    
/* generate a XTensor with no initialization */
XTensor * NewTensor()
{
    XTensor * tensor = new XTensor();
    return tensor;
}

/* 
generate a XTensor V2 
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

XTensor * NewTensorV2(const int myOrder, const int * myDimSize, const TENSOR_DATA_TYPE myDataType,
                    const float myDenseRatio, const int myDevID, XMem * myMem)
{
    if(myMem != NULL)
        return new XTensor(myOrder, myDimSize, myDataType, myDenseRatio, myDevID, myMem);
    else{
        XTensor * tensor = new XTensor();
        InitTensorV2(tensor, myOrder, myDimSize, myDataType, myDenseRatio, myDevID, myMem);
        return tensor;
    }
}

/* 
generate a dense XTensor 
>> myOrder - order of the tensor
>> myDimSize - the size of each dimension
>> myDataType - unit size (e.g., int, float, and double)
>> myDenseRatio - how often an element has non-zero value
>> myDevID - when myMem is NULL, myDevID specifies the device 
             on which we allocate the data on site.
*/

XTensor * NewTensor(const int myOrder, const int * myDimSize, const TENSOR_DATA_TYPE myDataType,
                      const int myDevID, const bool isEnableGrad)
{
    XMem * myMem = GMems.GetMem(myDevID);
    XTensor * tensor = new XTensor(myOrder, myDimSize, myDataType, 1.0F, myDevID, myMem);
    tensor->enableGrad = isEnableGrad;
    return tensor;
}

/*
generate a XTensor which allocates data on the buffer V2 
>> myOrder - order of the tensor
>> myDimSize - the size of each dimension
>> myMem - memory pool used to allocating the data array.
           we actually allocate the data on the buffer associated with
           the memory pool
>> devID - device id
>> myDataType - unit size (e.g., int, float, and double)
>> myDenseRatio - how often an element has non-zero value

*/
XTensor * NewTensorBufV2(const int myOrder, const int * myDimSize,
                       const TENSOR_DATA_TYPE myDataType, const float myDenseRatio,
                       const int devID, XMem * myMem)
{
    int dims[MAX_TENSOR_DIM_NUM];
    memcpy(dims, myDimSize, sizeof(int) * myOrder);

    dims[0] = -abs(dims[0]);

    XTensor * tensor = NewTensorV2(myOrder, dims, myDataType, myDenseRatio, devID, myMem);

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
generate a dense XTensor which allocates data on the buffer 
>> myOrder - order of the tensor
>> myDimSize - the size of each dimension
>> devID - device id
>> myDataType - unit size (e.g., int, float, and double)
>> myDenseRatio - how often an element has non-zero value

*/
XTensor * NewTensorBuf(const int myOrder, const int * myDimSize,
                       const TENSOR_DATA_TYPE myDataType, const int devID, const bool isEnableGrad)
{
    int dims[MAX_TENSOR_DIM_NUM];
    memcpy(dims, myDimSize, sizeof(int) * myOrder);

    dims[0] = -abs(dims[0]);

    XTensor * tensor = NewTensor(myOrder, dims, myDataType, devID, isEnableGrad);

    if (tensor->unitNum * tensor->unitSize == 176657664) {
        tensor->Dump(stderr, "", 200);
    }

    XMem * myMem = GMems.GetMem(devID);
    tensor->data = myMem->AllocBuf(myMem->devID, tensor->unitNum * tensor->unitSize);

    return tensor;
}

/* 
generate a XTensor which allocates data on the buffer V2 
>> reference - reference tensor
>> devID - device id
>> myMem - memory pool used to allocating the data array.
           we actually allocate the data on the buffer associated with
           the memory pool
*/
XTensor * NewTensorBufV2(const XTensor * reference, int devID, XMem * myMem)
{
    return NewTensorBufV2(reference->order, reference->dimSize, 
                        reference->dataType, reference->denseRatio,
                        devID, myMem);
}

/* 
generate a XTensor which allocates data on the buffer 
>> reference - reference tensor
>> devID - device id
*/
XTensor * NewTensorBuf(const XTensor * reference, int devID, const bool isEnableGrad)
{
    return NewTensorBuf(reference->order, reference->dimSize, 
                        reference->dataType, devID, isEnableGrad);
}

/* 
generate a dense vector V2 
>> num - number of entries
>> myDataType - unit size (e.g., int, float, and double) 
>> myDevID - when myMem is NULL, myDevID specifies the device 
             on which we allocate the data on site
>> myMem - memory pool used to allocating the data array
           myMem = NULL means that the tensor is allocated on
           the device dynamically, rather than on the memory pool.
*/

XTensor * NewTensor1DV2(const int num, 
                      const TENSOR_DATA_TYPE myDataType, const int myDevID, XMem * myMem)
{
    int dims[1];
    dims[0] = num;

    return NewTensorV2(1, dims, myDataType, 1.0F, myDevID, myMem);
}

/* 
generate a dense vector 
>> num - number of entries
>> myDataType - unit size (e.g., int, float, and double) 
>> myDevID - when myMem is NULL, myDevID specifies the device 
             on which we allocate the data on site.
*/

XTensor * NewTensor1D(const int num, 
                      const TENSOR_DATA_TYPE myDataType, const int myDevID, const bool isEnableGrad)
{
    int dims[1];
    dims[0] = num;

    return NewTensor(1, dims, myDataType, myDevID, isEnableGrad);
}

/* 
generate a dense matrix V2 
>> rowNum - number of rows
>> colNum - number of colums
>> myDataType - unit size (e.g., int, float, and double) 
>> myDevID - when myMem is NULL, myDevID specifies the device 
             on which we allocate the data on site
>> myMem - memory pool used to allocating the data array
           myMem = NULL means that the tensor is allocated on
           the device dynamically, rather than on the memory pool.
*/

XTensor * NewTensor2DV2(const int rowNum, const int colNum,
                      const TENSOR_DATA_TYPE myDataType, const int myDevID, XMem * myMem)
{
    int dims[2];
    dims[0] = rowNum;
    dims[1] = colNum;

    return NewTensorV2(2, dims, myDataType, 1.0F, myDevID, myMem);
}

/* 
generate a dense matrix 
>> rowNum - number of rows
>> colNum - number of colums
>> myDataType - unit size (e.g., int, float, and double) 
>> myDevID - when myMem is NULL, myDevID specifies the device 
             on which we allocate the data on site.
*/

XTensor * NewTensor2D(const int rowNum, const int colNum,
                      const TENSOR_DATA_TYPE myDataType, const int myDevID, const bool isEnableGrad)
{
    int dims[2];
    dims[0] = rowNum;
    dims[1] = colNum;

    return NewTensor(2, dims, myDataType, myDevID, isEnableGrad);
}

/* 
generate a dense 3d tensor V2 
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

XTensor * NewTensor3DV2(const int d0, const int d1, const int d2,
                      const TENSOR_DATA_TYPE myDataType, const int myDevID, XMem * myMem)
{
    int dims[3];
    dims[0] = d0;
    dims[1] = d1;
    dims[2] = d2;

    return NewTensorV2(3, dims, myDataType, 1.0F, myDevID, myMem);
}

/* 
generate a dense 3d tensor 
>> d0 - size of dimension 0
>> d1 - size of dimension 1
>> d2 - size of dimension 2
>> myDataType - unit size (e.g., int, float, and double) 
>> myDevID - when myMem is NULL, myDevID specifies the device 
             on which we allocate the data on site.
*/

XTensor * NewTensor3D(const int d0, const int d1, const int d2,
                      const TENSOR_DATA_TYPE myDataType, const int myDevID, const bool isEnableGrad)
{
    int dims[3];
    dims[0] = d0;
    dims[1] = d1;
    dims[2] = d2;

    return NewTensor(3, dims, myDataType, myDevID, isEnableGrad);
}

/* 
generate a dense 4d tensor V2 
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

XTensor * NewTensor4DV2(const int d0, const int d1, const int d2, const int d3,
                      const TENSOR_DATA_TYPE myDataType, const int myDevID, XMem * myMem)
{
    int dims[4];
    dims[0] = d0;
    dims[1] = d1;
    dims[2] = d2;
    dims[3] = d3;

    return NewTensorV2(4, dims, myDataType, 1.0F, myDevID, myMem);
}

/* 
generate a dense 4d tensor 
>> d0 - size of dimension 0
>> d1 - size of dimension 1
>> d2 - size of dimension 2
>> d3 - size of dimension 3
>> myDataType - unit size (e.g., int, float, and double) 
>> myDevID - when myMem is NULL, myDevID specifies the device 
             on which we allocate the data on site.
*/

XTensor * NewTensor4D(const int d0, const int d1, const int d2, const int d3,
                      const TENSOR_DATA_TYPE myDataType, const int myDevID, const bool isEnableGrad)
{
    int dims[4];
    dims[0] = d0;
    dims[1] = d1;
    dims[2] = d2;
    dims[3] = d3;

    return NewTensor(4, dims, myDataType, myDevID, isEnableGrad);
}

/* 
generate a dense 5d tensor V2
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

XTensor * NewTensor5DV2(const int d0, const int d1, const int d2, const int d3, const int d4,
                      const TENSOR_DATA_TYPE myDataType, const int myDevID, XMem * myMem)
{
    int dims[5];
    dims[0] = d0;
    dims[1] = d1;
    dims[2] = d2;
    dims[3] = d3;
    dims[4] = d4;

    return NewTensorV2(5, dims, myDataType, 1.0F, myDevID, myMem);
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
             on which we allocate the data on site.
*/

XTensor * NewTensor5D(const int d0, const int d1, const int d2, const int d3, const int d4,
                      const TENSOR_DATA_TYPE myDataType, const int myDevID, const bool isEnableGrad)
{
    int dims[5];
    dims[0] = d0;
    dims[1] = d1;
    dims[2] = d2;
    dims[3] = d3;
    dims[4] = d4;

    return NewTensor(5, dims, myDataType, myDevID, isEnableGrad);
}

XTensor * NewTensorRange(int lower, int upper, int step, const TENSOR_DATA_TYPE myDataType, const int myDevID, const bool isEnableGrad)
{
    int size = abs(upper - lower);
    int unitNum = ceil(1.0 * size / abs(step));

    XTensor * tensor = NewTensor1D(unitNum, myDataType, myDevID, isEnableGrad);
    tensor->Range(lower, upper, step);
    return tensor;
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

} // namespace nts(NiuTrans.Tensor)



