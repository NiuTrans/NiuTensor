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

#ifndef __XCALL_H__
#define __XCALL_H__

#include "XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
* we define the "new and delete" functions below
*/

/* initialize a XTensor V2 */
void InitTensorV2(XTensor * tensor,
                const int myOrder, const int * myDimSize, const TENSOR_DATA_TYPE myDataType = X_FLOAT,
                const float myDenseRatio = 1.0F, const int myDevID = -1, XMem * myMem = NULL);
                
/* initialize a dense XTensor */
void InitTensor(XTensor * tensor,
                const int myOrder, const int * myDimSize, const TENSOR_DATA_TYPE myDataType = X_FLOAT,
                const int myDevID = -1, const bool isEnableGrad = true);

/* initialize a dense vector V2 */
void InitTensor1DV2(XTensor * tensor, const int num, 
                  const TENSOR_DATA_TYPE myDataType = X_FLOAT, const int myDevID = -1, XMem * myMem = NULL);

/* initialize a dense vector */
void InitTensor1D(XTensor * tensor, const int num, 
                  const TENSOR_DATA_TYPE myDataType = X_FLOAT, const int myDevID = -1, const bool isEnableGrad = true);

/* initialize a dense matrix V2 */
void InitTensor2DV2(XTensor * tensor, const int rowNum, const int colNum,
                  const TENSOR_DATA_TYPE myDataType = X_FLOAT, const int myDevID = -1, XMem * myMem = NULL);

/* initialize a dense matrix */
void InitTensor2D(XTensor * tensor, const int rowNum, const int colNum,
                  const TENSOR_DATA_TYPE myDataType = X_FLOAT, const int myDevID = -1, const bool isEnableGrad = true);

/* initialize a dense 3d tensor V2 */
void InitTensor3DV2(XTensor * tensor, const int d0, const int d1, const int d2,
                  const TENSOR_DATA_TYPE myDataType = X_FLOAT, const int myDevID = -1, XMem * myMem = NULL);

/* initialize a dense 3d tensor */
void InitTensor3D(XTensor * tensor, const int d0, const int d1, const int d2,
                  const TENSOR_DATA_TYPE myDataType = X_FLOAT, const int myDevID = -1, const bool isEnableGrad = true);
    
/* initialize a dense 4d tensor V2 */
void InitTensor4DV2(XTensor * tensor, const int d0, const int d1, const int d2, const int d3,
                  const TENSOR_DATA_TYPE myDataType = X_FLOAT, const int myDevID = -1, XMem * myMem = NULL);

/* initialize a dense 4d tensor */
void InitTensor4D(XTensor * tensor, const int d0, const int d1, const int d2, const int d3,
                  const TENSOR_DATA_TYPE myDataType = X_FLOAT, const int myDevID = -1, const bool isEnableGrad = true);

/* initialize a dense 5d tensor V2 */
void InitTensor5DV2(XTensor * tensor, const int d0, const int d1, const int d2, const int d3, const int d4,
                  const TENSOR_DATA_TYPE myDataType = X_FLOAT, const int myDevID = -1, XMem * myMem = NULL);

/* initialize a dense 5d tensor */
void InitTensor5D(XTensor * tensor, const int d0, const int d1, const int d2, const int d3, const int d4,
                    const TENSOR_DATA_TYPE myDataType = X_FLOAT, const int myDevID = -1, const bool isEnableGrad = true);

/* initialize a tensor with a reference tensor V2 */
void InitTensorV2(XTensor * tensor, const XTensor * reference);

/* initialize a tensor with a reference tensor */
void InitTensor(XTensor * tensor, const XTensor * reference);
    
/* initialize a tensor on the CPU with a reference tensor */
void InitTensorOnCPU(XTensor * tensor, const XTensor * reference);
    
/* generate a XTensor with no initialization */
XTensor * NewTensor();

/* generate a XTensor V2 */
XTensor * NewTensorV2(const int myOrder, const int * myDimSize, const TENSOR_DATA_TYPE myDataType = X_FLOAT,
                    const float myDenseRatio = 1.0F, const int myDevID = -1, XMem * myMem = NULL);

/* generate a dense XTensor */
XTensor * NewTensor(const int myOrder, const int * myDimSize, const TENSOR_DATA_TYPE myDataType = X_FLOAT,
                      const int myDevID = -1, const bool isEnableGrad = true);

/* generate a XTensor which allocates data on the buffer V2 */
XTensor * NewTensorBufV2(const int myOrder, const int * myDimSize,
                       const TENSOR_DATA_TYPE myDataType = X_FLOAT, const float myDenseRatio = 1.0F,
                       const int myDevID = -1, XMem * myMem = NULL);

/* generate a dense XTensor which allocates data on the buffer */
XTensor * NewTensorBuf(const int myOrder, const int * myDimSize,
                       const TENSOR_DATA_TYPE myDataType = X_FLOAT, const int myDevID = -1, const bool isEnableGrad = true);

/* generate a XTensor which allocates data on the buffer V2 */
XTensor * NewTensorBufV2(const XTensor * reference, int devID, XMem * myMem);

/* generate a XTensor which allocates data on the buffer */
XTensor * NewTensorBuf(const XTensor * reference, int devID, const bool isEnableGrad = true);

/* generate a dense vector V2 */
XTensor * NewTensor1DV2(const int num, const TENSOR_DATA_TYPE myDataType = X_FLOAT, const int myDevID = -1, 
                      XMem * myMem = NULL);

/* generate a dense vector */
XTensor * NewTensor1D(const int num, const TENSOR_DATA_TYPE myDataType = X_FLOAT, const int myDevID = -1, const bool isEnableGrad = true);

/* generate a dense matrix V2 */
XTensor * NewTensor2DV2(const int rowNum, const int colNum, 
                      const TENSOR_DATA_TYPE myDataType = X_FLOAT, 
                      const int myDevID = -1, XMem * myMem = NULL);

/* generate a dense matrix */
XTensor * NewTensor2D(const int rowNum, const int colNum, 
                      const TENSOR_DATA_TYPE myDataType = X_FLOAT, 
                      const int myDevID = -1, const bool isEnableGrad = true);

/* generate a dense 3d tensor V2 */
XTensor * NewTensor3DV2(const int d0, const int d1, const int d2, 
                      const TENSOR_DATA_TYPE myDataType = X_FLOAT, 
                      const int myDevID = -1, XMem * myMem = NULL);

/* generate a dense 3d tensor */
XTensor * NewTensor3D(const int d0, const int d1, const int d2, 
                      const TENSOR_DATA_TYPE myDataType = X_FLOAT, 
                      const int myDevID = -1, const bool isEnableGrad = true);

/* generate a dense 4d tensor V2 */
XTensor * NewTensor4DV2(const int d0, const int d1, const int d2, const int d3,
                      const TENSOR_DATA_TYPE myDataType = X_FLOAT, 
                      const int myDevID = -1, XMem * myMem = NULL);

/* generate a dense 4d tensor */
XTensor * NewTensor4D(const int d0, const int d1, const int d2, const int d3,
                      const TENSOR_DATA_TYPE myDataType = X_FLOAT, 
                      const int myDevID = -1, const bool isEnableGrad = true);

/* generate a dense 5d tensor V2 */
XTensor * NewTensor5DV2(const int d0, const int d1, const int d2, const int d3, const int d4,
                      const TENSOR_DATA_TYPE myDataType = X_FLOAT, 
                      const int myDevID = -1, XMem * myMem = NULL);

/* generate a dense 5d tensor */
XTensor * NewTensor5D(const int d0, const int d1, const int d2, const int d3, const int d4,
                      const TENSOR_DATA_TYPE myDataType = X_FLOAT, 
                      const int myDevID = -1, const bool isEnableGrad = true);

/* generate a dense vector by range */
XTensor * NewTensorRange(int lower, int upper, int step, const TENSOR_DATA_TYPE myDataType = X_INT, const int myDevID = -1, const bool isEnableGrad = true);

/* generate a copy of XTensor (with a reference to a given tensor) */
XTensor * NewTensor(const XTensor * a, bool isFilledData = true);

/* free the data space of a given tensor */
void DelTensor(XTensor * tensor);

/* free the data space of a given tensor (on the buffer) */
void DelTensorBuf(XTensor * tensor);

} // namespace nts(NiuTrans.Tensor)

#endif // __XCALL_H__