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
* $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-04-24
*/

#ifndef __XTENSORBLAS_H__
#define __XTENSORBLAS_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* matrix multiplication (BLAS) */
void _MatrixMULCPU(const XTensor * a, MATRIX_TRANS_TYPE transposedA, const XTensor * b, MATRIX_TRANS_TYPE transposedB, 
                   XTensor * c, DTYPE alpha = (DTYPE)1.0, DTYPE beta = 0);

#ifdef USE_CUDA

/* matrix multiplication via cuda version BLAS */
void _CudaBLASMatrixMUL(cublasHandle_t * handle,
                        const void * a, MATRIX_TRANS_TYPE transposedA, TENSOR_DATA_TYPE dataTypeA,
                        const void * b, MATRIX_TRANS_TYPE transposedB, TENSOR_DATA_TYPE dataTypeB,
                        void * c, TENSOR_DATA_TYPE dataTypeC,
                        int na, int ma, int nb, int mb, int nc, int mc, DTYPE alpha = (DTYPE)1.0, DTYPE beta = 1.0);

/* matrix multiplication in batch mode via cuda version BLAS */
void _CudaBLASMatrixMULBatched(cublasHandle_t * handle,
                               const void ** a, MATRIX_TRANS_TYPE transposedA, TENSOR_DATA_TYPE dataTypeA,
                               const void ** b, MATRIX_TRANS_TYPE transposedB, TENSOR_DATA_TYPE dataTypeB,
                               void ** c, TENSOR_DATA_TYPE dataTypeC,
                               int count, int na, int ma, int nb, int mb, int nc, int mc, 
                               DTYPE alpha = (DTYPE)1.0, DTYPE beta = 1.0);

/* matrix multiplication in batch and strided mode via cuda version BLAS */
void _CudaBLASMatrixMULBatchedStrided(cublasHandle_t * handle,
                                      const void * a, MATRIX_TRANS_TYPE transposedA, TENSOR_DATA_TYPE dataTypeA, long long int strideA,
                                      const void * b, MATRIX_TRANS_TYPE transposedB, TENSOR_DATA_TYPE dataTypeB, long long int strideB,
                                      void * c, TENSOR_DATA_TYPE dataTypeC, long long int strideC,
                                      int count, int na, int ma, int nb, int mb, int nc, int mc, 
                                      DTYPE alpha = (DTYPE)1.0, DTYPE beta = 1.0);

/* matrix multiplication in batch mode via cuda version BLAS */
void _CudaBLASMatrixMULList(cublasHandle_t * handle, const TensorList * a, MATRIX_TRANS_TYPE transposedA, 
                            const TensorList * b, MATRIX_TRANS_TYPE transposedB, TensorList * c,
                            int count, DTYPE alpha = (DTYPE)1.0, DTYPE beta = 1.0);

#endif
} // namespace nts(NiuTrans.Tensor)

#endif // __XTENSORBLAS_H__
