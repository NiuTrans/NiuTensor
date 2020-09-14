/* NiuTrans.Tensor - an open-source tensor library
* Copyright (C) 2017, Natural Language Processing Lab, Northeastern University.
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
* $Update by: Lin Ye (email: linye2015@outlook.com) 2019-07-16 float16/int8 added
*/

#include "../../XUtility.h"
#include "../../XDevice.h"
#include "../../XTensor.h"
#include "../shape/IsSameShaped.h"
#include "XTensorBLAS.h"
#include "cuda_fp16.h"
#include "cublas_api.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/*
matrix multiplication via cuda version BLAS
*/
void _CudaBLASMatrixMUL(cublasHandle_t * handle,
                        const void * a, MATRIX_TRANS_TYPE transposedA, TENSOR_DATA_TYPE dataTypeA,
                        const void * b, MATRIX_TRANS_TYPE transposedB, TENSOR_DATA_TYPE dataTypeB,
                        void * c, TENSOR_DATA_TYPE dataTypeC,
                        int na, int ma, int nb, int mb, int nc, int mc,
                        DTYPE alpha, DTYPE beta)
{
    /*
    matrxi-matrix multiplication
    For row-major matrices (as in c/c++), the trick used here is (AB)^T = B^T * A^T
    */
    if (dataTypeA == X_DOUBLE && dataTypeB == X_DOUBLE && dataTypeC == X_DOUBLE) {
        double alpha2 = (double)alpha;
        double beta2 = (double)beta;
        if (transposedA == X_NOTRANS && transposedB == X_NOTRANS)
            cublasDgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, mc, nc, ma, &alpha2, (const double*)b, mb, (const double*)a, ma, &beta2, (double*)c, mc);
        else if (transposedA == X_TRANS && transposedB == X_NOTRANS)
            cublasDgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_T, mc, nc, na, &alpha2, (const double*)b, mb, (const double*)a, ma, &beta2, (double*)c, mc);
        else if (transposedA == X_NOTRANS && transposedB == X_TRANS)
            cublasDgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_N, mc, nc, ma, &alpha2, (const double*)b, mb, (const double*)a, ma, &beta2, (double*)c, mc);
        else if (transposedA == X_TRANS && transposedB == X_TRANS)
            cublasDgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_T, mc, nc, na, &alpha2, (const double*)b, nb, (const double*)a, ma, &beta2, (double*)c, mc);
    }
    else if (dataTypeA == X_FLOAT && dataTypeB == X_FLOAT && dataTypeC == X_FLOAT) {
        float alpha2 = (float)alpha;
        float beta2 = (float)beta;
        if (transposedA == X_NOTRANS && transposedB == X_NOTRANS)
            cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, mc, nc, ma, &alpha2, (const float*)b, mb, (const float*)a, ma, &beta2, (float*)c, mc);
        else if (transposedA == X_TRANS && transposedB == X_NOTRANS)
            cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_T, mc, nc, na, &alpha2, (const float*)b, mb, (const float*)a, ma, &beta2, (float*)c, mc);
        else if (transposedA == X_NOTRANS && transposedB == X_TRANS)
            cublasSgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_N, mc, nc, ma, &alpha2, (const float*)b, mb, (const float*)a, ma, &beta2, (float*)c, mc);
        else if (transposedA == X_TRANS && transposedB == X_TRANS)
            cublasSgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_T, mc, nc, na, &alpha2, (const float*)b, mb, (const float*)a, ma, &beta2, (float*)c, mc);
    }
    else if (dataTypeA == X_FLOAT16 && dataTypeB == X_FLOAT16 && dataTypeC == X_FLOAT16) {
        __half alpha2 = __float2half(alpha);
        __half beta2 = __float2half(beta);
        cublasSetMathMode(*handle, CUBLAS_TENSOR_OP_MATH);
        if (transposedA == X_NOTRANS && transposedB == X_NOTRANS)
            cublasGemmEx(*handle, CUBLAS_OP_N, CUBLAS_OP_N, mc, nc, ma, (void*)&alpha2, b, CUDA_R_16F, mb, a, CUDA_R_16F, ma, (void*)&beta2, c, CUDA_R_16F, mc, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        else if (transposedA == X_TRANS && transposedB == X_NOTRANS)
            cublasGemmEx(*handle, CUBLAS_OP_N, CUBLAS_OP_T, mc, nc, na, (void*)&alpha2, b, CUDA_R_16F, mb, a, CUDA_R_16F, ma, (void*)&beta2, c, CUDA_R_16F, mc, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        else if (transposedA == X_NOTRANS && transposedB == X_TRANS)
            cublasGemmEx(*handle, CUBLAS_OP_T, CUBLAS_OP_N, mc, nc, ma, (void*)&alpha2, b, CUDA_R_16F, mb, a, CUDA_R_16F, ma, (void*)&beta2, c, CUDA_R_16F, mc, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        else if (transposedA == X_TRANS && transposedB == X_TRANS)
            cublasGemmEx(*handle, CUBLAS_OP_T, CUBLAS_OP_T, mc, nc, na, (void*)&alpha2, b, CUDA_R_16F, mb, a, CUDA_R_16F, ma, (void*)&beta2, c, CUDA_R_16F, mc, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cublasSetMathMode(*handle, CUBLAS_DEFAULT_MATH);
    }
    else if (dataTypeA == X_FLOAT16 && dataTypeB == X_FLOAT16 && dataTypeC == X_FLOAT) {
        float alpha2 = (float)alpha;
        float beta2 = (float)beta;
        cublasSetMathMode(*handle, CUBLAS_TENSOR_OP_MATH);
        if (transposedA == X_NOTRANS && transposedB == X_NOTRANS)
            cublasGemmEx(*handle, CUBLAS_OP_N, CUBLAS_OP_N, mc, nc, ma, (void*)&alpha2, b, CUDA_R_16F, mb, a, CUDA_R_16F, ma, (void*)&beta2, c, CUDA_R_32F, mc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        else if (transposedA == X_TRANS && transposedB == X_NOTRANS)
            cublasGemmEx(*handle, CUBLAS_OP_N, CUBLAS_OP_T, mc, nc, na, (void*)&alpha2, b, CUDA_R_16F, mb, a, CUDA_R_16F, ma, (void*)&beta2, c, CUDA_R_32F, mc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        else if (transposedA == X_NOTRANS && transposedB == X_TRANS)
            cublasGemmEx(*handle, CUBLAS_OP_T, CUBLAS_OP_N, mc, nc, ma, (void*)&alpha2, b, CUDA_R_16F, mb, a, CUDA_R_16F, ma, (void*)&beta2, c, CUDA_R_32F, mc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        else if (transposedA == X_TRANS && transposedB == X_TRANS)
            cublasGemmEx(*handle, CUBLAS_OP_T, CUBLAS_OP_T, mc, nc, na, (void*)&alpha2, b, CUDA_R_16F, mb, a, CUDA_R_16F, ma, (void*)&beta2, c, CUDA_R_32F, mc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cublasSetMathMode(*handle, CUBLAS_DEFAULT_MATH);
    }
    else {
        ShowNTErrors("Unsupported data type!");
    }
}

/*
matrix multiplication via cuda version BLAS
*/
void _CudaBLASMatrixMULBatched(cublasHandle_t * handle,
                               const void ** a, MATRIX_TRANS_TYPE transposedA, TENSOR_DATA_TYPE dataTypeA,
                               const void ** b, MATRIX_TRANS_TYPE transposedB, TENSOR_DATA_TYPE dataTypeB,
                               void ** c, TENSOR_DATA_TYPE dataTypeC,
                               int count, int na, int ma, int nb, int mb, int nc, int mc,
                               DTYPE alpha, DTYPE beta)
{
    /*
    matrxi-matrix multiplication
    For row-major matrices (as in c/c++), the trick used here is (AB)^T = B^T * A^T
    */
    if (dataTypeA == X_DOUBLE && dataTypeB == X_DOUBLE && dataTypeC == X_DOUBLE) {
        double alpha2 = (double)alpha;
        double beta2 = (double)beta;
        if (transposedA == X_NOTRANS && transposedB == X_NOTRANS)
            cublasDgemmBatched(*handle, CUBLAS_OP_N, CUBLAS_OP_N, mc, nc, ma, &alpha2, (const double**)b, mb, (const double**)a, ma, &beta2, (double**)c, mc, count);
        else if (transposedA == X_TRANS && transposedB == X_NOTRANS)
            cublasDgemmBatched(*handle, CUBLAS_OP_N, CUBLAS_OP_T, mc, nc, na, &alpha2, (const double**)b, mb, (const double**)a, ma, &beta2, (double**)c, mc, count);
        else if (transposedA == X_NOTRANS && transposedB == X_TRANS)
            cublasDgemmBatched(*handle, CUBLAS_OP_T, CUBLAS_OP_N, mc, nc, ma, &alpha2, (const double**)b, mb, (const double**)a, ma, &beta2, (double**)c, mc, count);
        else if (transposedA == X_TRANS && transposedB == X_TRANS)
            cublasDgemmBatched(*handle, CUBLAS_OP_T, CUBLAS_OP_T, mc, nc, na, &alpha2, (const double**)b, nb, (const double**)a, ma, &beta2, (double**)c, mc, count);
    }
    else if (dataTypeA == X_FLOAT && dataTypeB == X_FLOAT && dataTypeC == X_FLOAT) {
        float alpha2 = (float)alpha;
        float beta2 = (float)beta;
        if (transposedA == X_NOTRANS && transposedB == X_NOTRANS)
            cublasSgemmBatched(*handle, CUBLAS_OP_N, CUBLAS_OP_N, mc, nc, ma, &alpha2, (const float**)b, mb, (const float**)a, ma, &beta2, (float**)c, mc, count);
        else if (transposedA == X_TRANS && transposedB == X_NOTRANS)
            cublasSgemmBatched(*handle, CUBLAS_OP_N, CUBLAS_OP_T, mc, nc, na, &alpha2, (const float**)b, mb, (const float**)a, ma, &beta2, (float**)c, mc, count);
        else if (transposedA == X_NOTRANS && transposedB == X_TRANS)
            cublasSgemmBatched(*handle, CUBLAS_OP_T, CUBLAS_OP_N, mc, nc, ma, &alpha2, (const float**)b, mb, (const float**)a, ma, &beta2, (float**)c, mc, count);
        else if (transposedA == X_TRANS && transposedB == X_TRANS)
            cublasSgemmBatched(*handle, CUBLAS_OP_T, CUBLAS_OP_T, mc, nc, na, &alpha2, (const float**)b, mb, (const float**)a, ma, &beta2, (float**)c, mc, count);
    }
    else if (dataTypeA == X_FLOAT16 && dataTypeB == X_FLOAT16 && dataTypeC == X_FLOAT16) {
        __half alpha2 = __float2half(alpha);
        __half beta2 = __float2half(beta);
        cublasSetMathMode(*handle, CUBLAS_TENSOR_OP_MATH);
        if (transposedA == X_NOTRANS && transposedB == X_NOTRANS)
            cublasGemmBatchedEx(*handle, CUBLAS_OP_N, CUBLAS_OP_N, mc, nc, ma, (void*)&alpha2, b, CUDA_R_16F, mb, a, CUDA_R_16F, ma, (void*)&beta2, c, CUDA_R_16F, mc, count, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        else if (transposedA == X_TRANS && transposedB == X_NOTRANS)
            cublasGemmBatchedEx(*handle, CUBLAS_OP_N, CUBLAS_OP_T, mc, nc, na, (void*)&alpha2, b, CUDA_R_16F, mb, a, CUDA_R_16F, ma, (void*)&beta2, c, CUDA_R_16F, mc, count, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        else if (transposedA == X_NOTRANS && transposedB == X_TRANS)
            cublasGemmBatchedEx(*handle, CUBLAS_OP_T, CUBLAS_OP_N, mc, nc, ma, (void*)&alpha2, b, CUDA_R_16F, mb, a, CUDA_R_16F, ma, (void*)&beta2, c, CUDA_R_16F, mc, count, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        else if (transposedA == X_TRANS && transposedB == X_TRANS)
            cublasGemmBatchedEx(*handle, CUBLAS_OP_T, CUBLAS_OP_T, mc, nc, na, (void*)&alpha2, b, CUDA_R_16F, mb, a, CUDA_R_16F, ma, (void*)&beta2, c, CUDA_R_16F, mc, count, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cublasSetMathMode(*handle, CUBLAS_DEFAULT_MATH);
    }
    else if (dataTypeA == X_FLOAT16 && dataTypeB == X_FLOAT16 && dataTypeC == X_FLOAT) {
        float alpha2 = (float)alpha;
        float beta2 = (float)beta;
        cublasSetMathMode(*handle, CUBLAS_TENSOR_OP_MATH);
        if (transposedA == X_NOTRANS && transposedB == X_NOTRANS)
            cublasGemmBatchedEx(*handle, CUBLAS_OP_N, CUBLAS_OP_N, mc, nc, ma, (void*)&alpha2, b, CUDA_R_16F, mb, a, CUDA_R_16F, ma, (void*)&beta2, c, CUDA_R_32F, mc, count, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        else if (transposedA == X_TRANS && transposedB == X_NOTRANS)
            cublasGemmBatchedEx(*handle, CUBLAS_OP_N, CUBLAS_OP_T, mc, nc, na, (void*)&alpha2, b, CUDA_R_16F, mb, a, CUDA_R_16F, ma, (void*)&beta2, c, CUDA_R_32F, mc, count, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        else if (transposedA == X_NOTRANS && transposedB == X_TRANS)
            cublasGemmBatchedEx(*handle, CUBLAS_OP_T, CUBLAS_OP_N, mc, nc, ma, (void*)&alpha2, b, CUDA_R_16F, mb, a, CUDA_R_16F, ma, (void*)&beta2, c, CUDA_R_32F, mc, count, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        else if (transposedA == X_TRANS && transposedB == X_TRANS)
            cublasGemmBatchedEx(*handle, CUBLAS_OP_T, CUBLAS_OP_T, mc, nc, na, (void*)&alpha2, b, CUDA_R_16F, mb, a, CUDA_R_16F, ma, (void*)&beta2, c, CUDA_R_32F, mc, count, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cublasSetMathMode(*handle, CUBLAS_DEFAULT_MATH);
    }
    else {
        ShowNTErrors("Unsupported data type!");
    }
}

/* matrix multiplication in batch and strided mode via cuda version BLAS */
void _CudaBLASMatrixMULBatchedStrided(cublasHandle_t * handle,
                                      const void * a, MATRIX_TRANS_TYPE transposedA, TENSOR_DATA_TYPE dataTypeA, long long int strideA,
                                      const void * b, MATRIX_TRANS_TYPE transposedB, TENSOR_DATA_TYPE dataTypeB, long long int strideB,
                                      void * c, TENSOR_DATA_TYPE dataTypeC, long long int strideC,
                                      int count, int na, int ma, int nb, int mb, int nc, int mc,
                                      DTYPE alpha, DTYPE beta)
{
    /*
    matrxi-matrix multiplication
    For row-major matrices (as in c/c++), the trick used here is (AB)^T = B^T * A^T
    */
    if (dataTypeA == X_DOUBLE && dataTypeB == X_DOUBLE && dataTypeC == X_DOUBLE) {
        double alpha2 = (double)alpha;
        double beta2 = (double)beta;
        if (transposedA == X_NOTRANS && transposedB == X_NOTRANS)
            cublasDgemmStridedBatched(*handle, CUBLAS_OP_N, CUBLAS_OP_N, mc, nc, ma, &alpha2, (const double*)b, mb, strideB, (const double*)a, ma, strideA, &beta2, (double*)c, mc, strideC, count);
        else if (transposedA == X_TRANS && transposedB == X_NOTRANS)
            cublasDgemmStridedBatched(*handle, CUBLAS_OP_N, CUBLAS_OP_T, mc, nc, na, &alpha2, (const double*)b, mb, strideB, (const double*)a, ma, strideA, &beta2, (double*)c, mc, strideC, count);
        else if (transposedA == X_NOTRANS && transposedB == X_TRANS)
            cublasDgemmStridedBatched(*handle, CUBLAS_OP_T, CUBLAS_OP_N, mc, nc, ma, &alpha2, (const double*)b, mb, strideB, (const double*)a, ma, strideA, &beta2, (double*)c, mc, strideC, count);
        else if (transposedA == X_TRANS && transposedB == X_TRANS)
            cublasDgemmStridedBatched(*handle, CUBLAS_OP_T, CUBLAS_OP_T, mc, nc, na, &alpha2, (const double*)b, nb, strideB, (const double*)a, ma, strideA, &beta2, (double*)c, mc, strideC, count);
    }
    else if (dataTypeA == X_FLOAT && dataTypeB == X_FLOAT && dataTypeC == X_FLOAT) {
        float alpha2 = (float)alpha;
        float beta2 = (float)beta;
        if (transposedA == X_NOTRANS && transposedB == X_NOTRANS)
            cublasSgemmStridedBatched(*handle, CUBLAS_OP_N, CUBLAS_OP_N, mc, nc, ma, &alpha2, (const float*)b, mb, strideB, (const float*)a, ma, strideA, &beta2, (float*)c, mc, strideC, count);
        else if (transposedA == X_TRANS && transposedB == X_NOTRANS)
            cublasSgemmStridedBatched(*handle, CUBLAS_OP_N, CUBLAS_OP_T, mc, nc, na, &alpha2, (const float*)b, mb, strideB, (const float*)a, ma, strideA, &beta2, (float*)c, mc, strideC, count);
        else if (transposedA == X_NOTRANS && transposedB == X_TRANS)
            cublasSgemmStridedBatched(*handle, CUBLAS_OP_T, CUBLAS_OP_N, mc, nc, ma, &alpha2, (const float*)b, mb, strideB, (const float*)a, ma, strideA, &beta2, (float*)c, mc, strideC, count);
        else if (transposedA == X_TRANS && transposedB == X_TRANS)
            cublasSgemmStridedBatched(*handle, CUBLAS_OP_T, CUBLAS_OP_T, mc, nc, na, &alpha2, (const float*)b, mb, strideB, (const float*)a, ma, strideA, &beta2, (float*)c, mc, strideC, count);
    }
    else if (dataTypeA == X_FLOAT16 && dataTypeB == X_FLOAT16 && dataTypeC == X_FLOAT16) {
        __half alpha2 = __float2half(alpha);
        __half beta2 = __float2half(beta);
        cublasSetMathMode(*handle, CUBLAS_TENSOR_OP_MATH);
        if (transposedA == X_NOTRANS && transposedB == X_NOTRANS)
            cublasGemmStridedBatchedEx(*handle, CUBLAS_OP_N, CUBLAS_OP_N, mc, nc, ma, (void*)&alpha2, b, CUDA_R_16F, mb, strideB, a, CUDA_R_16F, ma, strideA, (void*)&beta2, c, CUDA_R_16F, mc, strideC, count, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        else if (transposedA == X_TRANS && transposedB == X_NOTRANS)
            cublasGemmStridedBatchedEx(*handle, CUBLAS_OP_N, CUBLAS_OP_T, mc, nc, na, (void*)&alpha2, b, CUDA_R_16F, mb, strideB, a, CUDA_R_16F, ma, strideA, (void*)&beta2, c, CUDA_R_16F, mc, strideC, count, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        else if (transposedA == X_NOTRANS && transposedB == X_TRANS)
            cublasGemmStridedBatchedEx(*handle, CUBLAS_OP_T, CUBLAS_OP_N, mc, nc, ma, (void*)&alpha2, b, CUDA_R_16F, mb, strideB, a, CUDA_R_16F, ma, strideA, (void*)&beta2, c, CUDA_R_16F, mc, strideC, count, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        else if (transposedA == X_TRANS && transposedB == X_TRANS)
            cublasGemmStridedBatchedEx(*handle, CUBLAS_OP_T, CUBLAS_OP_T, mc, nc, na, (void*)&alpha2, b, CUDA_R_16F, mb, strideB, a, CUDA_R_16F, ma, strideA, (void*)&beta2, c, CUDA_R_16F, mc, strideC, count, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cublasSetMathMode(*handle, CUBLAS_DEFAULT_MATH);
    }
    else if (dataTypeA == X_FLOAT16 && dataTypeB == X_FLOAT16 && dataTypeC == X_FLOAT) {
        float alpha2 = (float)alpha;
        float beta2 = (float)beta;
        cublasSetMathMode(*handle, CUBLAS_TENSOR_OP_MATH);
        if (transposedA == X_NOTRANS && transposedB == X_NOTRANS)
            cublasGemmStridedBatchedEx(*handle, CUBLAS_OP_N, CUBLAS_OP_N, mc, nc, ma, (void*)&alpha2, b, CUDA_R_16F, mb, strideB, a, CUDA_R_16F, ma, strideA, (void*)&beta2, c, CUDA_R_32F, mc, strideC, count, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        else if (transposedA == X_TRANS && transposedB == X_NOTRANS)
            cublasGemmStridedBatchedEx(*handle, CUBLAS_OP_N, CUBLAS_OP_T, mc, nc, na, (void*)&alpha2, b, CUDA_R_16F, mb, strideB, a, CUDA_R_16F, ma, strideA, (void*)&beta2, c, CUDA_R_32F, mc, strideC, count, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        else if (transposedA == X_NOTRANS && transposedB == X_TRANS)
            cublasGemmStridedBatchedEx(*handle, CUBLAS_OP_T, CUBLAS_OP_N, mc, nc, ma, (void*)&alpha2, b, CUDA_R_16F, mb, strideB, a, CUDA_R_16F, ma, strideA, (void*)&beta2, c, CUDA_R_32F, mc, strideC, count, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        else if (transposedA == X_TRANS && transposedB == X_TRANS)
            cublasGemmStridedBatchedEx(*handle, CUBLAS_OP_T, CUBLAS_OP_T, mc, nc, na, (void*)&alpha2, b, CUDA_R_16F, mb, strideB, a, CUDA_R_16F, ma, strideA, (void*)&beta2, c, CUDA_R_32F, mc, strideC, count, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cublasSetMathMode(*handle, CUBLAS_DEFAULT_MATH);
    }
    else {
        ShowNTErrors("Unsupported data type!");
    }
}

/*
matrix multiplication via cuda version BLAS
*/
void _CudaBLASMatrixMULList(cublasHandle_t * handle,
                            const TensorList * a, MATRIX_TRANS_TYPE transposedA,
                            const TensorList * b, MATRIX_TRANS_TYPE transposedB,
                            TensorList * c,
                            int count, DTYPE alpha, DTYPE beta)
{
    CheckNTErrors((a && b && c), "Empty input lists!");
    CheckNTErrors((a->count == b->count && a->count == c->count), "Input lists must be of the same size!");

    if (a->count == 0)
        return;

    bool isUniform = true;
    bool isStrided = true;
    MTYPEINT strideA = MAX_INT;
    MTYPEINT strideB = MAX_INT;
    MTYPEINT strideC = MAX_INT;
    for (int i = 1; i < a->count; i++) {
        XTensor * aim = (XTensor*)a->GetItem(i - 1);
        XTensor * bim = (XTensor*)b->GetItem(i - 1);
        XTensor * cim = (XTensor*)c->GetItem(i - 1);
        XTensor * ai = (XTensor*)a->GetItem(i);
        XTensor * bi = (XTensor*)b->GetItem(i);
        XTensor * ci = (XTensor*)c->GetItem(i);
        if (!_IsSameShaped(aim, ai) ||
            !_IsSameShaped(bim, bi) ||
            !_IsSameShaped(cim, ci))
        {
            isUniform = false;
            break;
        }
        if (isStrided) {
            MTYPEINT gapA = MTYPEINT(ai->data) - MTYPEINT(aim->data);
            MTYPEINT gapB = MTYPEINT(bi->data) - MTYPEINT(bim->data);
            MTYPEINT gapC = MTYPEINT(ci->data) - MTYPEINT(cim->data);

            if (strideA == MAX_INT)
                strideA = gapA;
            if (strideB == MAX_INT)
                strideB = gapB;
            if (strideC == MAX_INT)
                strideC = gapC;

            if (strideA != gapA || strideB != gapB || strideC != gapC)
                isStrided = false;
        }
    }
    XTensor * a0 = (XTensor*)a->GetItem(0);
    XTensor * b0 = (XTensor*)b->GetItem(0);
    XTensor * c0 = (XTensor*)c->GetItem(0);

    if (isUniform) {
        XMem * mem = a0->mem;
        if (isStrided) {
            _CudaBLASMatrixMULBatchedStrided(handle,
                                             a0->data, transposedA, a0->dataType, strideA / a0->unitSize,
                                             b0->data, transposedB, b0->dataType, strideB / b0->unitSize,
                                             c0->data, c0->dataType, strideC / c0->unitSize, a->count,
                                             a0->dimSize[0], a0->dimSize[1],
                                             b0->dimSize[0], b0->dimSize[1],
                                             c0->dimSize[0], c0->dimSize[1], alpha, beta);
        }
        else {
            DTYPE ** ap = new DTYPE*[a->count];
            DTYPE ** bp = new DTYPE*[b->count];
            DTYPE ** cp = new DTYPE*[c->count];

            for (int i = 0; i < a->count; i++) {
                XTensor * ai = (XTensor*)a->GetItem(i);
                XTensor * bi = (XTensor*)b->GetItem(i);
                XTensor * ci = (XTensor*)c->GetItem(i);
                ap[i] = (DTYPE*)ai->data;
                bp[i] = (DTYPE*)bi->data;
                cp[i] = (DTYPE*)ci->data;
            }

            DTYPE ** apGPU = NULL;
            DTYPE ** bpGPU = NULL;
            DTYPE ** cpGPU = NULL;

            if (mem != NULL) {
                mem->SetPinBuf();
                apGPU = (DTYPE**)mem->AllocBuf(mem->devID, sizeof(DTYPE*) * a->count, 256);
                bpGPU = (DTYPE**)mem->AllocBuf(mem->devID, sizeof(DTYPE*) * a->count, 256);
                cpGPU = (DTYPE**)mem->AllocBuf(mem->devID, sizeof(DTYPE*) * a->count, 256);
            }
            else {
                apGPU = (DTYPE**)XMemAlloc(a0->devID, sizeof(DTYPE*) * a->count);
                bpGPU = (DTYPE**)XMemAlloc(a0->devID, sizeof(DTYPE*) * a->count);
                cpGPU = (DTYPE**)XMemAlloc(a0->devID, sizeof(DTYPE*) * a->count);
            }

            cudaMemcpy(apGPU, ap, sizeof(DTYPE*) * a->count, cudaMemcpyHostToDevice);
            cudaMemcpy(bpGPU, bp, sizeof(DTYPE*) * b->count, cudaMemcpyHostToDevice);
            cudaMemcpy(cpGPU, cp, sizeof(DTYPE*) * c->count, cudaMemcpyHostToDevice);

            _CudaBLASMatrixMULBatched(handle,
                                     (const void**)apGPU, transposedA, a0->dataType,
                                     (const void**)bpGPU, transposedB, b0->dataType,
                                     (void**)cpGPU, c0->dataType, a->count,
                                      a0->dimSize[0], a0->dimSize[1],
                                      b0->dimSize[0], b0->dimSize[1],
                                      c0->dimSize[0], c0->dimSize[1], alpha, beta);
            delete[] ap;
            delete[] bp;
            delete[] cp;

            if(mem != NULL)
                mem->BackToPinBuf();
            else {
                XMemFree(a0->devID, apGPU);
                XMemFree(a0->devID, bpGPU);
                XMemFree(a0->devID, cpGPU);
            }
        }

    }
    else {
        for (int i = 0; i < a->count; i++) {
            XTensor * ai = (XTensor*)a->GetItem(i);
            XTensor * bi = (XTensor*)b->GetItem(i);
            XTensor * ci = (XTensor*)c->GetItem(i);

            _CudaBLASMatrixMUL(handle,
                               ai->data, transposedA, ai->dataType,
                               bi->data, transposedB, bi->dataType,
                               ci->data, ci->dataType,
                               ai->dimSize[0], ai->dimSize[1],
                               bi->dimSize[0], bi->dimSize[1],
                               ci->dimSize[0], ci->dimSize[1], alpha, beta);
        }
    }
}

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)
