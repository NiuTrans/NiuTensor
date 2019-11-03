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
 * limitations under the License.b
 */

/*
 *
 * This is a wrapper of the BLAS (Basic Linear Algebra Subprograms http://www.netlib.org/blas/) 
 * libraries. By using BLAS, we can access very fast matrix operations although they
 * are also implemented in NiuTrans in a native manner. To use BLAS, 
 * specify USE_BLAS when compiling the code. 
 *
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2016-04-08
 *
 */

#ifndef __XBLAS_H__
#define __XBLAS_H__

/* the nts (NiuTrans.Tensor) namespace */
namespace nts{

/* some of the code below is from OpenBLAS (https://github.com/xianyi/OpenBLAS) */


#define OPENBLAS_CONST const
typedef int BLASINT;
typedef enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO      {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
typedef enum CBLAS_DIAG      {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
typedef enum CBLAS_SIDE      {CblasLeft=141, CblasRight=142} CBLAS_SIDE;


#if defined(USE_BLAS)
#ifdef OPENBLAS
#define XBLAS_SGEMM cblas_sgemm
#define XBLAS_DGEMM cblas_dgemm
#define XBLAS_SGER cblas_sger
#define XBLAS_DGER cblas_dger
#define XBLAS_SAXPY cblas_saxpy
#define XBLAS_DAXPY cblas_daxpy
#define XBLAS_SET_THREAD_NUM openblas_set_num_threads
#define XBLAS_GET_CORE_NUM openblas_get_num_procs
#endif
#ifdef MKL
#define XBLAS_SGEMM cblas_sgemm
#define XBLAS_DGEMM cblas_dgemm
#define XBLAS_SGER cblas_sger
#define XBLAS_DGER cblas_dger
#define XBLAS_SAXPY cblas_saxpy
#define XBLAS_DAXPY cblas_daxpy
#define XBLAS_SET_THREAD_NUM MKL_Set_Num_Threads
#define XBLAS_GET_CORE_NUM MKL_Get_Max_Threads
#endif
/* 
single/double-precision floating matrix-matrix multiplication (rank-3)
- SGEMM (ORDER, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
It implements C = \alpha * op(A)*op(B) + \beta * C
where A, B and C are matrices,
      \alpha and \beta are coefficients,
      TRANSA specifies we need a transposed matrix (op(A)=A**T); otherwise op(A) = A,
      M specifies the row number of op(A),
      N specifies the column number of op(B),
      K specifies the column number of op(A),
      LDA(=K) specifies the size of the first(or leading) dimension of A as declared in the calling (sub) program,
              E.g., if we are using CblasRowMajor, the leading dimension is the number of columns.
      LDB(=N) specifies the size of the first dimension of B as declared in the calling (sub) program,
      and LDC(=N) specifies the size of the first dimension of C as declared in the calling (sub) program.
*/
extern "C" void XBLAS_SGEMM(OPENBLAS_CONST enum CBLAS_ORDER, OPENBLAS_CONST enum CBLAS_TRANSPOSE, OPENBLAS_CONST enum CBLAS_TRANSPOSE,
                               OPENBLAS_CONST BLASINT, OPENBLAS_CONST BLASINT, OPENBLAS_CONST BLASINT, OPENBLAS_CONST float,  
                               OPENBLAS_CONST float *, OPENBLAS_CONST BLASINT,
                               OPENBLAS_CONST float *, OPENBLAS_CONST BLASINT, OPENBLAS_CONST float, 
                               float *, OPENBLAS_CONST BLASINT);

/* double-precision floating matrix-matrix multiplication */
extern "C" void XBLAS_DGEMM(OPENBLAS_CONST enum CBLAS_ORDER, OPENBLAS_CONST enum CBLAS_TRANSPOSE, OPENBLAS_CONST enum CBLAS_TRANSPOSE,
                               OPENBLAS_CONST BLASINT, OPENBLAS_CONST BLASINT, OPENBLAS_CONST BLASINT, OPENBLAS_CONST double,  
                               OPENBLAS_CONST double *, OPENBLAS_CONST BLASINT,
                               OPENBLAS_CONST double *, OPENBLAS_CONST BLASINT, OPENBLAS_CONST double, 
                               double *, OPENBLAS_CONST BLASINT);

/* 
single/double-precision floating vector-vector multiplication (rank-2)
- SGER (ORDER,M, N, ALPHA, X, INCX, Y, INCY, A, LDA)
It implements A = \alpha * X * (Y^T) + A
where X and Y are vectors with m and n elements respectively,
      A is an m by n matrix,
      \alpha is the scalar,
      INCX specifies the increment for the elements of X,
      INCY specifies the increment for the elements of Y,
      LDA specifies the size of the first(or leading) dimension of A as declared in the calling (sub) program,
          E.g., if we are using CblasRowMajor, the leading dimension is the number of columns of A.

*/
extern "C" void XBLAS_SGER(OPENBLAS_CONST enum CBLAS_ORDER, OPENBLAS_CONST BLASINT M, OPENBLAS_CONST BLASINT N, OPENBLAS_CONST float  alpha, 
                              OPENBLAS_CONST float *Y, OPENBLAS_CONST BLASINT, OPENBLAS_CONST float *, OPENBLAS_CONST BLASINT, 
                              float *, OPENBLAS_CONST BLASINT);

/* double-precision floating vector-vector multiplication (rank-1) */
extern "C" void XBLAS_DGER(OPENBLAS_CONST enum CBLAS_ORDER, OPENBLAS_CONST BLASINT M, OPENBLAS_CONST BLASINT N, OPENBLAS_CONST double  alpha, 
                              OPENBLAS_CONST double *Y, OPENBLAS_CONST BLASINT, OPENBLAS_CONST double *, OPENBLAS_CONST BLASINT, 
                              double *, OPENBLAS_CONST BLASINT);

/*
some description

*/
extern "C" void XBLAS_SAXPY(OPENBLAS_CONST BLASINT n, OPENBLAS_CONST float a, OPENBLAS_CONST float *x, OPENBLAS_CONST BLASINT incx, OPENBLAS_CONST float *y, OPENBLAS_CONST BLASINT  incy);

/* double-precision floating sumMe function */
extern "C" void XBLAS_DAXPY(OPENBLAS_CONST BLASINT n, OPENBLAS_CONST double a, OPENBLAS_CONST double *x, OPENBLAS_CONST BLASINT incx, OPENBLAS_CONST double *y, OPENBLAS_CONST BLASINT  incy);

/* set the number of threads */
extern "C" void XBLAS_SET_THREAD_NUM(int);

/* get the number of threads */
//extern "C" int (*XBLAS_GET_THREAD_NUM)();


/* get the number of physical processors (cores).*/
extern "C" int XBLAS_GET_CORE_NUM();

/* get the CPU corename */
//extern "C" char * (*XBLAS_GET_CORE_NAME)();

/* get the parallelization type used by OpenBLAS */
//extern "C" int (*XBLAS_GET_PARALLEL_TYPE)(void);


#if defined(CUDA_BLAS)

// Utilities and system includes
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>

/* Matrix multiplication */
extern void BLASMatrixMULS(int deviceID, float * a, float * b, float * c, int na, int ma, int nb, int mb, int nc, int mc, float alpha = 1.0F, float beta = 0);
extern void BLASMatrixMULD(int deviceID, double * a, double * b, double * c, int na, int ma, int nb, int mb, int nc, int mc, double alpha = 1.0F, double beta = 0);

#endif


#endif
} /* end of the nts (NiuTrans.Tensor) namespace */
#endif

