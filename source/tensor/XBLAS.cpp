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
 * This is a wrapper of the BLAS (Basic Linear Algebra Subprograms http://www.netlib.org/blas/) 
 * libraries. By using BLAS, we can access very fast matrix operations although they
 * are also implemented in NiuTrans in a native manner. To use BLAS, 
 * set USE_BLAS. 
 *
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2016-04-08
 *
 */

#ifdef WIN32
#include <wtypes.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include "XBLAS.h"
#include "XGlobal.h"

/* the nts (NiuTrans.Tensor) namespace */
namespace nts{

#ifdef WIN32
HINSTANCE hBLASDll;
#endif


/* single-precision floating matrix-matrix multiplication */
void (*XBLAS_SGEMM)(OPENBLAS_CONST enum CBLAS_ORDER, OPENBLAS_CONST enum CBLAS_TRANSPOSE, OPENBLAS_CONST enum CBLAS_TRANSPOSE,
                    OPENBLAS_CONST BLASINT, OPENBLAS_CONST BLASINT, OPENBLAS_CONST BLASINT, OPENBLAS_CONST float,  
                    OPENBLAS_CONST float *, OPENBLAS_CONST BLASINT,
                    OPENBLAS_CONST float *, OPENBLAS_CONST BLASINT, OPENBLAS_CONST float, 
                    float *, OPENBLAS_CONST BLASINT);

/* double-precision floating matrix-matrix multiplication */
void (*XBLAS_DGEMM)(OPENBLAS_CONST enum CBLAS_ORDER, OPENBLAS_CONST enum CBLAS_TRANSPOSE, OPENBLAS_CONST enum CBLAS_TRANSPOSE,
                    OPENBLAS_CONST BLASINT, OPENBLAS_CONST BLASINT, OPENBLAS_CONST BLASINT, OPENBLAS_CONST double,  
                    OPENBLAS_CONST double *, OPENBLAS_CONST BLASINT,
                    OPENBLAS_CONST double *, OPENBLAS_CONST BLASINT, OPENBLAS_CONST double, 
                    double *, OPENBLAS_CONST BLASINT);

/* single-precision floating vector-vector multiplication (rank-1) */
void (*XBLAS_SGER)(OPENBLAS_CONST enum CBLAS_ORDER, OPENBLAS_CONST BLASINT M, OPENBLAS_CONST BLASINT N, OPENBLAS_CONST float  alpha, 
                   OPENBLAS_CONST float *Y, OPENBLAS_CONST BLASINT, OPENBLAS_CONST float *, OPENBLAS_CONST BLASINT, 
                   float *, OPENBLAS_CONST BLASINT);

/* double-precision floating vector-vector multiplication (rank-1) */
void (*XBLAS_DGER)(OPENBLAS_CONST enum CBLAS_ORDER, OPENBLAS_CONST BLASINT M, OPENBLAS_CONST BLASINT N, OPENBLAS_CONST double  alpha, 
                   OPENBLAS_CONST double *Y, OPENBLAS_CONST BLASINT, OPENBLAS_CONST double *, OPENBLAS_CONST BLASINT, 
                   double *, OPENBLAS_CONST BLASINT);

/* set the number of threads */
void (*XBLAS_SET_THREAD_NUM)(int);

/* get the number of threads */
//int (*XBLAS_GET_THREAD_NUM)();


/* get the number of physical processors (cores).*/
int (*XBLAS_GET_CORE_NUM)();


/* get the CPU corename */
//char * (*XBLAS_GET_CORE_NAME)();

/* get the parallelization type used by OpenBLAS */
//int (*XBLAS_GET_PARALLEL_TYPE)(void);


#if defined(USE_BLAS)

/* load some stuff for BLAS */
void LoadBLAS(const char * dllFileName)
{
#ifndef CUDA_BLAS
#ifdef _WIN32

#if defined(OPENBLAS)
    /* non-ascii characters are not supported yet */
    wchar_t * fn = new wchar_t[strlen(dllFileName) + 1];
    memset(fn, 0, sizeof(wchar_t) * (strlen(dllFileName) + 1));
    for(int i = 0; i < strlen(dllFileName); i++)
        fn[i] = dllFileName[i];

    hBLASDll = LoadLibrary((LPCWSTR)fn);

    if(!hBLASDll){
        XPRINT1(0, stderr, "[LoadBLAS] Error! Cannot load dll %s!\n", dllFileName);
        exit(1);
    }

    /* matrix-matrix multiplicatoin */
    (FARPROC&)XBLAS_SGEMM = GetProcAddress(hBLASDll, "cblas_sgemm");
    (FARPROC&)XBLAS_DGEMM = GetProcAddress(hBLASDll, "cblas_dgemm");

    /* vector-vector multiplication */
    (FARPROC&)XBLAS_SGER = GetProcAddress(hBLASDll, "cblas_sger");
    (FARPROC&)XBLAS_DGER = GetProcAddress(hBLASDll, "cblas_dger");

    /* multi-threading */
    (FARPROC&)XBLAS_SET_THREAD_NUM = GetProcAddress(hBLASDll, "openblas_set_num_threads");
    //(FARPROC&)XBLAS_SET_THREAD_NUM = GetProcAddress(hBLASDll, "goto_set_num_threads");
    //(FARPROC&)XBLAS_GET_THREAD_NUM = GetProcAddress(hBLASDll, "openblas_get_num_threads");
    (FARPROC&)XBLAS_GET_CORE_NUM = GetProcAddress(hBLASDll, "openblas_get_num_procs");
    //(FARPROC&)XBLAS_GET_CORE_NAME = GetProcAddress(hBLASDll, "openblas_get_corename");
    //(FARPROC&)XBLAS_GET_PARALLEL_TYPE = GetProcAddress(hBLASDll, "openblas_get_parallel");

    delete[] fn;
#endif // defined(OPENBLAS)

#if defined(MKL)
    /* non-ascii characters are not supported yet */
    wchar_t * fn = new wchar_t[strlen(dllFileName) + 1];
    memset(fn, 0, sizeof(wchar_t) * (strlen(dllFileName) + 1));
    for(int i = 0; i < strlen(dllFileName); i++)
        fn[i] = dllFileName[i];

    hBLASDll = LoadLibrary((LPCWSTR)fn);

    if(!hBLASDll){
        XPRINT1(0, stderr, "[LoadBLAS] Error! Cannot load dll %s!\n", dllFileName);
        exit(1);
    }

    /* matrix-matrix multiplicatoin */
    (FARPROC&)XBLAS_SGEMM = GetProcAddress(hBLASDll, "cblas_sgemm");
    (FARPROC&)XBLAS_DGEMM = GetProcAddress(hBLASDll, "cblas_dgemm");

    /* vector-vector multiplication */
    (FARPROC&)XBLAS_SGER = GetProcAddress(hBLASDll, "cblas_sger");
    (FARPROC&)XBLAS_DGER = GetProcAddress(hBLASDll, "cblas_dger");

    /* multi-threading */
    (FARPROC&)XBLAS_SET_THREAD_NUM = GetProcAddress(hBLASDll, "MKL_Set_Num_Threads");
    (FARPROC&)XBLAS_GET_CORE_NUM   = GetProcAddress(hBLASDll, "MKL_Get_Max_Threads");
#endif // defined(MKL)

#else // _WIN32

    XBLAS_SGEMM = &cblas_sgemm;
    XBLAS_DGEMM = &cblas_dgemm;
    XBLAS_SGER  = &cblas_sger;
    XBLAS_DGER  = &cblas_dger;
#if defined(OPENBLAS)
    XBLAS_SET_THREAD_NUM    = &openblas_set_num_threads;
    XBLAS_GET_CORE_NUM      = &openblas_get_num_procs;
#endif // defined(OPENBLAS)
#if defined(MKL)
    XBLAS_SET_THREAD_NUM    = &mkl_set_num_threads;
    XBLAS_GET_CORE_NUM      = &mkl_get_max_num_threads;
#endif // defined(MKL)

#endif // _WIN32

    XBLAS_SET_THREAD_NUM(1);
#endif // ndef(CUDA_BLAS)
}

/* unload the libs */
void UnloadBLAS()
{
#ifdef _WIN32

    if(!FreeLibrary(hBLASDll)){
        XPRINT(0, stderr, "[UnloadBLAS] Error! Cannot free the BLAS dll!\n");
        exit(1);
    }

#else

#endif
}

#else  // undefined(USE_BLAS) || undefined(OPENBLAS)

void LoadBLAS(const char * dllFileName)
{
    XPRINT(0, stderr, "[LoadBLAS] Error! No Blas lib is available. Please use OPENBLAS or MKL!\n");
    exit(1);
}

void UnloadBLAS()
{
    XPRINT(0, stderr, "[UnloadBLAS] Error! No Blas lib is available. Please use OPENBLAS or MKL!\n");
    exit(1);
}

#endif // defined(USE_BLAS) && defined(OPENBLAS)

} /* end of the nts (NiuTrans.Tensor) namespace */