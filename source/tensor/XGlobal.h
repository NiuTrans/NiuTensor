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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2016-01-20
 *
 */

#ifndef __XGLOBAL_H__
#define __XGLOBAL_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#ifndef WIN32
#include <sys/time.h>
#include <unistd.h>
#endif

// the CUDA stuff
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

/* the nts (NiuTrans.Tensor) namespace */
namespace nts {

#define _XINLINE_  

//#define DOUBELPRICSION

#ifdef DOUBELPRICSION
#define DTYPE double
#define DTYPE_MIN (DTYPE)-1.79E+308
#else
#define DTYPE float
#define DTYPE_MIN (DTYPE)-3.40E+38
#endif

#define LOGPROB_MIN (DTYPE)-2E+1
#define GRAD_MAX (DTYPE)1E+5

#if WIN32
#define DELIMITER '\\'
#else
#define DELIMITER '/'
#endif

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? a : b)
#endif
#ifndef MAX
#define MAX(a,b) ((a) > (b) ? a : b)
#endif

#define __FILENAME__ ( strrchr(__FILE__, DELIMITER) != NULL ? strrchr(__FILE__, DELIMITER)+1 : __FILE__ )

#define CheckNTErrors(x, msg) \
{ \
    if(!(x)) \
    { \
        fprintf(stderr, "[ERROR] calling '%s' (%s line %d): %s\n", #x, __FILENAME__, __LINE__, msg); \
        exit(1); \
    } \
} \

#define CheckNTErrorsV0(x) \
{ \
    if(!(x)) \
    { \
        fprintf(stderr, "[ERROR] calling '%s' (%s line %d): %s\n", #x, __FILENAME__, __LINE__); \
        exit(1); \
    } \
} \

#define ShowNTErrors(msg) \
{ \
    { \
        fprintf(stderr, "[ERROR] (%s line %d): %s\n", __FILENAME__, __LINE__, msg); \
        exit(1); \
    } \
} \

#define MAX_FILE_NAME_LENGTH 1024 * 2
#define MAX_LINE_LENGTH 1024*1024
#define MAX_SENTENCE_LEN 512
#define X_MILLION 1000000
#define MAX_INT 2147483647
#define MAX_FLOAT FLT_MAX
#define FIELD_SEP " ||| "
#define FLOAT_MIN float(-1.0E38)
#define FLOAT16_MIN float(-65504)
#define MILLION 1000000
#define LOG_E_10 2.302585
#define LEADING_DIM 1

/* cuda setting */
#define MAX_CUDA_THREAD_NUM_PER_BLOCK 512
#define MIN_CUDA_SHARED_MEM_COL_SIZE 8
#define MAX_MODEL_NUM 512
#define SHARED_MEMORY_SIZE (48 << 10)

/* memory pool setting */
extern int MAX_MEM_BLOCK_NUM;
extern int MAX_MEM_BLOCK_SIZE;
extern int MIN_MEM_BLOCK_SIZE;
extern int MINOR_MEM_BLOCK_SIZE;
extern int MAX_MEM_BUF_SIZE;
extern int MIN_MEM_BUF_SIZE;
extern int TRAINING_SAMPLE_BUF_SIZE;

extern int CONST_MINUSONE;
extern bool CONST_TRUE;

//#define USE_CUDA_RESURSION 1

#define NIUTRANSNNDEBUG

extern int verboseLevel;

#define FFLUSH(FILEH) \
{ \
    fflush(FILEH); \
} \

#define XPRINT(VERBOSE,FILEH,STR) {if(VERBOSE<=verboseLevel) {fprintf(FILEH,STR);FFLUSH(FILEH);}}
#define XPRINT1(VERBOSE,FILEH,STR,ARG) {if(VERBOSE<=verboseLevel) {fprintf(FILEH,STR,ARG);FFLUSH(FILEH);}}
#define XPRINT2(VERBOSE,FILEH,STR,ARG,ARG2) {if(VERBOSE<=verboseLevel) {fprintf(FILEH,STR,ARG,ARG2);FFLUSH(FILEH);}}
#define XPRINT3(VERBOSE,FILEH,STR,ARG,ARG2,ARG3) {if(VERBOSE<=verboseLevel) {fprintf(FILEH,STR,ARG,ARG2,ARG3);FFLUSH(FILEH);}}
#define XPRINT4(VERBOSE,FILEH,STR,ARG,ARG2,ARG3,ARG4) {if(VERBOSE<=verboseLevel) {fprintf(FILEH,STR,ARG,ARG2,ARG3,ARG4);FFLUSH(FILEH);}}
#define XPRINT5(VERBOSE,FILEH,STR,ARG,ARG2,ARG3,ARG4,ARG5) {if(VERBOSE<=verboseLevel) {fprintf(FILEH,STR,ARG,ARG2,ARG3,ARG4,ARG5);FFLUSH(FILEH);}}
#define XPRINT6(VERBOSE,FILEH,STR,ARG,ARG2,ARG3,ARG4,ARG5,ARG6) {if(VERBOSE<=verboseLevel) {fprintf(FILEH,STR,ARG,ARG2,ARG3,ARG4,ARG5,ARG6);FFLUSH(FILEH);}}
#define XPRINT7(VERBOSE,FILEH,STR,ARG,ARG2,ARG3,ARG4,ARG5,ARG6,ARG7) {if(VERBOSE<=verboseLevel) {fprintf(FILEH,STR,ARG,ARG2,ARG3,ARG4,ARG5,ARG6,ARG7);FFLUSH(FILEH);}}
#define XPRINT8(VERBOSE,FILEH,STR,ARG,ARG2,ARG3,ARG4,ARG5,ARG6,ARG7,ARG8) {if(VERBOSE<=verboseLevel) {fprintf(FILEH,STR,ARG,ARG2,ARG3,ARG4,ARG5,ARG6,ARG7,ARG8);FFLUSH(FILEH);}}

#define B2I(V) V == 0 ? false : true

#define MODX(a, b) int(b == 0 ? a : a - floor(double(a)/b) * b)

/* BLAS interfaces */
#ifdef DOUBELPRICSION
#define GEMM XBLAS_DGEMM
#define AXPY XBLAS_DAXPY
#else
#define GEMM XBLAS_SGEMM
#define AXPY XBLAS_SAXPY
#endif

extern void InitGlobalAll();

extern FILE * tmpLog;

extern int dEdWCount;
extern FILE * tF;
extern int tmpCountV2;
extern int nnnTotal;

} /* end of the nts (NiuTrans.Tensor) namespace */

#endif
