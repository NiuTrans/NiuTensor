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

#include "../../XDevice.h"
#include "../../XTensor.h"
#include "MatrixMul2D.h"
#include "MatrixMul2D.cuh"
#include "XTensorBLAS.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA
/*
mutilication of a dense matrix with a sparse matrix
c = a * b * \alpha

>> a - a dense matrix
>> transposedA - indicates whether a is transposed
>> aColSize - column size of matrix a
>> aRowSize - row size of matrix a
>> b - a sparse matrix
>> transposedB - indicates whether b is transposed
>> bNonZeroNum - number of non-zero items in b
>> bColSize - column size of matrix b
>> bRowSize - row size of matrix b
>> c - the resulting (dense) matrix
>> cColSize - column size of matrix c
>> cRowSize - row size of matrix c
>> alpha - the scaling factor
*/
__global__
void KernelMatrixMulDenseMSparseMV2(DTYPE * a, MATRIX_TRANS_TYPE transposedA, int aColSize, int aRowSize,
                                    void * b, MATRIX_TRANS_TYPE transposedB, int bNonZeroNum, int bColSize, int bRowSize,
                                    DTYPE * c, int cColSize, int cRowSize, DTYPE alpha)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    char * bData = (char*)b;
    int tupleSize = sizeof(int) + sizeof(DTYPE);

    for (int k = 0; k < bNonZeroNum; k += blockDim.x) {
        __shared__ int   bEntryRow[MAX_CUDA_THREAD_NUM_PER_BLOCK];
        __shared__ int   bEntryCol[MAX_CUDA_THREAD_NUM_PER_BLOCK];
        __shared__ DTYPE bValue[MAX_CUDA_THREAD_NUM_PER_BLOCK];

        if (k + threadIdx.x < bNonZeroNum) {
            /* load the sub-block of the sparse matrix b */
            int key = *(int*)(bData + tupleSize * (k + threadIdx.x));

            bEntryRow[threadIdx.x] = key / bRowSize;
            bEntryCol[threadIdx.x] = key % bRowSize;
            bValue[threadIdx.x] = *(DTYPE*)(bData + tupleSize * (k + threadIdx.x) + sizeof(int));
        }

        /* synchronize to make sure the sub-block of the sparse matrix b is loaded */
        __syncthreads();

        if (i < cColSize) {
            if (transposedA == X_NOTRANS && transposedB == X_NOTRANS) {
                for (int m = 0; m < blockDim.x && k + m < bNonZeroNum; m++) {
                    DTYPE * aRow = a + aRowSize * i;
                    c[i * cRowSize + bEntryCol[m]] += aRow[bEntryRow[m]] * bValue[m] * alpha;
                }
            }
            else if (transposedA == X_TRANS && transposedB == X_NOTRANS) {
                for (int m = 0; m < blockDim.x && k + m < bNonZeroNum; m++) {
                    DTYPE * aCol = a + i;
                    c[i * cRowSize + bEntryCol[m]] += aCol[bEntryRow[m] * aRowSize] * bValue[m] * alpha;
                }
            }
            else if (transposedA == X_NOTRANS && transposedB == X_TRANS) {
                for (int m = 0; m < blockDim.x && k + m < bNonZeroNum; m++) {
                    DTYPE * aRow = a + aRowSize * i;
                    c[i * cRowSize + bEntryRow[m]] += aRow[bEntryCol[m]] * bValue[m] * alpha;
                }
            }
            else if (transposedA == X_TRANS && transposedB == X_TRANS) {
                for (int m = 0; m < blockDim.x && k + m < bNonZeroNum; m++) {
                    DTYPE * aCol = a + i;
                    c[i * cRowSize + bEntryRow[m]] += aCol[bEntryCol[m] * aRowSize] * bValue[m] * alpha;
                }
            }
        }

        /* synchronize to the preceding computation is done before loading new sub-blocks */
        __syncthreads();
    }
}


/*
matrix multiplication (for 2d tensors) (cuda version)

c = trans(a) * trans(b) * alpha + c * beta
where trans() return the transposed matrix if the flag is fired

>> a - tensor a
>> transposedA - indicates whether the matrices in a are transposed
>> b - tensor b
>> transposedB - indicates whether teh matrices in b are transposed
>> c - where we put a*b
>> alpha - a coefficient
>> beta - another coefficient
>> stream - the string for creating the job pipeline
*/
void _CudaMatrixMul2D(const XTensor * a, MATRIX_TRANS_TYPE transposedA,
                      const XTensor * b, MATRIX_TRANS_TYPE transposedB,
                      XTensor * c, DTYPE alpha, DTYPE beta, XStream * stream)
{
    int an = transposedA == X_TRANS ? a->dimSize[1] : a->dimSize[0];
    int am = transposedA == X_TRANS ? a->dimSize[0] : a->dimSize[1];
    int bn = transposedB == X_TRANS ? b->dimSize[1] : b->dimSize[0];
    int bm = transposedB == X_TRANS ? b->dimSize[0] : b->dimSize[1];
    int cn = c->dimSize[0];
    int cm = c->dimSize[1];

    CheckNTErrors((a && b && c),
        "Empty matrices in multiplication!");

    CheckNTErrors((am == bn && an == cn && bm == cm),
        "Unmatched matrices in multiplication!");

    CheckNTErrors((a->devID >= 0), "Cuda version matrix mutiplication must be run on GPUs.");

    CheckNTErrors(a->devID == b->devID && a->devID == c->devID,
        "Matrices used in multiplication are not on the same GPU.");

    int devIDBackup = 0;
    ProtectCudaDev(a->devID, devIDBackup);

    /* a dense matrix multiply a dense matrix */
    if (!a->isSparse && !b->isSparse) {
        CheckNTErrors((!c->isSparse), "Illegal use of sparse matrix in multiplication!");

        cublasHandle_t * handle = a->mem == NULL ? GDevs.GetCudaHandle(a->devID) : a->mem->GetCublasHandle();

        /* !!!! might have problems */
        if (stream != NULL)
            cublasSetStream(*handle, stream->stream);

        if (a->dataType == X_FLOAT && b->dataType == X_FLOAT && c->dataType == X_FLOAT) {
            _CudaBLASMatrixMUL(handle, a->data, transposedA, a->dataType, 
                               b->data, transposedB, a->dataType, c->data, c->dataType,
                               a->dimSize[0], a->dimSize[1], 
                               b->dimSize[0], b->dimSize[1], 
                               c->dimSize[0], c->dimSize[1],
                               alpha, beta);
        }
        else {
            // TODO!!
            ShowNTErrors("TODO!");
        }
    }
    /* a dense matrix multiply a sparse matrix */
    else if (!a->isSparse && b->isSparse) {

        CheckNTErrors(!c->isSparse, "Illegal use of sparse matrix in multiplication!");
        CheckNTErrors((beta == 0 || beta == 1.0), "beta must be 0 or 1.");

        if (a->dataType == DEFAULT_DTYPE && b->dataType == DEFAULT_DTYPE && c->dataType == DEFAULT_DTYPE) {
            int gridSize[3], blockSize[3];

            GDevs.GetCudaThread(c->devID, a->dimSize[0], gridSize, blockSize);

            dim3 blocks(gridSize[0]);
            dim3 threads(blockSize[0]);

            void * bData = (void*)((char*)b->data + sizeof(int));

            if (beta == 0)
                c->SetZeroAll();
            else if (beta != 1.0F) {
                ShowNTErrors("TODO!");
            }

            KernelMatrixMulDenseMSparseMV2 << <blocks, threads >> >((DTYPE*)a->data, transposedA, a->dimSize[0], a->dimSize[1],
                bData, transposedB, b->unitNumNonZero, b->dimSize[0], b->dimSize[1],
                (DTYPE*)c->data, c->dimSize[0], c->dimSize[1], alpha);
        }
        else {
            // TODO!!
            ShowNTErrors("TODO!");
        }

    }
    else {
        // TODO!!
        ShowNTErrors("TODO!");
    }

    BacktoCudaDev(a->devID, devIDBackup);
}
#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)