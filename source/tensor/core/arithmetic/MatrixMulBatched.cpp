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

#include "../../XTensor.h"
#include "../../XDevice.h"
#include "../../XName.h"
#include "../shape/IsSameShaped.h"
#include "MatrixMulBatched.h"
#include "XTensorBLAS.h"
#include "MatrixMul2D.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
matrix multiplication of the two tensors

for each 2-dimensional data array in a (denoted as ai) and
each 2-dimensional data array in b (denoted as bi), we have
ci = trans(ai) * trans(bi) * alpha + cm * beta
where trans() returns the transposed matrix if the flag is fired

>> a - tensor a
>> transposedA - indicates whether the matrices in a are transposed
>> b - tensor b
>> transposedB - indicates whether teh matrices in b are transposed
>> c - where we keep a*b
>> alpha - a coefficient
>> beta - another coefficient
>> parallelRunner - parallel processing module
*/
void _MatrixMulBatched(const XTensor * a, MATRIX_TRANS_TYPE transposedA,
                       const XTensor * b, MATRIX_TRANS_TYPE transposedB,
                       XTensor * c, DTYPE alpha, DTYPE beta, XPRunner * parallelRunner)
{
    CheckNTErrors((a && b && c), "Empty input tensors!");
    CheckNTErrors((a->dataType == b->dataType && a->dataType == c->dataType),
                  "Input tensors should have the same data type!");
    CheckNTErrors((a->order >= 2 && b->order >= 2 && c->order >= 2),
                  "Input tensors must have a order >= 2!");
    CheckNTErrors((a->order == b->order && a->order == c->order), 
                  "Input tensor and output tensor must have same order!");

    if (a->devID >= 0 || b->devID >= 0 || c->devID >= 0)
        _MatrixMulBatchedGPU(a, transposedA, b, transposedB, c, alpha, beta);
    else
        _MatrixMulBatchedCPU(a, transposedA, b, transposedB, c, alpha, beta);
}

/*
matrix multiplication of the two tensors
optimized for GPU

for each 2-dimensional data array in a (denoted as ai) and
each 2-dimensional data array in b (denoted as bi), we have
ci = trans(ai) * trans(bi) * alpha + cm * beta
where trans() returns the transposed matrix if the flag is fired

>> a - tensor a
>> transposedA - indicates whether the matrices in a are transposed
>> b - tensor b
>> transposedB - indicates whether teh matrices in b are transposed
>> c - where we keep a*b
>> alpha - a coefficient
>> beta - another coefficient
*/
void _MatrixMulBatchedGPU(const XTensor * a, MATRIX_TRANS_TYPE transposedA,
                          const XTensor * b, MATRIX_TRANS_TYPE transposedB,
                          XTensor * c, DTYPE alpha, DTYPE beta)
{
#ifdef USE_CUDA
    CheckNTErrors((a && b && c), "Empty input tensors!");
    CheckNTErrors((a->dataType == b->dataType && a->dataType == c->dataType),
                  "Input tensors should have the same data type!");
    CheckNTErrors((a->order >= 2 && b->order >= 2 && c->order >= 2),
                  "Input tensors must have a order >= 2!");
    CheckNTErrors((a->order == b->order && a->order == c->order), 
                  "Input tensor and output tensor must have same order!");
    CheckNTErrors(a->devID >= 0 && b->devID >= 0 && c->devID >= 0, "The tensors must be on GPUs");

    int an = transposedA == X_TRANS ? a->dimSize[a->order - 1] : a->dimSize[a->order - 2];
    int am = transposedA == X_TRANS ? a->dimSize[a->order - 2] : a->dimSize[a->order - 1];
    int bn = transposedB == X_TRANS ? b->dimSize[b->order - 1] : b->dimSize[b->order - 2];
    int bm = transposedB == X_TRANS ? b->dimSize[b->order - 2] : b->dimSize[b->order - 1];
    int cn = c->dimSize[c->order - 2];
    int cm = c->dimSize[c->order - 1];

    CheckNTErrors((am == bn && an == cn && bm == cm), "Unmatched tensors in multiplication!");

    int aBlockSize = a->dimSize[a->order - 1] * a->dimSize[a->order - 2];
    int bBlockSize = b->dimSize[b->order - 1] * b->dimSize[b->order - 2];
    int cBlockSize = c->dimSize[c->order - 1] * c->dimSize[c->order - 2];
    int aRealBlockSize = aBlockSize * a->unitSize;
    int bRealBlockSize = bBlockSize * b->unitSize;
    int cRealBlockSize = cBlockSize * c->unitSize;
    int blockNum = 1;

    for (int i = 0; i < a->order - 2; i++) {
        CheckNTErrors((a->dimSize[i] == c->dimSize[i]), "Incorrect tensor sizes!");
        CheckNTErrors((b->dimSize[i] == c->dimSize[i]), "Incorrect tensor sizes!");
        blockNum *= a->dimSize[i];
    }

    int devIDBackup = 0;
    ProtectCudaDev(a->devID, devIDBackup);

    cublasHandle_t * handle = a->mem != NULL ? a->mem->GetCublasHandle() : GDevs.GetCudaHandle(a->devID);
    _CudaBLASMatrixMULBatchedStrided(handle,
                                     a->data, transposedA, a->dataType, aBlockSize,
                                     b->data, transposedB, b->dataType, bBlockSize,
                                     c->data, c->dataType, cBlockSize, blockNum,
                                     a->dimSize[a->order - 2], a->dimSize[a->order - 1],
                                     b->dimSize[b->order - 2], b->dimSize[b->order - 1],
                                     c->dimSize[c->order - 2], c->dimSize[c->order - 1], alpha, beta);

    BacktoCudaDev(a->devID, devIDBackup);
#endif
}

/*
matrix multiplication of the two tensors
optimized for CPU

for each 2-dimensional data array in a (denoted as ai) and
each 2-dimensional data array in b (denoted as bi), we have
ci = trans(ai) * trans(bi) * alpha + cm * beta
where trans() returns the transposed matrix if the flag is fired

>> a - tensor a
>> transposedA - indicates whether the matrices in a are transposed
>> b - tensor b
>> transposedB - indicates whether teh matrices in b are transposed
>> c - where we keep a*b
>> alpha - a coefficient
>> beta - another coefficient
*/
void _MatrixMulBatchedCPU(const XTensor * a, MATRIX_TRANS_TYPE transposedA,
                          const XTensor * b, MATRIX_TRANS_TYPE transposedB,
                          XTensor * c, DTYPE alpha, DTYPE beta)
{
    CheckNTErrors(a && b && c, "Empty input tensors!");
    CheckNTErrors(a->dataType == b->dataType && a->dataType == c->dataType,
                 "Input tensors should have the same data type!");
    CheckNTErrors(a->order >= 2 && b->order >= 2 && c->order >= 2,
                 "Input tensors must have a order >= 2!");
    CheckNTErrors(a->order == b->order && a->order == c->order, 
                 "Input tensor and output tensor must have same order!");


    int an = transposedA == X_TRANS ? a->dimSize[a->order - 1] : a->dimSize[a->order - 2];
    int am = transposedA == X_TRANS ? a->dimSize[a->order - 2] : a->dimSize[a->order - 1];
    int bn = transposedB == X_TRANS ? b->dimSize[b->order - 1] : b->dimSize[b->order - 2];
    int bm = transposedB == X_TRANS ? b->dimSize[b->order - 2] : b->dimSize[b->order - 1];
    int cn = c->dimSize[c->order - 2];
    int cm = c->dimSize[c->order - 1];

    CheckNTErrors(am == bn && an == cn && bm == cm, "Unmatched tensors in multiplication!");

    int aBlockSize = a->dimSize[a->order - 1] * a->dimSize[a->order - 2];
    int bBlockSize = b->dimSize[b->order - 1] * b->dimSize[b->order - 2];
    int cBlockSize = c->dimSize[c->order - 1] * c->dimSize[c->order - 2];
    int aRealBlockSize = aBlockSize * a->unitSize;
    int bRealBlockSize = bBlockSize * b->unitSize;
    int cRealBlockSize = cBlockSize * c->unitSize;
    int blockNum = 1;

    for (int i = 0; i < a->order - 2; i++) {
        CheckNTErrors((a->dimSize[i] == c->dimSize[i]), "Incorrect tensor sizes!");
        CheckNTErrors((b->dimSize[i] == c->dimSize[i]), "Incorrect tensor sizes!");
        blockNum *= a->dimSize[i];
    }

    int aDimSize[2] = {-a->dimSize[a->order - 2], a->dimSize[a->order - 1]};
    int bDimSize[2] = {-b->dimSize[b->order - 2], b->dimSize[b->order - 1]};
    int cDimSize[2] = {-c->dimSize[c->order - 2], c->dimSize[c->order - 1]};

    XTensor * ai = NewTensor2DV2(aDimSize[0], aDimSize[1], a->dataType, a->devID, a->mem);
    XTensor * bi = NewTensor2DV2(bDimSize[0], bDimSize[1], b->dataType, b->devID, b->mem);
    XTensor * ci = NewTensor2DV2(cDimSize[0], cDimSize[1], c->dataType, c->devID, c->mem);

    for (int i = 0; i < blockNum; i++) {
        ai->data = (char*)a->data + i * aRealBlockSize;
        bi->data = (char*)b->data + i * bRealBlockSize;
        ci->data = (char*)c->data + i * cRealBlockSize;
#ifdef USE_BLAS
        _MatrixMULCPU(ai, transposedA, bi, transposedB, ci, alpha, beta);
#else
        _MatrixMul2D(ai, transposedA, bi, transposedB, ci, alpha, beta);
#endif
    }

    ai->data = NULL;
    bi->data = NULL;
    ci->data = NULL;
    delete ai;
    delete bi;
    delete ci;
}

/*
matrix multiplication in batch mode for list inputs (BLAS)
c_i = trans(a_i) * trans(b_i) * \alpha + c_i * \beta for each i in [0,count-1]
>> a - list of input matrices (2d tensors)
>> transposedA - indicate whether the matrix a is transposed
>> b - another list of input matrices (2d tensors)
>> transposedB - indicate whether the matrix b is transposed
>> c - output matrix (2d tensor)
>> alpha - scalar
>> beta - scalar
*/
void _MatrixMulBatchedCPU(const TensorList * a, MATRIX_TRANS_TYPE transposedA,
                          const TensorList * b, MATRIX_TRANS_TYPE transposedB,
                          TensorList * c, DTYPE alpha, DTYPE beta)
{
    CheckNTErrors(a && b && c, "Empty input lists!");
    CheckNTErrors(a->count == b->count && a->count == c->count, "Input lists must be of the same size!");

    if (a->count == 0)
        return;

    bool isUniform = true;
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
    }

    for (int i = 0; i < a->count; i++) {
        XTensor * ai = (XTensor*)a->GetItem(i);
        XTensor * bi = (XTensor*)b->GetItem(i);
        XTensor * ci = (XTensor*)c->GetItem(i);
        CheckNTErrors((ai->order == 2), "2d tensor (i.e., matrix) is required!");
        CheckNTErrors((bi->order == 2), "2d tensor (i.e., matrix) is required!");
        CheckNTErrors((ci->order == 2), "2d tensor (i.e., matrix) is required!");
#ifdef USE_BLAS
            _MatrixMULCPU(ai, transposedA, bi, transposedB, ci, alpha, beta);
#else
        _MatrixMul2D(ai, transposedA, bi, transposedB, ci, alpha, beta);
#endif
    }
}

/*
matrix multiplication of the two tensors (do it on site)
c = trans(a) * trans(b) * alpha
make a new tensor to keep the result and return it

for each 2-dimensional data array in a (denoted as ai) and
each 2-dimensional data array in b (denoted as bi), we have
ci = trans(ai) * trans(bi) * alpha + cm * beta
where trans() returns the transposed matrix if the flag is fired.

>> a - tensor a
>> transposedA - indicates whether the matrices in a are transposed
>> b - tensor b
>> transposedB - indicates whether teh matrices in b are transposed
>> alpha - a coefficient
>> parallelRunner - parallel processing module
<< return - the result of matrix multiplication of the two tensors
*/
XTensor MatrixMulBatched(const XTensor &a, MATRIX_TRANS_TYPE transposedA, const XTensor &b, MATRIX_TRANS_TYPE transposedB,
                         DTYPE alpha, XPRunner * parallelRunner)
{
    CheckNTErrors(a.dataType == b.dataType, "Input tensors should have the same data type!");
    CheckNTErrors(a.order >= 2 && b.order >= 2, "Input tensors must have a order >= 2!");
    CheckNTErrors(a.order == b.order, "Input tensor and output tensor must have same order!");

    int an = transposedA == X_TRANS ? a.dimSize[a.order - 1] : a.dimSize[a.order - 2];
    int am = transposedA == X_TRANS ? a.dimSize[a.order - 2] : a.dimSize[a.order - 1];
    int bn = transposedB == X_TRANS ? b.dimSize[b.order - 1] : b.dimSize[b.order - 2];
    int bm = transposedB == X_TRANS ? b.dimSize[b.order - 2] : b.dimSize[b.order - 1];

    CheckNTErrors(am == bn, "Unmatched tensors in multiplication!");

    int order = a.order;
    int sub = 0;
    int * dimSize = new int[order];
    for (int i = 0; i < a.order - 2; i++)
        dimSize[sub++] = a.dimSize[i];
    dimSize[sub++] = an;
    dimSize[sub++] = bm;

    float dr = (!a.isSparse || !b.isSparse) ? 1.0F : MAX(a.denseRatio, b.denseRatio);
    XTensor c(order, dimSize, a.dataType, dr, a.devID, a.mem);
    c.SetTMPFlag();

    /*call _MatrixMulBatched function */
    _MatrixMulBatched(&a, transposedA, &b, transposedB, &c, alpha, 0, parallelRunner);

    /* tensor connections */
    if (a.enableGrad && b.enableGrad) {
        XLink::MakeLink(&a, &b, &c, MATH_MATRIXMULBATCHED);
        XLink::AddParamToHeadTrans(&c, transposedA);
        XLink::AddParamToHeadTrans(&c, transposedB);
        XLink::AddParamToHead(&c, alpha);
    }

    /* destroy variables */
    delete[] dimSize;

    return c;
}

/*
matrix multiplication of the two tensors (do it on site)
c = a * b * alpha
make a new tensor to keep the result and return it

for each 2-dimensional data array in a (denoted as ai) and
each 2-dimensional data array in b (denoted as bi), we have
ci = ai * bi * alpha + cm * beta

>> a - tensor a
>> b - tensor b
>> alpha - a coefficient
>> parallelRunner - parallel processing module
<< return - the result of matrix multiplication of the two tensors
*/
XTensor MatrixMulBatched(const XTensor &a, const XTensor &b,
                         DTYPE alpha, XPRunner * parallelRunner)
{
    CheckNTErrors(a.dataType == b.dataType, "Input tensors should have the same data type!");
    CheckNTErrors(a.order >= 2 && b.order >= 2, "Input tensors must have a order >= 2!");
    CheckNTErrors(a.order == b.order, "Input tensor and output tensor must have same order!");

    int an = a.dimSize[a.order - 2];
    int am = a.dimSize[a.order - 1];
    int bn = b.dimSize[b.order - 2];
    int bm = b.dimSize[b.order - 1];

    CheckNTErrors(am == bn, "Unmatched tensors in multiplication!");

    int order = a.order;
    int sub = 0;
    int * dimSize = new int[order];
    for (int i = 0; i < a.order - 2; i++)
        dimSize[sub++] = a.dimSize[i];
    dimSize[sub++] = an;
    dimSize[sub++] = bm;

    float dr = (!a.isSparse || !b.isSparse) ? 1.0F : MAX(a.denseRatio, b.denseRatio);
    XTensor c(order, dimSize, a.dataType, dr, a.devID, a.mem);
    c.SetTMPFlag();

    /*call _MatrixMulBatched function */
    _MatrixMulBatched(&a, X_NOTRANS, &b, X_NOTRANS, &c, alpha, 0, parallelRunner);

    /* tensor connections */
    if (a.enableGrad && b.enableGrad) {
        XLink::MakeLink(&a, &b, &c, MATH_MATRIXMULBATCHED);
        XLink::AddParamToHeadTrans(&c, X_NOTRANS);
        XLink::AddParamToHeadTrans(&c, X_NOTRANS);
        XLink::AddParamToHead(&c, alpha);
    }

    /* destroy variables */
    delete[] dimSize;

    return c;
}

} // namespace nts(NiuTrans.Tensor)
