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
#include "MatrixMul.h"
#include "MatrixMul2D.h"
#include "XTensorBLAS.h"
#include "MatrixMulBatched.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
matrix multiplication c = trans(a) * trans(b) * alpha + c * beta

For the input tensors a and b, we perform matrix multiplication on the first two dimentsions. 
E.g., let A be a tensor of size y * z * m and B be a tensor of size x * y * n. 
For A * B, we go over each order-2 tensor of A (of size x * y) and each order-2 tensor B (of size z * x), 
like this c_{i,j} = trans(ai) * trans(bj) * alpha + c_{i,j} * beta
where trans() returns the transposed matrix if the flag is fired, ai is the i-th element tensor of A, 
bj is the j-th element tensor of B, and c_{i,j} is the (i,j) element tensor of the result C. 
C should be a tensor of z * x * n * m. 
Obviously C = A * B performs normal matrix multiplication if A = y * z and B = x * y.

>> a - tensor a
>> transposedA - indicates whether the matrices in a are transposed
>> b - tensor b
>> transposedB - indicates whether teh matrices in b are transposed
>> alpha - a coefficient
>> beta - another coefficient
>> parallelRunner - parallel processing module
*/
void _MatrixMul(const XTensor * a, MATRIX_TRANS_TYPE transposedA,
                const XTensor * b, MATRIX_TRANS_TYPE transposedB,
                XTensor * c, DTYPE alpha, DTYPE beta, XPRunner * parallelRunner)
{
    CheckNTErrors(a && b && c, "Empty input tensors!");
    CheckNTErrors(a->dataType == b->dataType && a->dataType == c->dataType,
                  "Input tensors should have the same data type!");
    CheckNTErrors(a->order >= 2 && b->order >= 2 && c->order >= 2,
                  "Input tensors must have a order >= 2!");
    CheckNTErrors(c->order == a->order + b->order - 2, "wrong tensor order")
    
    /* we transform a higher order tensor to a matrix to kill the number
       of calls of matrix multiplication */
    if(transposedA == X_NOTRANS && a->order > 2 && b->order == 2){
        int ncolA = a->dimSize[a->order - 1];
        int ncolC = c->dimSize[c->order - 1];
        XTensor * a2 = NewTensor2D(a->unitNum/ncolA, -ncolA, a->dataType, a->devID, a->mem);
        XTensor * c2 = NewTensor2D(c->unitNum/ncolC, -ncolC, c->dataType, c->devID, c->mem);
        a2->data = a->data;
        c2->data = c->data;
        _MatrixMul2D(a2, transposedA, b, transposedB, c2, alpha, beta, parallelRunner);
        a2->data = NULL;
        c2->data = NULL;
        delete a2;
        delete c2;
        return;
    }

    int an = transposedA == X_TRANS ? a->dimSizeRDI[0] : a->dimSizeRDI[1];
    int am = transposedA == X_TRANS ? a->dimSizeRDI[1] : a->dimSizeRDI[0];
    int bn = transposedB == X_TRANS ? b->dimSizeRDI[0] : b->dimSizeRDI[1];
    int bm = transposedB == X_TRANS ? b->dimSizeRDI[1] : b->dimSizeRDI[0];
    int cn = c->dimSizeRDI[1];
    int cm = c->dimSizeRDI[0];

    CheckNTErrors((am == bn && an == cn && bm == cm), "Unmatched tensors in multiplication!");

    int aBlockSize = a->dimSizeRDI[0] * a->dimSizeRDI[1];
    int bBlockSize = b->dimSizeRDI[0] * b->dimSizeRDI[1];
    int cBlockSize = c->dimSizeRDI[0] * c->dimSizeRDI[1];
    int aRealBlockSize = aBlockSize * a->unitSize;
    int bRealBlockSize = bBlockSize * b->unitSize;
    int cRealBlockSize = cBlockSize * c->unitSize;
    int aBlockNum = 1;
    int bBlockNum = 1;
    int cBlockNum = 1;

    for (int i = 2; i < a->order; i++) {
        CheckNTErrors(a->dimSizeRDI[i] == c->dimSizeRDI[i - 2 + b->order], "Incorrect tensor sizes!");
        aBlockNum *= a->dimSizeRDI[i];
        cBlockNum *= a->dimSizeRDI[i];
    }

    for (int i = 2; i < b->order; i++) {
        CheckNTErrors(b->dimSizeRDI[i] == c->dimSizeRDI[i], "Incorrect tensor sizes!");
        bBlockNum *= b->dimSizeRDI[i];
        cBlockNum *= b->dimSizeRDI[i];
    }

    XList * aList = new XList(10);
    XList * bList = new XList(10);
    XList * cList = new XList(10);
    int aDimSize[2] = { -a->dimSizeRDI[1], a->dimSizeRDI[0] };
    int bDimSize[2] = { -b->dimSizeRDI[1], b->dimSizeRDI[0] };
    int cDimSize[2] = { -c->dimSizeRDI[1], c->dimSizeRDI[0] };

    bool isSparseMul = false;

    for (int p = 0; p < aBlockNum; p++) {
        void * ap = (char*)a->data + aRealBlockSize * p;
        for (int q = 0; q < bBlockNum; q++) {
            void * bp = (char*)b->data + bRealBlockSize * q;
            void * cp = (char*)c->data + cRealBlockSize * (p * bBlockNum + q);

            CheckNTErrors((bRealBlockSize * q < b->unitNum * b->unitSize), "Something wrong!");
            CheckNTErrors((cRealBlockSize * (p * bBlockNum + q) < c->unitNum * c->unitSize), "Something wrong!");

            XTensor * ai = NewTensor(2, aDimSize, a->dataType, a->denseRatio, a->devID, a->mem);
            XTensor * bi = NewTensor(2, bDimSize, b->dataType, b->denseRatio, b->devID, b->mem);
            XTensor * ci = NewTensor(2, cDimSize, c->dataType, c->denseRatio, c->devID, c->mem);
            ai->data = ap;
            bi->data = bp;
            ci->data = cp;
            aList->Add(ai);
            bList->Add(bi);
            cList->Add(ci);
            if (a->isSparse || b->isSparse) {
                CheckNTErrors((a->order == 2 && b->order == 2), "TODO!");
                ai->unitNumNonZero = a->unitNumNonZero;
                bi->unitNumNonZero = b->unitNumNonZero;
                isSparseMul = true;
            }
        }
    }

    if (isSparseMul) {
        for (int i = 0; i < aList->count; i++) {
            XTensor * ai = (XTensor*)aList->GetItem(i);
            XTensor * bi = (XTensor*)bList->GetItem(i);
            XTensor * ci = (XTensor*)cList->GetItem(i);
            _MatrixMul2D(ai, transposedA, bi, transposedB, ci, alpha, beta, parallelRunner);
        }
    }
    else if (a->devID >= 0 && b->devID >= 0 && c->devID >= 0) {
#ifdef USE_CUDA
        CheckNTErrors((a->devID == b->devID && a->devID == c->devID),
                      "The code must be run on the same GPU!");
        
        int devIDBackup;
        ProtectCudaDev(a->devID, devIDBackup);

        cublasHandle_t * handle = a->mem != NULL ? a->mem->GetCublasHandle() : GDevs.GetCudaHandle(a->devID);
        _CudaBLASMatrixMULList(handle,
                               aList, transposedA,
                               bList, transposedB,
                               cList, aList->count,
                               alpha, beta);

        BacktoCudaDev(a->devID, devIDBackup);
#else
        ShowNTErrors("Plesae specify USE_CUDA and recompile the code!");
#endif
    }
    else {
        CheckNTErrors((a->dataType == DEFAULT_DTYPE), "TODO!");
        _MatrixMulBatchedCPU(aList, transposedA,
                             bList, transposedB,
                             cList, alpha, beta);
    }

    for (int i = 0; i < aList->count; i++) {
        XTensor * ai = (XTensor*)aList->GetItem(i);
        ai->data = NULL;
        delete ai;
    }

    for (int i = 0; i < bList->count; i++) {
        XTensor * bi = (XTensor*)bList->GetItem(i);
        bi->data = NULL;
        delete bi;
    }

    for (int i = 0; i < cList->count; i++) {
        XTensor * ci = (XTensor*)cList->GetItem(i);
        ci->data = NULL;
        delete ci;
    }

    delete aList;
    delete bList;
    delete cList;
}

/*
matrix multiplication (return an XTensor structure) c = trans(a) * trans(b) * alpha
make a new tensor to keep the result and return it

For the input tensors a and b, we perform matrix multiplication on the first two dimentsions. 
E.g., let A be a tensor of size y * z * m and B be a tensor of size x * y * n. 
For A * B, we go over each order-2 tensor of A (of size x * y) and each order-2 tensor B (of size z * x), 
like this c_{i,j} = trans(ai) * trans(bj) * alpha + c_{i,j} * beta
where trans() returns the transposed matrix if the flag is fired, ai is the i-th element tensor of A, 
bj is the j-th element tensor of B, and c_{i,j} is the (i,j) element tensor of the result C. 
The result C should be a tensor of z * x * n * m. 
Obviously C = A * B performs normal matrix multiplication if A = y * z and B = x * y.

>> a - tensor a
>> transposedA - indicates whether the matrices in a are transposed
>> b - tensor b
>> transposedB - indicates whether teh matrices in b are transposed
>> alpha - a coefficient
>> parallelRunner - parallel processing module
<< return - the result of matrix multiplication
*/
XTensor MatrixMul(const XTensor &a, MATRIX_TRANS_TYPE transposedA, 
                  const XTensor &b, MATRIX_TRANS_TYPE transposedB, 
                  DTYPE alpha, XPRunner * parallelRunner)
{
    CheckNTErrors(a.dataType == b.dataType, "Input tensors should have the same data type!");
    CheckNTErrors(a.order >= 2 && b.order >= 2, "Input tensors must have a order >= 2!");

    int an = transposedA == X_TRANS ? a.dimSizeRDI[0] : a.dimSizeRDI[1];
    int am = transposedA == X_TRANS ? a.dimSizeRDI[1] : a.dimSizeRDI[0];
    int bn = transposedB == X_TRANS ? b.dimSizeRDI[0] : b.dimSizeRDI[1];
    int bm = transposedB == X_TRANS ? b.dimSizeRDI[1] : b.dimSizeRDI[0];

    CheckNTErrors(am == bn, "Unmatched tensors in multiplication!");

    int order = a.order + b.order - 2;
    int sub = 0;
    int * dimSize = new int[order];
    for (int i = 2; i < a.order; i++)
        dimSize[sub++] = a.dimSizeRDI[a.order + 1 - i];
    for (int i = 2; i < b.order; i++)
        dimSize[sub++] = b.dimSizeRDI[b.order + 1 - i];    
    dimSize[sub++] = an;
    dimSize[sub++] = bm;

    float dr = (!a.isSparse || !b.isSparse) ? 1.0F : MAX(a.denseRatio, b.denseRatio);
    XTensor c(order, dimSize, a.dataType, dr, a.devID, a.mem);
    c.SetTMPFlag();

    /* call _MatrixMul function */
    _MatrixMul(&a, transposedA, &b, transposedB, &c, alpha, 0, parallelRunner);

    /* tensor connections */
    XLink::MakeLink(&a, &b, &c, MATH_MATRIXMUL);
    XLink::AddParamToHeadTrans(&c, transposedA);
    XLink::AddParamToHeadTrans(&c, transposedB);
    XLink::AddParamToHead(&c, alpha);

    /* destroy variables */
    delete[] dimSize;

    return c;
}

/* 
matrix multiplication with no transposition c = a * b * alpha
>> a - tensor a
>> b - tensor b
>> alpha - a coefficient
>> parallelRunner - parallel processing module
<< return - the result of matrix multiplication
*/
XTensor MatrixMul(const XTensor &a, const XTensor &b, 
                  DTYPE alpha, XPRunner * parallelRunner)
{
    CheckNTErrors(a.dataType == b.dataType, "Input tensors should have the same data type!");
    CheckNTErrors(a.order >= 2 && b.order >= 2, "Input tensors must have a order >= 2!");

    int an = a.dimSizeRDI[1];
    int am = a.dimSizeRDI[0];
    int bn = b.dimSizeRDI[1];
    int bm = b.dimSizeRDI[0];

    CheckNTErrors(am == bn, "Unmatched tensors in multiplication!");

    int order = a.order + b.order - 2;
    int sub = 0;
    int * dimSize = new int[order];
    for (int i = 2; i < a.order; i++)
        dimSize[sub++] = a.dimSizeRDI[a.order + 1 - i];
    for (int i = 2; i < b.order; i++)
        dimSize[sub++] = b.dimSizeRDI[b.order + 1 - i];    
    dimSize[sub++] = an;
    dimSize[sub++] = bm;

    float dr = (!a.isSparse || !b.isSparse) ? 1.0F : MAX(a.denseRatio, b.denseRatio);
    XTensor c(order, dimSize, a.dataType, dr, a.devID, a.mem);
    c.SetTMPFlag();

    /* call _MatrixMul function */
    _MatrixMul(&a, X_NOTRANS, &b, X_NOTRANS, &c, alpha, 0, parallelRunner);

    /* tensor connections */
    XLink::MakeLink(&a, &b, &c, MATH_MATRIXMUL);
    XLink::AddParamToHeadTrans(&c, X_NOTRANS);
    XLink::AddParamToHeadTrans(&c, X_NOTRANS);
    XLink::AddParamToHead(&c, alpha);

    /* destroy variables */
    delete[] dimSize;

    return c;
}

} // namespace nts(NiuTrans.Tensor)



