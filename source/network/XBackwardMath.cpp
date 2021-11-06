/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2018, Natural Language Processing Lab, Northeastern University.
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
 * backward computation for math operations
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-18
 */

#include "XNoder.h"
#include "XBackwardMath.h"
#include "../tensor/XName.h"
#include "../tensor/core/CHeader.h"
#include "../tensor/function/FHeader.h"

namespace nts{

/* compute dE/dx of a node */
void XMathGrad::MakeGrad(XTensor * node, bool isEfficient)
{
    if (!isEfficient) {
        CheckNTErrors(node->grad != NULL, "No gradient found!");
    }
    else {
        CheckNTErrors(!node->isGrad || node->grad != NULL, "No gradient found!");
    }

    XLink &income = node->income;
    int operID = income.typeID;

    if (operID == MATH_ABSOLUTE)
        GradAbsolute(node, isEfficient);
    else if (operID == MATH_COS)
        GradCos(node, isEfficient);
    else if (operID == MATH_EXP)
        GradExp(node, isEfficient);
    else if (operID == MATH_LOG)
        GradLog(node, isEfficient);
    else if (operID == MATH_ROUND)
        GradRound(node, isEfficient);
    else if (operID == MATH_SIGN)
        GradSign(node, isEfficient);
    else if (operID == MATH_SIN)
        GradSin(node, isEfficient);
    else if (operID == MATH_TAN)
        GradTan(node, isEfficient);

    else if (operID == MATH_CLIP)
        GradClip(node, isEfficient);
    else if (operID == MATH_DIV)
        GradDiv(node, isEfficient);
    else if (operID == MATH_DIVDIM)
        GradDivDim(node, isEfficient);
    else if (operID == MATH_MATRIXMUL)
        GradMatrixMul(node, isEfficient);
    else if (operID == MATH_MATRIXMULBATCHED)
        GradMatrixMulBatched(node, isEfficient);
    else if (operID == MATH_MULTIPLY)
        GradMultiply(node, isEfficient);
    else if (operID == MATH_MULTIPLYDIM)
        GradMultiplyDim(node, isEfficient);
    else if (operID == MATH_MULTIPLY_INPLACE)
        GradMultiply(node, isEfficient);
    else if (operID == MATH_MULTIPLYDIM_INPLACE)
        GradMultiplyDim(node, isEfficient);
    else if (operID == MATH_MULTIPLYBROADCAST)
        GradMultiplyBroadcast(node, isEfficient);
    else if (operID == MATH_NEGATE)
        GradNegate(node, isEfficient);
    else if (operID == MATH_NORMALIZE)
        GradNormalize(node, isEfficient);
    else if (operID == MATH_POWER)
        GradPower(node, isEfficient);
    else if (operID == MATH_RECIPROCAL)
        GradReciprocal(node, isEfficient);
    else if (operID == MATH_SQRT)
        GradSqrt(node, isEfficient);
    else if (operID == MATH_SQUARE)
        GradSquare(node, isEfficient);
    else if (operID == MATH_SCALEANDSHIFT)
        GradScaleAndShift(node, isEfficient);
    else if (operID == MATH_SCALE)
        GradScale(node, isEfficient);
    else if (operID == MATH_DESCALE)
        GradDescale(node, isEfficient);
    else if (operID == MATH_SHIFT)
        GradShift(node, isEfficient);
    else if (operID == MATH_SUB)
        GradSub(node, isEfficient);
    else if (operID == MATH_SUBDIM)
        GradSubDim(node, isEfficient);
    else if (operID == MATH_SUM)
        GradSum(node, isEfficient);
    else if (operID == MATH_SUMDIM)
        GradSumDim(node, isEfficient);
    else if (operID == MATH_SUMBROADCAST)
        GradSumBroadcast(node, isEfficient);
    else if (operID == REDUCE_REDUCEMEAN)
        GradReduceMean(node, isEfficient);
    else if (operID == REDUCE_REDUCESUM)
        GradReduceSum(node, isEfficient);
    else if (operID == REDUCE_REDUCESUMALL)
        GradReduceSumAll(node, isEfficient);
    else if (operID == REDUCE_REDUCESUMSQUARED)
        GradReduceSumSquared(node, isEfficient);
    else if (operID == REDUCE_REDUCEVARIANCE)
        GradReduceVariance(node, isEfficient);
    else if (operID == MATH_MULANDSHIFT)
        GradMulAndShift(node, isEfficient);
    else if (operID == MATH_MLP)
        GradMLP(node, isEfficient);
    else{
        ShowNTErrors("Unsupported backward computation! TODO!");
    }

    node->visitMark = NODE_FINISHED;
    node->isGradFinished = true;
}

/* indicates whether the node is for a math operation */
bool XMathGrad::IsMathOP(XTensor * node)
{
    XLink &income = node->income;
    return (income.typeID & MATH_BASE) != 0;
}

/*
gradient for absolute
for
c = |a|
we have
dE/da = dE/dc   a >= 0
        -dE/dc  a < 0
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradAbsolute(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for ABSOLUTE!");

    XTensor * a = income.tails[0];

    /* dE/da = dE/dc * sign(a) */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);

        if (a->mem != NULL)
            a->mem->LockBuf();
        XTensor * tmp = NewTensorBufV2(a, a->devID, a->mem);
        _Sign(a, tmp);
        _Multiply(node->grad, tmp, a->grad, 1.0F);

        DelTensorBuf(tmp);
        if (a->mem != NULL)
            a->mem->UnlockBuf();
    }
}

/*
gradient for cos
for
c = cos(a)
we have
dE/da = dE/dc * -sin(a)
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradCos(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for COS!");

    XTensor * a = income.tails[0];

    /* dE/da = dE/dc * -sin(a) */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);

        if (a->mem != NULL)
            a->mem->LockBuf();
        XTensor * tmp = NewTensorBufV2(a, a->devID, a->mem);
        _Sin(a, tmp);
        _NegateMe(tmp);
        _Multiply(node->grad, tmp, a->grad, 1.0F);

        DelTensorBuf(tmp);
        if (a->mem != NULL)
            a->mem->UnlockBuf();
    }
}

/*
gradient for exp
for
c = exp(a)
we have
dE/da = dE/dc * exp(a)
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradExp(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for EXP!");

    XTensor * a = income.tails[0];

    /* dE/da = dE/dc * exp(a) */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);

        if (a->mem != NULL)
            a->mem->LockBuf();
        XTensor * tmp = NewTensorBufV2(a, a->devID, a->mem);
        _Exp(a, tmp);
        _Multiply(node->grad, tmp, a->grad, 1.0F);

        DelTensorBuf(tmp);
        if (a->mem != NULL)
            a->mem->UnlockBuf();
    }
}

/*
gradient for log
for
c = log(a)
we have
dE/da = dE/dc * 1/a
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradLog(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for LOG!");

    XTensor * a = income.tails[0];

    /* dE/da = dE/dc * 1/a */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);
        _Div(node->grad, a, a->grad, 1.0F);
    }
}

/*
gradient for round
for
c = round(a)
we have
dE/da = 0
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradRound(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for ROUND!");

    XTensor * a = income.tails[0];

    /* dE/da = 0, we do nothing here */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);
    }
}

/*
gradient for sign
for
c = sign(a)
we have
dE/da = 0
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradSign(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for SIGN!");

    XTensor * a = income.tails[0];

    /* dE/da = 0, we do nothing here */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);
    }
}

/*
gradient for sin
for
c = sin(a)
we have
dE/da = dE/dc * cos(a)
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradSin(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for SIN!");

    XTensor * a = income.tails[0];

    /* dE/da = dE/dc * cos(a) */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);

        if (a->mem != NULL)
            a->mem->LockBuf();
        XTensor * tmp = NewTensorBufV2(a, a->devID, a->mem);
        _Cos(a, tmp);
        _Multiply(node->grad, tmp, a->grad, 1.0F);

        DelTensorBuf(tmp);
        if (a->mem != NULL)
            a->mem->UnlockBuf();
    }
}

/*
gradient for tan
for
c = tan(a)
we have
dE/da = dE/dc * 1/(cos(a))^2
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradTan(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for TAN!");

    XTensor * a = income.tails[0];

    /* dE/da = dE/dc * 1/(cos(a))^2
             = dE/dc * (cos(a))^-2 */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);

        if (a->mem != NULL)
            a->mem->LockBuf();
        XTensor * tmp = NewTensorBufV2(a, a->devID, a->mem);
        _Cos(a, tmp);
        _PowerMe(tmp, -2.0F);
        _Multiply(node->grad, tmp, a->grad, 1.0F);

        DelTensorBuf(tmp);
        if (a->mem != NULL)
            a->mem->UnlockBuf();
    }
}

/*
gradient for clip
we have
dE/da = 1  lower < a < upper
dE/da = 0  otherwise 
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradClip(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for CLIP!");

    XTensor * a = income.tails[0];

    DTYPE lower = income.GetParam(0);
    DTYPE upper = income.GetParam(1);

    /* dE/da = 1  lower < a < upper
             = 0  otherwise */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);

        if (a->mem != NULL)
            a->mem->LockBuf();
        XTensor * tmp = NewTensorBufV2(a, a->devID, a->mem);
        _ClipBackward(node, a, node->grad, tmp, lower, upper);
        _SumMe(a->grad, tmp);

        DelTensorBuf(tmp);
        if (a->mem != NULL)
            a->mem->UnlockBuf();
    }
}

/*
gradient for divide
for
c =  a / b
we have
dE/da = dE/dc / b
dE/db = dE/dc * a / -b^2
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradDiv(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 2, "Wrong input tensor number for DIVIDE!");

    XTensor * a = income.tails[0];
    XTensor * b = income.tails[1];
    CheckNTErrors(_IsSameShaped(a, b), "Wrong sized input tensors!");
    
    /* dE/da = dE/dc / b */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);
        _Div(node->grad, b, a->grad, 1.0F);
    }

    /* dE/db = dE/dc * a/(-b^2)
             = dE/dc * a * (-b^-2) */
    if (!isEfficient || b->isGrad) {
        XNoder::MakeGrad(b);
        if (a->mem != NULL)
            a->mem->LockBuf();
        XTensor * tmp = NewTensorBufV2(a, a->devID, a->mem);
        _Power(b, tmp, -2.0F);
        _NegateMe(tmp);
        _MultiplyMe(tmp, a);
        _Multiply(node->grad, tmp, b->grad, 1.0F);

        DelTensorBuf(tmp);
        if (a->mem != NULL)
            a->mem->UnlockBuf();
    }
}

/* 
gradient for division with one dimension
c = a / b
where the size of b is equal to dimension n of a, i.e., |b| = a.dimSize[n]
dE/da = dE/dc * (1/b)
dE/db = (dE/dc * (-a/b^2)).reduce(0,...,n-1,n+1,...)

>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradDivDim(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 2, "Wrong input tensor number for DIVDIM!");

    XTensor * a = income.tails[0];
    XTensor * b = income.tails[1];
    int n = income.GetParamInt(0);

    /* dE/da = dE/dc * (1/b) */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);
        _DivDim(node->grad, b, a->grad, n, 1.0);
    }

    /* dE/db = dE/dc * dc/db
             = (dE/dc * (-a/b^2)).reduce(0,...,n-1,n+1,...) */
    if (!isEfficient || b->isGrad) {
        XNoder::MakeGrad(b);
        int order = a->order;
        int dimSize[MAX_TENSOR_DIM_NUM];
        memcpy(dimSize, a->dimSize, sizeof(int) * a->order);

        if (a->mem != NULL)
            a->mem->LockBuf();
        XTensor * aTMP1 = NewTensorBufV2(a, a->devID, a->mem);
        XTensor * aTMP2 = NewTensorBufV2(a, a->devID, a->mem);
        if ((b->mem != NULL) && (b->mem != a->mem)) {
            b->mem->LockBuf();
        }
        XTensor * bTMP = NewTensorBufV2(b, b->devID, b->mem);
        if ((node->mem != NULL) && (node->mem != a->mem) && (node->mem != b->mem)) {
            node->mem->LockBuf();
        }
        XTensor * interGradTMP = NewTensorBufV2(node->grad, node->devID, node->mem);

        _Negate(a, aTMP1);
        _Power(b, bTMP, -2.0F);
        _MultiplyDim(aTMP1, bTMP, aTMP2, n);

        _Multiply(node->grad, aTMP2, interGradTMP);

        if (n == order - 1) {
            int reshapedSize[MAX_TENSOR_DIM_NUM];
            reshapedSize[0] = a->unitNum / dimSize[order - 1];
            reshapedSize[1] = dimSize[order - 1];

            /* we reshape dE/dc * a to a matrix whose column number is equal to the
               size of b. Then we can reduce the matrix into a row vector. */
            interGradTMP->Reshape(2, reshapedSize);

            XTensor * bGradTMP = NewTensorBufV2(b->grad, b->devID, b->mem);
            _ReduceSum(interGradTMP, bGradTMP, 0);

            _SumMe(b->grad, bGradTMP);

            DelTensorBuf(bGradTMP);
        }
        else {
            int reshapedSize[MAX_TENSOR_DIM_NUM];
            reshapedSize[0] = 1;
            reshapedSize[1] = dimSize[n];
            reshapedSize[2] = 1;

            for (int i = 0; i < order; i++) {
                if (i < n)
                    reshapedSize[0] *= dimSize[i];
            }

            reshapedSize[2] = a->unitNum / (reshapedSize[0] * reshapedSize[1]);

            /* we reshape dE/dc to a 3D tensor of size (x, y, z) where y = |b|.
               Then reduce along with z and x to obtain dE/db. */
            interGradTMP->Reshape(3, reshapedSize);

            // b->mem->LockBuf();
            XTensor * interGrad = NewTensorBufV2(2, reshapedSize, b->dataType, b->denseRatio, b->devID, b->mem);
            _ReduceSum(interGradTMP, interGrad, 2);

            XTensor * bGradTMP2 = NewTensorBufV2(b->grad, b->devID, b->mem);
            _ReduceSum(interGrad, bGradTMP2, 0);

            _SumMe(b->grad, bGradTMP2);

            DelTensorBuf(bGradTMP2);
            DelTensorBuf(interGrad);
            // b->mem->UnlockBuf();
        }

        DelTensorBuf(interGradTMP);
        if ((node->mem != NULL) && (node->mem != a->mem) && (node->mem != b->mem)) {
            node->mem->UnlockBuf();
        }
        DelTensorBuf(bTMP);
        if ((b->mem != NULL) && (b->mem != a->mem)) {
            b->mem->UnlockBuf();
        }
        DelTensorBuf(aTMP2);
        DelTensorBuf(aTMP1);
        if (a->mem != NULL)
            a->mem->UnlockBuf();
    }
}

/* 
gradient for matrix multiply
for c = matmul(a, b) * \alpha
we have 
dE/da = dE/dc * b^T * \alpha
dE/db = a^T * dE/dc * \alpha
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradMatrixMul(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 2, "Wrong input tensor number for MULTIPLY!");
    CheckNTErrors(income.paramNum == 3, "Wrong parameter number for MULTIPLY!");

    XTensor * a = income.tails[0]; 
    XTensor * b = income.tails[1];
    MATRIX_TRANS_TYPE transA = income.GetParamTrans(0);
    MATRIX_TRANS_TYPE transB = income.GetParamTrans(1);
    DTYPE alpha = income.GetParam(2);

    if (!isEfficient || a->isGrad)
        XNoder::MakeGrad(a);
    if (!isEfficient || b->isGrad)
        XNoder::MakeGrad(b);

    XTensor * c = node;
    XTensor * dedc = node->grad;
    XTensor * deda = a->grad;
    XTensor * dedb = b->grad;

    if (a->order == 2 && b->order == 2)
        GradMatrixMul(a, deda, transA, b, dedb, transB, dedc, alpha, isEfficient);
    else if (transA == X_NOTRANS && a->order > 2 && b->order == 2){
        int orderBackupA = a->order;
        int orderBackupC = c->order;
        int dimsBackupA[MAX_TENSOR_DIM_NUM];
        int dimsBackupC[MAX_TENSOR_DIM_NUM];
        memcpy(dimsBackupA, a->dimSize, sizeof(int) * a->order);
        memcpy(dimsBackupC, c->dimSize, sizeof(int) * c->order);

        a->Reshape(a->unitNum/a->GetDim(-1), a->GetDim(-1));
        c->Reshape(c->unitNum/c->GetDim(-1), c->GetDim(-1));
        if (!isEfficient || a->isGrad)
            deda->Reshape(deda->unitNum/deda->GetDim(-1), deda->GetDim(-1));
        dedc->Reshape(dedc->unitNum/dedc->GetDim(-1), dedc->GetDim(-1));

        GradMatrixMul(a, deda, transA, b, dedb, transB, dedc, alpha, isEfficient);

        a->Reshape(orderBackupA, dimsBackupA);
        c->Reshape(orderBackupC, dimsBackupC);
        if (!isEfficient || a->isGrad)
            deda->Reshape(orderBackupA, dimsBackupA);
        dedc->Reshape(orderBackupC, dimsBackupC);
    }
    else{
        ShowNTErrors("TODO!");
    }
}
    
/*
gradient for matrix multiply: c = matmul(a, b) * \alpha
>> a - as it is
>> deda - dE/da
>> b - as it is
>> dedb - dE/db
>> dedc - dE/dc
>> alpha - the scalar
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradMatrixMul(XTensor * a, XTensor * deda, MATRIX_TRANS_TYPE transA,
                              XTensor * b, XTensor * dedb, MATRIX_TRANS_TYPE transB,
                              XTensor * dedc, DTYPE alpha, bool isEfficient)
{
    /* c = a * b * \alpha */
    if (transA == X_NOTRANS && transB == X_NOTRANS) {
        
        /* dE/da = dE/dc * b^T * \alpha */
        if (!isEfficient || a->isGrad)
            _MatrixMul(dedc, X_NOTRANS, b, X_TRANS, deda, alpha, 1.0F);
        
        /* dE/db = a^T * dE/dc * \alpha */
        if (!isEfficient || b->isGrad)
            _MatrixMul(a, X_TRANS, dedc, X_NOTRANS, dedb, alpha, 1.0F);
    }
    
    /* c = a^T * b * \alpha */
    else if (transA == X_TRANS && transB == X_NOTRANS){
        
        /* dE/da = (dE/dc * b^T)^T * \alpha 
                 = b * dE/dc^T * \alpha */
        if (!isEfficient || a->isGrad)
            _MatrixMul(b, X_NOTRANS, dedc, X_TRANS, deda, alpha, 1.0F);
        
        /* dE/db = a * dE/dc * \alpha */
        if (!isEfficient || b->isGrad)
            _MatrixMul(a, X_NOTRANS, dedc, X_NOTRANS, dedb, alpha, 1.0F);
    }
    
    /* c = a * b^T * \alpha */
    else if (transA == X_NOTRANS && transB == X_TRANS){
        
        /* dE/da = dE/dc * b * \alpha */
        if (!isEfficient || a->isGrad)
            _MatrixMul(dedc, X_NOTRANS, b, X_NOTRANS, deda, alpha, 1.0F);
        
        /* dE/db = (a^T * dE/dc)^T * \alpha 
                 = dE/dc^T * a * \alpha */
        if (!isEfficient || b->isGrad)
            _MatrixMul(dedc, X_TRANS, a, X_NOTRANS, dedb, alpha, 1.0F);
    }
    
    /* c = a^T * b^T * \alpha */
    else if (transA == X_TRANS && transB == X_TRANS){
        
        /* dE/da = (dE/dc * b)^T * \alpha 
                 = b^T * dE/dc^T * \alpha */
        if (!isEfficient || a->isGrad)
            _MatrixMul(b, X_TRANS, dedc, X_TRANS, deda, alpha, 1.0F);
        
        /* dE/db = (a * dE/dc)^T * \alpha 
                 = dE/dc^T * a^T * \alpha */
        if (!isEfficient || b->isGrad)
            _MatrixMul(dedc, X_TRANS, a, X_TRANS, dedb, alpha, 1.0F);
    }
}

/* 
gradient for matrix multiply in batch mode.
for each batch: c_i = matmul(a_i, b_i) * \alpha
for c_i = matmul(a_i, b_i) * \alpha
we have 
dE/da_i = dE/dc_i * b_i^T * \alpha
dE/db_i = a_i^T * dE/dc_i * \alpha
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradMatrixMulBatched(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 2, "Wrong input tensor number for MULTIPLY!");
    CheckNTErrors(income.paramNum == 3, "Wrong parameter number for MULTIPLY!");

    XTensor * a = income.tails[0]; 
    XTensor * b = income.tails[1];
    MATRIX_TRANS_TYPE transA = income.GetParamTrans(0);
    MATRIX_TRANS_TYPE transB = income.GetParamTrans(1);
    DTYPE alpha = income.GetParam(2);

    if (!isEfficient || a->isGrad)
        XNoder::MakeGrad(a);
    if (!isEfficient || b->isGrad)
        XNoder::MakeGrad(b);

    XTensor * dedc = node->grad;
    XTensor * deda = a->grad;
    XTensor * dedb = b->grad;

    /* c = a * b * \alpha */
    if (transA == X_NOTRANS && transB == X_NOTRANS) {
        
        /* dE/da = dE/dc * b^T * \alpha */
        if (!isEfficient || a->isGrad)
            _MatrixMulBatched(dedc, X_NOTRANS, b, X_TRANS, deda, alpha, 1.0F);
        
        /* dE/db = a^T * dE/dc * \alpha */
        if (!isEfficient || b->isGrad)
            _MatrixMulBatched(a, X_TRANS, dedc, X_NOTRANS, dedb, alpha, 1.0F);
    }
    
    /* c = a^T * b * \alpha */
    else if (transA == X_TRANS && transB == X_NOTRANS) {
        
        /* dE/da = (dE/dc * b^T)^T * \alpha 
                 = b * dE/dc^T * \alpha */
        if (!isEfficient || a->isGrad)
            _MatrixMulBatched(b, X_NOTRANS, dedc, X_TRANS, deda, alpha, 1.0F);
        
        /* dE/db = a * dE/dc * \alpha */
        if (!isEfficient || b->isGrad)
            _MatrixMulBatched(a, X_NOTRANS, dedc, X_NOTRANS, dedb, alpha, 1.0F);
    }
    
    /* c = a * b^T * \alpha */
    else if (transA == X_NOTRANS && transB == X_TRANS) {
        
        /* dE/da = dE/dc * b * \alpha */
        if (!isEfficient || a->isGrad)
            _MatrixMulBatched(dedc, X_NOTRANS, b, X_NOTRANS, deda, alpha, 1.0F);
        
        /* dE/db = (a^T * dE/dc)^T * \alpha 
                 = dE/dc^T * a * \alpha */
        if (!isEfficient || b->isGrad)
            _MatrixMulBatched(dedc, X_TRANS, a, X_NOTRANS, dedb, alpha, 1.0F);
    }
    
    /* c = a^T * b^T * \alpha */
    else if (transA == X_TRANS && transB == X_TRANS) {
        
        /* dE/da = (dE/dc * b)^T * \alpha 
                 = b^T * dE/dc^T * \alpha */
        if (!isEfficient || a->isGrad)
            _MatrixMulBatched(b, X_TRANS, dedc, X_TRANS, deda, alpha, 1.0F);
        
        /* dE/db = (a * dE/dc)^T * \alpha 
                 = dE/dc^T * a^T * \alpha */
        if (!isEfficient || b->isGrad)
            _MatrixMulBatched(dedc, X_TRANS, a, X_TRANS, dedb, alpha, 1.0F);
    }
}

/* 
gradient for multiply (dot production)
for
c =  a * b 
we have
dE/da = dE/dc * b
dE/db = dE/dc * a 
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradMultiply(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 2, "Wrong input tensor number for MULTIPLY!");

    XTensor * a = income.tails[0]; 
    XTensor * b = income.tails[1];

    CheckNTErrors(_IsSameShaped(a, b), "Wrong sized input tensors!");

    /* dE/da = dE/dc * b */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);
        _Multiply(node->grad, b, a->grad, 1.0F);
    }

    /* dE/db = dE/dc * a */
    if (!isEfficient || b->isGrad) {
        XNoder::MakeGrad(b);
        _Multiply(node->grad, a, b->grad, 1.0F);
    }
}

/*
gradient for multiply with one dimension
c = a * b
where the size of b is equal to dimension n of a, i.e., |b| = a.dimSize[n]
dE/da = dE/dc * b
dE/db = (dE/dc * a).reduce(0,...,n-1,n+1,...)

>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradMultiplyDim(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 2, "Wrong input tensor number for MULTIPLYDIM!");

    XTensor * a = income.tails[0];
    XTensor * b = income.tails[1];
    int n = income.GetParamInt(0);

    /* dE/da = dE/dc * b */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);
        _MultiplyDim(node->grad, b, a->grad, n, 1.0F);
    }

    /* dE/db = (dE/dc * a).reduce(0,...,n-1,n+1,...) */
    if (!isEfficient || b->isGrad) {
        XNoder::MakeGrad(b);
        int order = a->order;
        int dimSize[MAX_TENSOR_DIM_NUM];
        memcpy(dimSize, a->dimSize, sizeof(int) * a->order);

        if (node->mem != NULL)
            node->mem->LockBuf();
        XTensor * bGradTMP = NewTensorBufV2(node->grad, node->devID, node->mem);
        _Multiply(node->grad, a, bGradTMP);

        if (n == order - 1) {
            int reshapedSize[MAX_TENSOR_DIM_NUM];
            reshapedSize[0] = a->unitNum / dimSize[order - 1];
            reshapedSize[1] = dimSize[order - 1];

            /* we reshape dE/dc * a to a matrix whose column number is equal to the
               size of b. Then we can reduce the matrix into a row vector. */
            bGradTMP->Reshape(2, reshapedSize);

            if ((b->mem != NULL) && (b->mem != node->mem)) {
                b->mem->LockBuf();
            }
            XTensor * bGradTMP2 = NewTensorBufV2(b->grad, b->devID, b->mem);
            _ReduceSum(bGradTMP, bGradTMP2, 0);

            _Sum(b->grad, bGradTMP2, b->grad);

            DelTensorBuf(bGradTMP2);
            if ((b->mem != NULL) && (b->mem != node->mem)) {
                b->mem->UnlockBuf();
            }
        }
        else {
            int reshapedSize[MAX_TENSOR_DIM_NUM];
            reshapedSize[0] = 1;
            reshapedSize[1] = dimSize[n];
            reshapedSize[2] = 1;

            for (int i = 0; i < order; i++) {
                if (i < n)
                    reshapedSize[0] *= dimSize[i];
            }

            reshapedSize[2] = a->unitNum / (reshapedSize[0] * reshapedSize[1]);

            /* we reshape dE/dc to a 3D tensor of size (x, y, z) where y = |b|.
               Then reduce along with z and x to obtain dE/db. */
            bGradTMP->Reshape(3, reshapedSize);

            if ((b->mem != NULL) && (b->mem != node->mem)) {
                b->mem->LockBuf();
            }
            XTensor * interGrad = NewTensorBufV2(2, reshapedSize, b->dataType, b->denseRatio, b->devID, b->mem);
            _ReduceSum(bGradTMP, interGrad, 2);

            XTensor * bGradTMP2 = NewTensorBufV2(b->grad, b->devID, b->mem);
            _ReduceSum(interGrad, bGradTMP2, 0);

            _Sum(b->grad, bGradTMP2, b->grad);

            DelTensorBuf(bGradTMP2);
            DelTensorBuf(interGrad);
            if ((b->mem != NULL) && (b->mem != node->mem)) {
                b->mem->UnlockBuf();
            }
        }
        DelTensorBuf(bGradTMP);
        if (node->mem != NULL)
            node->mem->UnlockBuf();
    }
}

/*
gradient for multiplication by broadcasting: 
c = a * b
where some dimensions of b are of size 1

dE/da = dE/dc * b
dE/db = (dE/dc * a).reduce(0...n)
where a.reduce(0...n) is the reduction along the dimension
whose size is 1 in b. Note that there might be several reductions.

>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradMultiplyBroadcast(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 2, "Wrong input tensor number for MULTIPLYBROADCAST!");

    XTensor * a = income.tails[0];
    XTensor * b = income.tails[1];

    XNoder::MakeGrad(a);

    /* dE/da = dE/dc * b */
    if (!isEfficient || a->isGrad)
        _MultiplyBroadcast(node->grad, b, a->grad, 1.0F);

    /* dE/db = (dE/dc * a).reduce(0...n) */
    if (!isEfficient || b->isGrad) {
        if (b->isVar || b->income.tailNum > 0)
            ShowNTErrors("TODO");
    }
}

/*
gradient for negate
for
c = -a
we have
dE/da = dE/dc * (-1)
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradNegate(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for NEGATE!");

    XTensor * a = income.tails[0];

    /* dE/da = dE/dc * (-1) */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);        
        _Sum(a->grad, node->grad, a->grad, -1.0F);
    }
}

/*
gradient for normalize
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradNormalize(XTensor * node, bool isEfficient)
{
    ShowNTErrors("TODO!");
}

/*
gradient for power
for
c = pow(a,p)
we have
dE/da = (dE/dc) * p * a^(p-1)
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradPower(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for POWER!");

    XTensor * a = income.tails[0];

    DTYPE p = income.GetParam(0);

    /* dE/da = (dE/dc) * p * a^(p-1) */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);

        if (a->mem != NULL)
            a->mem->LockBuf();
        XTensor * tmp = NewTensorBufV2(a, a->devID, a->mem);
        _Power(a, tmp, p - 1.0F);
        _ScaleAndShiftMe(tmp, p);
        _Multiply(node->grad, tmp, a->grad, 1.0F);

        DelTensorBuf(tmp);
        if (a->mem != NULL)
            a->mem->UnlockBuf();
    }
}


/*
gradient for reciprocal
for
c = reciprocal(a)
we have
dE/da = (dE/dc) * -a^(-2)
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in an efficient manner
*/
void XMathGrad::GradReciprocal(XTensor* node, bool isEfficient)
{
    XLink& income = node->income;
    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for RECIPROCAL!");

    XTensor* a = income.tails[0];

    /* dE/da = (dE/dc) * -a^(-2) */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);

        if (a->mem != NULL)
            a->mem->LockBuf();
        XTensor* tmp = NewTensorBufV2(a, a->devID, a->mem);
        _Power(a, tmp, -2.0F);
        _NegateMe(tmp);
        _Multiply(node->grad, tmp, a->grad, 1.0F);

        DelTensorBuf(tmp);
        if (a->mem != NULL)
            a->mem->UnlockBuf();
    }
}

/*
gradient for sqrt
for
c = sqrt(a)
we have
dE/da = (dE/dc) * 2 * a
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in an efficient manner
*/
void XMathGrad::GradSqrt(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for SQRT!");

    XTensor * a = income.tails[0];

    /* dE/da = (dE/dc) * 2 * a */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);

        if (a->mem != NULL)
            a->mem->LockBuf();
        XTensor* tmp = NewTensorBufV2(a, a->devID, a->mem);
        _ScaleMe(tmp, 2.0F);
        _Multiply(node->grad, tmp, a->grad, 1.0F);

        DelTensorBuf(tmp);
        if (a->mem != NULL)
            a->mem->UnlockBuf();
    }
}

/*
gradient for square
for
c = square(a)
we have
dE/da = (dE/dc) * (1/2) * a^(-1/2)
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in an efficient manner
*/
void XMathGrad::GradSquare(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for SQUARE!");

    XTensor * a = income.tails[0];

    /* dE/da = (dE/dc) * (1/2) * a^(-1/2)*/
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);

        if (a->mem != NULL)
            a->mem->LockBuf();
        XTensor* tmp = NewTensorBufV2(a, a->devID, a->mem);
        _Power(a, tmp, -0.5F);
        _ScaleMe(tmp, 0.5);
        _Multiply(node->grad, tmp, a->grad, 1.0F);

        DelTensorBuf(tmp);
        if (a->mem != NULL)
            a->mem->UnlockBuf();
    }
}

/*
gradient for ScaleAndShift
for
c = a * scale + shift
we have
dE/da = dE/dc * scale
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradScaleAndShift(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for SCALEANDSHIFT!");

    XTensor * a = income.tails[0];

    DTYPE scale = income.GetParam(0);

    /* dE/da = dE/dc * scale */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);

        _Sum(a->grad, node->grad, a->grad, scale);
    }
}

/*
gradient for Scale
for
c = a * scale
we have
dE/da = dE/dc * scale
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
an efficient manner
*/
void XMathGrad::GradScale(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for SCALE!");

    XTensor * a = income.tails[0];

    DTYPE scale = income.GetParam(0);

    /* dE/da = dE/dc * scale */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);

        _Sum(a->grad, node->grad, a->grad, scale);
    }
}

/*
gradient for Descale
for
c = a / descale
we have
dE/da = dE/dc / descale
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
an efficient manner
*/
void XMathGrad::GradDescale(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for DESCALE!");

    XTensor * a = income.tails[0];

    DTYPE descale = income.GetParam(0);

    /* dE/da = dE/dc / descale */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);

        _Sum(a->grad, node->grad, a->grad, 1 / descale);
    }
}

/*
gradient for Shift
for
c = a + shift
we have
dE/da = dE/dc
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
an efficient manner
*/
void XMathGrad::GradShift(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for SHIFT!");

    XTensor * a = income.tails[0];

    /* dE/da = dE/dc */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);

        _Sum(a->grad, node->grad, a->grad);
    }
}

/*
gradient for minus
for
c =  a - b * \beta
we have
dE/da = dE/dc
dE/db = -dE/dc * \beta
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradSub(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 2, "Wrong input tensor number for SUBSTRACT!");

    XTensor * a = income.tails[0];
    XTensor * b = income.tails[1];
    DTYPE beta = income.GetParam(0);

    /* dE/da = dE/dc */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);
        _Sum(a->grad, node->grad, a->grad);
    }

    /* dE/db = -dE/dc * \beta */
    if (!isEfficient || b->isGrad) {
        XNoder::MakeGrad(b);
        _Sum(b->grad, node->grad, b->grad, -beta);
    }
}

/*
gradient for subtraction with one dimension
c = a - b * \beta
where the size of b is equal to dimension n of a, i.e., |b| = a.dimSize[n]
dE/da = dE/dc
dE/db = - dE/dc * b.reduce(0,...,n-1,n+1,...) * \beta
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradSubDim(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 2, "Wrong input tensor number for SUBDIM!");

    XTensor * a = income.tails[0];
    XTensor * b = income.tails[1];
    int n = income.GetParamInt(0);
    DTYPE beta = income.GetParam(1);

    /* dE/da = dE/dc */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);
        _Sum(a->grad, node->grad, a->grad);
    }

    /* dE/db = - dE/dc * b.reduce(0,...,n-1,n+1,...) * \beta */
    if (!isEfficient || b->isGrad) {
        XNoder::MakeGrad(b);
        int order = a->order;
        int dimSize[MAX_TENSOR_DIM_NUM];
        memcpy(dimSize, a->dimSize, sizeof(int) * a->order);

        if (n == order - 1) {
            int reshapedSize[MAX_TENSOR_DIM_NUM];
            reshapedSize[0] = a->unitNum / dimSize[order - 1];
            reshapedSize[1] = dimSize[order - 1];

            /* we reshape dE/dc to a matrix whose column number is equal to the
               size of b. Then we can reduce the matrix into a row vector. */
            node->grad->Reshape(2, reshapedSize);

            if (b->mem != NULL)
                b->mem->LockBuf();
            XTensor * bGradTMP = NewTensorBufV2(b->grad, b->devID, b->mem);
            _ReduceSum(node->grad, bGradTMP, 0);
            if (beta != 1.0F)
                _ScaleAndShiftMe(bGradTMP, beta);
            _Sub(b->grad, bGradTMP, b->grad);
            DelTensorBuf(bGradTMP);
            if (b->mem != NULL)
                b->mem->UnlockBuf();

            node->grad->Reshape(order, dimSize);
        }
        else {
            int reshapedSize[MAX_TENSOR_DIM_NUM];
            reshapedSize[0] = 1;
            reshapedSize[1] = dimSize[n];
            reshapedSize[2] = 1;

            for (int i = 0; i < order; i++) {
                if (i < n)
                    reshapedSize[0] *= dimSize[i];
            }

            reshapedSize[2] = a->unitNum / (reshapedSize[0] * reshapedSize[1]);

            /* we reshape dE/dc to a 3D tensor of size (x, y, z) where y = |b|.
               Then reduce along with z and x to obtain dE/db. */
            node->grad->Reshape(3, reshapedSize);

            if (b->mem != NULL)
                b->mem->LockBuf();
            XTensor * interGrad = NewTensorBufV2(2, reshapedSize, b->dataType, b->denseRatio, b->devID, b->mem);

            _ReduceSum(node->grad, interGrad, 2);

            XTensor * bGradTMP = NewTensorBufV2(b->grad, b->devID, b->mem);
            _ReduceSum(interGrad, bGradTMP, 0);
            if (beta != 1.0F)
                _ScaleAndShiftMe(bGradTMP, beta);
            _Sub(b->grad, bGradTMP, b->grad);
            DelTensorBuf(bGradTMP);

            node->grad->Reshape(order, dimSize);

            DelTensorBuf(interGrad);
            if (b->mem != NULL)
                b->mem->UnlockBuf();
        }
    }
}

/* 
gradient for sum
for 
c =  a + b * \beta
we have
dE/da = dE/dc 
dE/db = dE/dc * \beta
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradSum(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 2, "Wrong input tensor number for SUM!");

    XTensor * a = income.tails[0];
    XTensor * b = income.tails[1];
    DTYPE beta = income.GetParam(0);

    /* dE/da = dE/dc */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);
        _Sum(a->grad, node->grad, a->grad);
    }

    /* dE/db = dE/dc * \beta */
    if (!isEfficient || b->isGrad) {
        XNoder::MakeGrad(b);
        _Sum(b->grad, node->grad, b->grad, beta);
    }
}

/* 
gradient for sum with one dimension
c = a + b * \beta
where the size of b is equal to dimension n of a, i.e., |b| = a.dimSize[n]
dE/da = dE/dc
dE/db = dE/dc * a.reduce(0,...,n-1,n+1,...) * \beta

>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradSumDim(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 2, "Wrong input tensor number for SUMDIM!");

    XTensor * a = income.tails[0];
    XTensor * b = income.tails[1];
    int n = income.GetParamInt(0);
    DTYPE beta = income.GetParam(1);

    if (!isEfficient || a->isGrad) {
        /* dE/da = dE/dc */
        XNoder::MakeGrad(a);
        _Sum(a->grad, node->grad, a->grad);
    }

    /* dE/db = dE/dc * a.reduce(0,...,n-1,n+1,...) * \beta */
    if (!isEfficient || b->isGrad) {
        XNoder::MakeGrad(b);
        int order = a->order;
        int dimSize[MAX_TENSOR_DIM_NUM];
        memcpy(dimSize, a->dimSize, sizeof(int) * a->order);

        if (n == order - 1) {
            int reshapedSize[MAX_TENSOR_DIM_NUM];
            reshapedSize[0] = a->unitNum / dimSize[order - 1];
            reshapedSize[1] = dimSize[order - 1];

            /* we reshape dE/dc to a matrix whose column number is equal to the
               size of b. Then we can reduce the matrix into a row vector. */
            node->grad->Reshape(2, reshapedSize);

            if (b->mem != NULL)
                b->mem->LockBuf();
            XTensor * bGradTMP = NewTensorBufV2(b->grad, b->devID, b->mem);
            _ReduceSum(node->grad, bGradTMP, 0);
            if (beta != 1.0F)
                _ScaleAndShiftMe(bGradTMP, beta);
            _Sum(bGradTMP, b->grad, b->grad);
            DelTensorBuf(bGradTMP);
            if (b->mem != NULL)
                b->mem->UnlockBuf();

            node->grad->Reshape(order, dimSize);
        }
        else {
            int reshapedSize[MAX_TENSOR_DIM_NUM];
            reshapedSize[0] = 1;
            reshapedSize[1] = dimSize[n];
            reshapedSize[2] = 1;

            for (int i = 0; i < order; i++) {
                if (i < n)
                    reshapedSize[0] *= dimSize[i];
            }

            reshapedSize[2] = a->unitNum / (reshapedSize[0] * reshapedSize[1]);

            /* we reshape dE/dc to a 3D tensor of size (x, y, z) where y = |b|.
               Then reduce along with z and x to obtain dE/db. */
            node->grad->Reshape(3, reshapedSize);

            if (b->mem != NULL)
                b->mem->LockBuf();
            XTensor * interGrad = NewTensorBufV2(2, reshapedSize, b->dataType, b->denseRatio, b->devID, b->mem);

            _ReduceSum(node->grad, interGrad, 2);

            XTensor * bGradTMP = NewTensorBufV2(b->grad, b->devID, b->mem);
            _ReduceSum(interGrad, bGradTMP, 0);
            if (beta != 1.0F)
                _ScaleAndShiftMe(bGradTMP, beta);
            _Sum(bGradTMP, b->grad, b->grad);
            DelTensorBuf(bGradTMP);

            node->grad->Reshape(order, dimSize);

            DelTensorBuf(interGrad);
            if (b->mem != NULL)
                b->mem->UnlockBuf();
        }
    }
}

/* 
gradient for sum by broadcasting: 
c = a + b * \beta
where some dimensions of b are of size 1

dE/da = dE/dc
dE/db = dE/dc * a.reduce(0..n) * \beta 
where a.reduce(0..n) is the reduction along the dimension
whose size is 1 in b

>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradSumBroadcast(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 2, "Wrong input tensor number for SUMBROADCAST!");

    XTensor * a = income.tails[0];
    XTensor * b = income.tails[1];
    //DTYPE beta = income.GetParam(0);

    /* dE/da = dE/dc */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);
        _Sum(a->grad, node->grad, a->grad);
    }

    /* dE/db = dE/dc * a.reduce(0..n) * \beta  */
    if (!isEfficient || b->isGrad) {
        if (b->isVar || b->income.tailNum > 0) {
            ShowNTErrors("TODO");
        }
    }
}

/*
gradient for reduceMean
for
c = reduceMean(a, dim)
we have
dE/da = Unsqueeze(dE/dc) * 1/dimSizeA[dim]

>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradReduceMean(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for Reduce!");

    XTensor * a = income.tails[0];

    int dim = income.GetParamInt(0);
    int n = a->GetDim(dim);

    /* dE/da = Unsqueeze(dE/dc) * 1/dimSizeA[dim] */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);

        if (a->mem != NULL)
            a->mem->LockBuf();
        XTensor * tmp = NewTensorBufV2(a, a->devID, a->mem);
        _Unsqueeze(node->grad, tmp, dim, n);
        _ScaleAndShiftMe(tmp, 1.0F / n);
        _Sum(a->grad, tmp, a->grad);

        DelTensorBuf(tmp);
        if (a->mem != NULL)
            a->mem->UnlockBuf();
    }
}

/*
gradient for reduceSum
for
c = reduceSum(a, dim)
we have
dE/da = Unsqueeze(dE/dc)

>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradReduceSum(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for Reduce!");

    XTensor * a = income.tails[0];

    int dim = income.GetParamInt(0);
    int n = a->GetDim(dim);

    /* dE/da = Unsqueeze(dE/dc) */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);

        if (a->mem != NULL)
            a->mem->LockBuf();
        XTensor * tmp = NewTensorBufV2(a, a->devID, a->mem);
        _Unsqueeze(node->grad, tmp, dim, n);
        _Sum(a->grad, tmp, a->grad);
        DelTensorBuf(tmp);
        if (a->mem != NULL)
            a->mem->UnlockBuf();
    }
}

/*
gradient for reduceSumAll
for
c = reduceSumAll(a)
we have
dE/da = dE/dc * 1

>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradReduceSumAll(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for Reduce!");

    XTensor * a = income.tails[0];

    /* dE/da = dE/dc * 1 */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);

        if (a->mem != NULL)
            a->mem->LockBuf();
        XTensor * tmp = NewTensorBufV2(a, a->devID, a->mem);
        DTYPE value = node->grad->Get0D();
        tmp->SetDataFixed(value);
        _Sum(a->grad, tmp, a->grad);
        DelTensorBuf(tmp);
        if (a->mem != NULL)
            a->mem->UnlockBuf();
    }
}

/*
gradient for reduceSumSquared
for
c = \sum_i (a_i - b)^2 
we have
dE/da_i = Unsqueeze(dE/dc) * 2 * (a_i - b)
dE/db = dE/dc * -2 * n * \sum_i (a_i - b)

>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradReduceSumSquared(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 2, "Wrong input tensor number for Reduce!");

    XTensor * a = income.tails[0];
    XTensor * b = income.tails[1];
    if (a->mem != NULL)
        a->mem->LockBuf();
    XTensor * c = NewTensorBufV2(a, a->devID, a->mem);
    XTensor * d = NewTensorBufV2(a, a->devID, a->mem);
    XTensor * e = NewTensorBufV2(a, a->devID, a->mem);
    if ((b->mem != NULL) && (b->mem != a->mem)) {
        b->mem->LockBuf();
    }
    XTensor * f = NewTensorBufV2(b, b->devID, b->mem);

    int dim = income.GetParamInt(0);
    int n = a->GetDim(dim);

    _Unsqueeze(b, c, dim, n);
    _Sub(a, c, d);

    /* dE/da_i = Unsqueeze(dE/dc) * 2 * (a_i - b) */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);

        
        _ScaleAndShiftMe(d, 2.0F);
        _Unsqueeze(node->grad, e, dim, n);
        _Multiply(d, e, a->grad, 1.0F);
    }

    /* dE/db = dE/dc * -2 * n * \sum_i (a_i - b) */
    if (!isEfficient || b->isGrad) {
        XNoder::MakeGrad(b);

        _ReduceSum(d, f, dim);
        _ScaleAndShiftMe(f, -2.0F);
        _Multiply(node->grad, f, b->grad, 1.0F);
    }

    DelTensorBuf(f);
    if ((b->mem != NULL) && (b->mem != a->mem)) {
        b->mem->UnlockBuf();
    }
    DelTensorBuf(e);
    DelTensorBuf(d);
    DelTensorBuf(c);
    if (a->mem != NULL)
        a->mem->UnlockBuf();
}

/*
gradient for reduceVariance
for
c = (sum_i (a_i - b)^2) * 1/n
where b is the mean, and n is the size of a
we have
dE/da_i = Unsqueeze(dE/dc) * 2 * (a_i - b)/n
dE/db = dE/dc * -2 * \sum_i (a_i - b)

>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XMathGrad::GradReduceVariance(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 2, "Wrong input tensor number for Reduce!");

    XTensor * a = income.tails[0];
    XTensor * b = income.tails[1];
    if (a->mem != NULL)
        a->mem->LockBuf();
    XTensor * c = NewTensorBufV2(a, a->devID, a->mem);
    XTensor * d = NewTensorBufV2(a, a->devID, a->mem);
    XTensor * e = NewTensorBufV2(a, a->devID, a->mem);
    if ((b->mem != NULL) && (b->mem != a->mem)) {
        b->mem->LockBuf();
    }
    XTensor * f = NewTensorBufV2(b, b->devID, b->mem);

    int dim = income.GetParamInt(0);
    int n = a->GetDim(dim);

    _Unsqueeze(b, c, dim, n);
    _Sub(a, c, d);

    /* dE/da_i = Unsqueeze(dE/dc) * 2 * (a_i - b) / n */
    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);
        
        _ScaleAndShiftMe(d, 2.0F / n);
        _Unsqueeze(node->grad, e, dim, n);
        _Multiply(d, e, a->grad, 1.0F);
    }

    /* dE/db = dE/dc * -2 * \sum_i (a_i - b) */
    if (!isEfficient || b->isGrad) {
        XNoder::MakeGrad(b);

        _ReduceSum(d, f, dim);
        _ScaleAndShiftMe(f, -2.0F / n);
        _Multiply(node->grad, f, b->grad, 1.0F);
    }

    DelTensorBuf(f);
    if ((b->mem != NULL) && (b->mem != a->mem)) {
        b->mem->UnlockBuf();
    }
    DelTensorBuf(e);
    DelTensorBuf(d);
    DelTensorBuf(c);
    if (a->mem != NULL)
        a->mem->UnlockBuf();
}

/*
gradient for operation
for c = matmul(x, w) + b 
we have
dE/dx = dE/dc * w^T
dE/dw = x^T * dE/dc
dE/db = dE/dc * x.reduce(0,...,n-1,n+1,...)
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
an efficient manner
*/
void XMathGrad::GradMulAndShift(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 3, "wrong input tensor number")

    XTensor * x = income.tails[0];
    XTensor * w = income.tails[1];
    XTensor * b = income.tails[2];

    int n = income.GetParamInt(0);
    MATRIX_TRANS_TYPE transW = income.GetParamTrans(1);
    MATRIX_TRANS_TYPE transX = income.GetParamTrans(2);
    DTYPE alpha = income.GetParam(3);
    /* dE/db = dE/dc * x.reduce(0,...,n-1,n+1,...) */
    if (!isEfficient || b->isGrad) {
        XNoder::MakeGrad(b);

        int order = node->order;
        int dimSize[MAX_TENSOR_DIM_NUM];
        memcpy(dimSize, node->dimSize, sizeof(int) * node->order);

        /* compute dE/db */
        if (n == order - 1) {
            int reshapedSize[MAX_TENSOR_DIM_NUM];
            reshapedSize[0] = node->unitNum / dimSize[order - 1];
            reshapedSize[1] = dimSize[order - 1];

            /* we reshape dE/dc to a matrix whose column number is equal to the
            size of b. Then we can reduce the matrix into a row vector. */
            node->grad->Reshape(2, reshapedSize);

            if (b->mem != NULL)
                b->mem->LockBuf();
            XTensor * bGradTMP = NewTensorBufV2(b->grad, b->devID, b->mem);
            _ReduceSum(node->grad, bGradTMP, 0);
            _Sum(bGradTMP, b->grad, b->grad);
            DelTensorBuf(bGradTMP);
            if (b->mem != NULL)
                b->mem->UnlockBuf();

            node->grad->Reshape(order, dimSize);
        }
        else {
            int reshapedSize[MAX_TENSOR_DIM_NUM];
            reshapedSize[0] = 1;
            reshapedSize[1] = dimSize[n];
            reshapedSize[2] = 1;

            for (int i = 0; i < order; i++) {
                if (i < n)
                    reshapedSize[0] *= dimSize[i];
            }

            reshapedSize[2] = node->unitNum / (reshapedSize[0] * reshapedSize[1]);

            /* we reshape dE/dc to a 3D tensor of size (x, y, z) where y = |b|.
            Then reduce along with z and x to obtain dE/db. */
            node->grad->Reshape(3, reshapedSize);

            if (b->mem != NULL)
                b->mem->LockBuf();
            XTensor * interGrad = NewTensorBufV2(2, reshapedSize, b->dataType, b->denseRatio, b->devID, b->mem);
            _ReduceSum(node->grad, interGrad, 2);

            XTensor * bGradTMP = NewTensorBufV2(b->grad, b->devID, b->mem);
            _ReduceSum(interGrad, bGradTMP, 0);
            _Sum(bGradTMP, b->grad, b->grad);
            DelTensorBuf(bGradTMP);

            node->grad->Reshape(order, dimSize);

            DelTensorBuf(interGrad);
            if (b->mem != NULL)
                b->mem->UnlockBuf();
        }
    }

    if (!isEfficient || w->isGrad)
        XNoder::MakeGrad(w);
    if (!isEfficient || x->isGrad)
        XNoder::MakeGrad(x);

    /* compute dE/dx, dE/dw */
    XTensor * c = node;
    XTensor * dedc = node->grad;
    XTensor * dedw = w->grad;
    XTensor * dedx = x->grad;

    if (x->order == 2 && w->order == 2)
        GradMatrixMul(x, dedx, transX, w, dedw, transW, dedc, alpha, isEfficient);
    else if (transX == X_NOTRANS && x->order > 2 && w->order == 2) {
        int orderBackupX = x->order;
        int orderBackupC = c->order;
        int dimsBackupX[MAX_TENSOR_DIM_NUM];
        int dimsBackupC[MAX_TENSOR_DIM_NUM];
        memcpy(dimsBackupX, x->dimSize, sizeof(int) * x->order);
        memcpy(dimsBackupC, c->dimSize, sizeof(int) * c->order);

        x->Reshape(x->unitNum / x->GetDim(-1), x->GetDim(-1));
        c->Reshape(c->unitNum / c->GetDim(-1), c->GetDim(-1));
        if (!isEfficient || x->isGrad)
            dedx->Reshape(dedx->unitNum / dedx->GetDim(-1), dedx->GetDim(-1));
        dedc->Reshape(dedc->unitNum / dedc->GetDim(-1), dedc->GetDim(-1));

        GradMatrixMul(x, dedx, transX, w, dedw, transW, dedc, alpha, isEfficient);

        x->Reshape(orderBackupX, dimsBackupX);
        c->Reshape(orderBackupC, dimsBackupC);
        if (!isEfficient || x->isGrad)
            dedx->Reshape(orderBackupX, dimsBackupX);
        dedc->Reshape(orderBackupC, dimsBackupC);
    }
}

/*
gradient for operation
for c = relu(matmul(x, w) + b)
we have
dE/dx = dE/drelu * dE/dc * w^T
dE/dw = dE/drelu * dE/dc * x^T 
dE/db = dE/drelu * dE/dc * x.reduce(0,...,n-1,n+1,...)
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
an efficient manner
*/
void XMathGrad::GradMLP(XTensor* node, bool isEfficient)
{
    XLink& income = node->income;
    CheckNTErrors(income.tailNum == 3, "wrong input tensor number")

    _RectifyBackward(node, node, node->grad, node->grad);
    XTensor* x = income.tails[0];
    XTensor* w = income.tails[1];
    XTensor* b = income.tails[2];

    int n = income.GetParamInt(0);
    MATRIX_TRANS_TYPE transW = income.GetParamTrans(1);
    MATRIX_TRANS_TYPE transX = income.GetParamTrans(2);
    DTYPE alpha = income.GetParam(3);
    /* dE/db = dE/dc * x.reduce(0,...,n-1,n+1,...) */
    if (!isEfficient || b->isGrad) {
        XNoder::MakeGrad(b);

        int order = node->order;
        int dimSize[MAX_TENSOR_DIM_NUM];
        memcpy(dimSize, node->dimSize, sizeof(int) * node->order);

        /* compute dE/db */
        if (n == order - 1) {
            int reshapedSize[MAX_TENSOR_DIM_NUM];
            reshapedSize[0] = node->unitNum / dimSize[order - 1];
            reshapedSize[1] = dimSize[order - 1];

            /* we reshape dE/dc to a matrix whose column number is equal to the
            size of b. Then we can reduce the matrix into a row vector. */
            node->grad->Reshape(2, reshapedSize);

            _ReduceSum(node->grad, b->grad, 0);

            node->grad->Reshape(order, dimSize);
        }
        else {
            int reshapedSize[MAX_TENSOR_DIM_NUM];
            reshapedSize[0] = 1;
            reshapedSize[1] = dimSize[n];
            reshapedSize[2] = 1;

            for (int i = 0; i < order; i++) {
                if (i < n)
                    reshapedSize[0] *= dimSize[i];
            }

            reshapedSize[2] = node->unitNum / (reshapedSize[0] * reshapedSize[1]);

            /* we reshape dE/dc to a 3D tensor of size (x, y, z) where y = |b|.
            Then reduce along with z and x to obtain dE/db. */
            node->grad->Reshape(3, reshapedSize);

            if (b->mem != NULL)
                b->mem->LockBuf();
            XTensor* interGrad = NewTensorBufV2(2, reshapedSize, b->dataType, b->denseRatio, b->devID, b->mem);
            _ReduceSum(node->grad, interGrad, 2);

            XTensor* bGradTMP = NewTensorBufV2(b->grad, b->devID, b->mem);
            _ReduceSum(interGrad, bGradTMP, 0);
            _Sum(bGradTMP, b->grad, b->grad);
            DelTensorBuf(bGradTMP);

            node->grad->Reshape(order, dimSize);

            DelTensorBuf(interGrad);
            if (b->mem != NULL)
                b->mem->UnlockBuf();
        }
    }

    if (!isEfficient || w->isGrad)
        XNoder::MakeGrad(w);
    if (!isEfficient || x->isGrad)
        XNoder::MakeGrad(x);

    /* compute dE/dx, dE/dw */
    XTensor* c = node;
    XTensor* dedc = node->grad;
    XTensor* dedw = w->grad;
    XTensor* dedx = x->grad;

    if (x->order == 2 && w->order == 2)
        GradMatrixMul(x, dedx, transX, w, dedw, transW, dedc, alpha, isEfficient);
    else if (transX == X_NOTRANS && x->order > 2 && w->order == 2) {
        int orderBackupX = x->order;
        int orderBackupC = c->order;
        int dimsBackupX[MAX_TENSOR_DIM_NUM];
        int dimsBackupC[MAX_TENSOR_DIM_NUM];
        memcpy(dimsBackupX, x->dimSize, sizeof(int) * x->order);
        memcpy(dimsBackupC, c->dimSize, sizeof(int) * c->order);

        x->Reshape(x->unitNum / x->GetDim(-1), x->GetDim(-1));
        c->Reshape(c->unitNum / c->GetDim(-1), c->GetDim(-1));
        if (!isEfficient || x->isGrad)
            dedx->Reshape(dedx->unitNum / dedx->GetDim(-1), dedx->GetDim(-1));
        dedc->Reshape(dedc->unitNum / dedc->GetDim(-1), dedc->GetDim(-1));

        GradMatrixMul(x, dedx, transX, w, dedw, transW, dedc, alpha, isEfficient);

        x->Reshape(orderBackupX, dimsBackupX);
        c->Reshape(orderBackupC, dimsBackupC);
        if (!isEfficient || x->isGrad)
            dedx->Reshape(orderBackupX, dimsBackupX);
        dedc->Reshape(orderBackupC, dimsBackupC);
    }
}

}
