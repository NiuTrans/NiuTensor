/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2018, Natural Language Processing Lab, Northestern University.
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

#include "../tensor/XTensor.h"

#ifndef __XBACKWARDMATH_H__
#define __XBACKWARDMATH_H__

namespace nts{

/* this class computes the gradient for math operations given a node */
class XMathGrad
{
public:
    /* compute dE/dx of a node */
    static
    void MakeGrad(XTensor * node, bool isEfficient);

    /* indicates whether the node is for a math operation */
    static
    bool IsMathOP(XTensor * node);

private:
    
    /* gradient for absolute */
    static
    void GradAbsolute(XTensor * node, bool isEfficient);
    
    /* gradient for cos */
    static
    void GradCos(XTensor * node, bool isEfficient);
    
    /* gradient for exp */
    static
    void GradExp(XTensor * node, bool isEfficient);

    /* gradient for log: c =  log(a) */
    static
    void GradLog(XTensor * node, bool isEfficient);
    
    /* gradient for round */
    static
    void GradRound(XTensor * node, bool isEfficient);
    
    /* gradient for sign */
    static
    void GradSign(XTensor * node, bool isEfficient);

    /* gradient for sin */
    static
    void GradSin(XTensor * node, bool isEfficient);

    /* gradient for tan */
    static
    void GradTan(XTensor * node, bool isEfficient);

    /* gradient for clip */
    static
    void GradClip(XTensor * node, bool isEfficient);

    /* gradient for Divide */
    static
    void GradDiv(XTensor * node, bool isEfficient);

    /* gradient for DivideDim */
    static
    void GradDivDim(XTensor * node, bool isEfficient);

    /* gradient for matrix multiply: c = matmul(a, b) * \alpha */
    static
    void GradMatrixMul(XTensor * node, bool isEfficient);
    
    /* gradient for matrix multiply: c = matmul(a, b) * \alpha */
    static
    void GradMatrixMul(XTensor * a, XTensor * deda, MATRIX_TRANS_TYPE transA,
                       XTensor * b, XTensor * dedb, MATRIX_TRANS_TYPE transB,
                       XTensor * dedc, DTYPE alpha, bool isEfficient);

    /* gradient for matrix multiply in batch mode.
       for each batch: c_i = matmul(a_i, b_i) * \alpha */
    static
    void GradMatrixMulBatched(XTensor * node, bool isEfficient);

    /* gradient for multiply (dot production): c =  a * b * \alpha */
    static
    void GradMultiply(XTensor * node, bool isEfficient);

    /* gradient for multiply one dimension: c =  a * b * \alpha 
       where the size of b is equal to that of one dimension of a */
    static
    void GradMultiplyDim(XTensor * node, bool isEfficient);

    /* gradient for multiply one dimension: c =  a * b
       where some dimensions of b are of size 1 */
    static
    void GradMultiplyBroadcast(XTensor * node, bool isEfficient);

    /* gradient for negate */
    static
    void GradNegate(XTensor * node, bool isEfficient);
    
    /* gradient for normalize */
    static
    void GradNormalize(XTensor * node, bool isEfficient);

    /* gradient for power */
    static
    void GradPower(XTensor * node, bool isEfficient);

    /* gradient for ScaleAndShift */
    static
    void GradScaleAndShift(XTensor * node, bool isEfficient);

    /* gradient for Scale */
    static
    void GradScale(XTensor * node, bool isEfficient);

    /* gradient for Shift */
    static
    void GradShift(XTensor * node, bool isEfficient);

    /* gradient for Descale */
    static
    void GradDescale(XTensor * node, bool isEfficient);

    /* gradient for Minus */
    static
    void GradSub(XTensor * node, bool isEfficient);
    
    /* gradient for sub with one dimension: c = a - b * \beta
    where the size of b is equal to that of one dimension of a */
    static
    void GradSubDim(XTensor * node, bool isEfficient);

    /* gradient for sum: c =  a + b * \beta */
    static
    void GradSum(XTensor * node, bool isEfficient);

    /* gradient for sum with one dimension: c = a + b * \beta
       where the size of b is equal to that of one dimension of a */
    static
    void GradSumDim(XTensor * node, bool isEfficient);

    /* gradient for sum by broadcasting: c = a + b * \beta
       where some dimensions of b are of size 1 */
    static
    void GradSumBroadcast(XTensor * node, bool isEfficient);

    /* gradient for reduceMean */
    static
    void GradReduceMean(XTensor * node, bool isEfficient);

    /* gradient for reduceSum */
    static
    void GradReduceSum(XTensor * node, bool isEfficient);

    /* gradient for reduceSumSquared */
    static
    void GradReduceSumSquared(XTensor * node, bool isEfficient);

    /* gradient for reduceVariance */
    static
    void GradReduceVariance(XTensor * node, bool isEfficient);

    /* gradient for operation */
    static
    void GradMulAndShift(XTensor * node, bool isEfficient);
};

}

#endif