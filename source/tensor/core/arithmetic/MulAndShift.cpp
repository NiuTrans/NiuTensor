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
* $Created by: JIANG Yufan (email: jiangyufan2018@outlook.com) 2019-02-27
*/

#include "../../XTensor.h"
#include "../../XDevice.h"
#include "../../XName.h"
#include "MulAndShift.h"
#include "MatrixMul.h"
#include "Sum.h"

namespace nts { // namespace nts(NiuTrans.Tensor)
/*
operation c = x * w + b  MulAndShift
>> x - tensor x
>> w - tensor w
>> b - tensor b
>> parallelRunner - parallel processing module
<< return - the result of matrix multiplication
*/
XTensor MulAndShift(const XTensor &x, const XTensor &w, const XTensor &b,
                    DTYPE alpha, XPRunner * parallelRunner)
{
    CheckNTErrors(x.dataType == w.dataType, "Input tensors should have the same data type!");
    CheckNTErrors(x.order >= 2 && w.order >= 2, "Input tensors must have a order >= 2!");

    int xn = x.dimSize[x.order - 2];
    int xm = x.dimSize[x.order - 1];
    int wn = w.dimSize[w.order - 2];
    int wm = w.dimSize[w.order - 1];

    CheckNTErrors(xm == wn, "Unmatched tensors in multiplication!");

    int order = x.order + w.order - 2;
    int sub = 0;
    int * dimSize = new int[order];
    for (int i = 0; i < x.order - 2; i++)
        dimSize[sub++] = x.dimSize[i];
    for (int i = 0; i < w.order - 2; i++)
        dimSize[sub++] = w.dimSize[i];

    dimSize[sub++] = xn;
    dimSize[sub++] = wm;

    float dr = (!x.isSparse || !w.isSparse) ? 1.0F : MAX(x.denseRatio, w.denseRatio);

    if (x.mem != NULL)
        x.mem->LockBuf();
    XTensor * tmp = NewTensorBufV2(order, dimSize, x.dataType, dr, x.devID, x.mem);

    /* call _MatrixMul function */
    _MatrixMul(&x, X_NOTRANS, &w, X_NOTRANS, tmp, alpha, 0, parallelRunner);

    XTensor c(tmp);
    c.SetTMPFlag();

    if (b.order == 0)
        ScaleAndShift(*tmp, c, 1.0F, b.Get0D());
    else {
        int n = GetBroadcastDimIndex(tmp, b);

        if (n == -1) {
            /* call _Sum function */
            _Sum(tmp, &b, &c);

            // TODO!!
            ShowNTErrors("TODO!");
        }
        else if (n >= 0 && n < tmp->order) {
            /* call _SumDim function */
            _SumDim(tmp, &b, &c, n);
        }
        else {
            ShowNTErrors("Something is wrong!");
        }
        /* tensor connections */
        if (w.enableGrad && b.enableGrad) {
            XLink::MakeLink(&x, &w, &b, &c, MATH_MULANDSHIFT);
            XLink::AddParamToHeadInt(&c, n);
            XLink::AddParamToHeadTrans(&c, X_NOTRANS);
            XLink::AddParamToHeadTrans(&c, X_NOTRANS);
            XLink::AddParamToHead(&c, alpha);
        }
    }

    /* destroy variables */
    delete[] dimSize;
    DelTensorBuf(tmp);
    if (x.mem != NULL)
        x.mem->UnlockBuf();

    return c;
}

/*
operation c = x * w + b  MulAndShift
>> x - tensor x
>> w - tensor w
>> b - tensor b
>> parallelRunner - parallel processing module
<< return - the result of matrix multiplication
*/
XTensor MulAndShift(const XTensor& x, MATRIX_TRANS_TYPE transposedX,
                    const XTensor& w, MATRIX_TRANS_TYPE transposedW,
                    const XTensor& b, DTYPE alpha, XPRunner* parallelRunner)
{
    CheckNTErrors(x.dataType == w.dataType, "Input tensors should have the same data type!");
    CheckNTErrors(x.order >= 2 && w.order >= 2, "Input tensors must have a order >= 2!");

    int xn = transposedX == X_TRANS ? x.dimSize[x.order - 1] : x.dimSize[x.order - 2];
    //int xm = transposedX == X_TRANS ? x.dimSize[x.order - 2] : x.dimSize[x.order - 1];
    //int wn = transposedW == X_TRANS ? w.dimSize[w.order - 1] : w.dimSize[w.order - 2];
    int wm = transposedW == X_TRANS ? w.dimSize[w.order - 2] : w.dimSize[w.order - 1];

    int order = x.order + w.order - 2;
    int sub = 0;
    int * dimSize = new int[order];
    for (int i = 0; i < x.order - 2; i++)
        dimSize[sub++] = x.dimSize[i];
    for (int i = 0; i < w.order - 2; i++)
        dimSize[sub++] = w.dimSize[i];
    dimSize[sub++] = xn;
    dimSize[sub++] = wm;

    float dr = (!x.isSparse || !w.isSparse) ? 1.0F : MAX(x.denseRatio, w.denseRatio);

    if (x.mem != NULL)
        x.mem->LockBuf();
    XTensor * tmp = NewTensorBufV2(order, dimSize, x.dataType, dr, x.devID, x.mem);

    /* call _MatrixMul function */
    _MatrixMul(&x, transposedX, &w, transposedW, tmp, alpha, 0, parallelRunner);

    XTensor c(tmp);
    c.SetTMPFlag();

    int n = GetBroadcastDimIndex(tmp, b);

    if (n == -1) {
        /* call _Sum function */
        _Sum(tmp, &b, &c);

        // TODO!!
        ShowNTErrors("TODO!");

    }
    else if (n >= 0 && n < tmp->order) {
        /* call _SumDim function */
        _SumDim(tmp, &b, &c, n);

    }
    else {
        ShowNTErrors("Something is wrong!");
    }

    /* tensor connections */
    if (w.enableGrad && b.enableGrad) {
        XLink::MakeLink(&x, &w, &b, &c, MATH_MULANDSHIFT);
        XLink::AddParamToHeadInt(&c, n);
        XLink::AddParamToHeadTrans(&c, transposedX);
        XLink::AddParamToHeadTrans(&c, transposedW);
    }

    /* destroy variables */
    delete[] dimSize;
    DelTensorBuf(tmp);
    if (x.mem != NULL)
        x.mem->UnlockBuf();

    return c;
}

}
