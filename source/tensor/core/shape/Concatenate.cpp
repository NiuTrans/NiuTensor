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
#include "../../XName.h"
#include "../shape/IsSameShaped.h"
#include "Concatenate.h"
#include "Merge.h"
#include "ConcatenateSolely.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
concatenate a list of tensors along a given dimension

Note that this is actually a wrapper that selects "ConcatenateSolely"
or "Merge" by means of the tensor shapes

>> smalls - a list of tensors for concatenation
>> big - the resulting tensor
>> dim - which dimension we perform the concatenation
*/
void _Concatenate(const TensorList * smalls, XTensor * big, int dim)
{
    bool uniform = true;
    for (int i = 1; i < smalls->count; i++) {
        XTensor * a = (XTensor*)smalls->GetItem(i - 1);
        XTensor * b = (XTensor*)smalls->GetItem(i);
        CheckNTErrors((a && b), "Empty input tensors!");
        if (!_IsSameShaped(a, b))
            uniform = false;
    }

    if (uniform)
        _Merge(smalls, big, dim);
    else
        _ConcatenateSolely(smalls, big, dim);
}

/*
concatenate a list of tensors along a given dimension (return an XTensor structure)
make a new tensor to keep the result and return it

Note that this is actually a wrapper that selects "ConcatenateSolely"
or "Merge" by means of the tensor shapes

>> smalls - a list of tensors for concatenation
>> big - the resulting tensor
>> dim - which dimension we perform the concatenation
<< return - the tensor of concatenating a list of tensors along a given dimension
*/
XTensor Concatenate(const TensorList &smalls, int dim)
{
    CheckNTErrors(smalls.count > 0, "Empty list!");
    CheckNTErrors(dim >= 0, "Illegal dimension to concatenate!");

    bool uniform = true;
    for (int i = 1; i < smalls.count; i++) {
        XTensor * a = (XTensor*)smalls.GetItem(i - 1);
        XTensor * b = (XTensor*)smalls.GetItem(i);
        CheckNTErrors((a && b), "Empty input tensors!");
        if (!_IsSameShaped(a, b))
            uniform = false;
    }
    XTensor * tensor = (XTensor*)smalls.GetItem(0);
    int order = tensor->order;
    int * dimSize = new int[order];

    if (uniform) {
        for (int i = 0; i < tensor->order; i++) {
            if (i != dim)
                dimSize[i] = tensor->dimSize[i];
            else
                dimSize[i] = tensor->dimSize[dim] * smalls.count;
        }

        float dr = (!tensor->isSparse) ? 1.0F : tensor->denseRatio;
        XTensor big(order, dimSize, tensor->dataType, dr, tensor->devID, tensor->mem);
        big.SetTMPFlag();

        /* call _Merge function */
        _Merge(&smalls, &big, dim);
                
        /* tensor connection */
        if (tensor->enableGrad) {
            XLink::MakeLink(&smalls, &big, SHAPE_MERGE);
            XLink::AddParamToHeadInt(&big, dim);
        }

        /* destroy variables */
        delete[] dimSize;

        return big;
    }
    else {
        for (int i = 0; i < tensor->order; i++)
            if (i != dim)
                dimSize[i] = tensor->dimSize[i];

        int catDimSize = 0;
        for (int i = 0; i < smalls.count; i++) {
            XTensor * tensor = (XTensor*)smalls.GetItem(i);
            catDimSize += tensor->dimSize[dim];
        }
        dimSize[dim] = catDimSize;

        float dr = (!tensor->isSparse) ? 1.0F : tensor->denseRatio;
        XTensor big(order, dimSize, tensor->dataType, dr, tensor->devID, tensor->mem);
        big.SetTMPFlag();

        /* call _ConcatenateSolely function */
        _ConcatenateSolely(&smalls, &big, dim);

        /* tensor connection */
        if (tensor->enableGrad) {
            XLink::MakeLink(&smalls, &big, SHAPE_CONCATENATE);
            XLink::AddParamToHeadInt(&big, dim);
        }

        /* destroy variables */
        delete[] dimSize;

        return big;
    }
}

bool CheckConcatenateShape(const TensorList &smalls, int dim, XTensor &big, bool uniform)
{
    XTensor * tensor = (XTensor*)smalls.GetItem(0);
    int order = tensor->order;
    int * dimSize = new int[order];

    if (uniform) {
        for (int i = 0; i < tensor->order; i++) {
            if (i != dim)
                dimSize[i] = tensor->dimSize[i];
            else
                dimSize[i] = tensor->dimSize[dim] * smalls.count;
        }
    }
    else {
        for (int i = 0; i < tensor->order; i++)
            if (i != dim)
                dimSize[i] = tensor->dimSize[i];

        int catDimSize = 0;
        for (int i = 0; i < smalls.count; i++) {
            XTensor * tensor = (XTensor*)smalls.GetItem(i);
            catDimSize += tensor->dimSize[dim];
        }
        dimSize[dim] = catDimSize;
    }

    for (int i = 0; i < order; i++) {
        if (dimSize[i] != big.dimSize[i]) {
            delete[] dimSize;
            return false;
        }
    }

    delete[] dimSize;
    return false;
}

void Concatenate(const TensorList & smalls, XTensor & big, int dim)
{
    CheckNTErrors(smalls.count > 0, "Empty list!");
    CheckNTErrors(dim >= 0, "Illegal dimension to concatenate!");

    bool uniform = true;
    for (int i = 1; i < smalls.count; i++) {
        XTensor * a = (XTensor*)smalls.GetItem(i - 1);
        XTensor * b = (XTensor*)smalls.GetItem(i);
        CheckNTErrors((a && b), "Empty input tensors!");
        if (!_IsSameShaped(a, b))
            uniform = false;
    }

    if (!big.isInit || !CheckConcatenateShape(smalls, dim, big, uniform)) {
        XTensor * tensor = (XTensor*)smalls.GetItem(0);
        int order = tensor->order;
        int * dimSize = new int[order];

        if (uniform) {
            for (int i = 0; i < tensor->order; i++) {
                if (i != dim)
                    dimSize[i] = tensor->dimSize[i];
                else
                    dimSize[i] = tensor->dimSize[dim] * smalls.count;
            }

            float dr = (!tensor->isSparse) ? 1.0F : tensor->denseRatio;
            InitTensorV2(&big, order, dimSize, tensor->dataType, dr, tensor->devID, tensor->mem);
        }
        else {
            for (int i = 0; i < tensor->order; i++)
                if (i != dim)
                    dimSize[i] = tensor->dimSize[i];

            int catDimSize = 0;
            for (int i = 0; i < smalls.count; i++) {
                XTensor * tensor = (XTensor*)smalls.GetItem(i);
                catDimSize += tensor->dimSize[dim];
            }
            dimSize[dim] = catDimSize;

            float dr = (!tensor->isSparse) ? 1.0F : tensor->denseRatio;
            InitTensorV2(&big, order, dimSize, tensor->dataType, dr, tensor->devID, tensor->mem);
        }    
        /* destroy variables */
        delete[] dimSize;
    }

    if (uniform) {
        /* call _Merge function */
        _Merge(&smalls, &big, dim);
                
        /* tensor connection */
        if (big.enableGrad) {
            XLink::MakeLink(&smalls, &big, SHAPE_MERGE);
            XLink::AddParamToHeadInt(&big, dim);
        }
    }
    else {
        /* call _ConcatenateSolely function */
        _ConcatenateSolely(&smalls, &big, dim);

        /* tensor connection */
        if (big.enableGrad) {
            XLink::MakeLink(&smalls, &big, SHAPE_CONCATENATE);
            XLink::AddParamToHeadInt(&big, dim);    
        }
    }
}

/*
concatenate two tensors along a given dimension

>> smallA - one tensor for concatenation
>> smallB - the other tensor for concatenation
>> big - the resulting tensor
>> dim - which dimension we perform the concatenation
*/
void _Concatenate(const XTensor * smallA, const XTensor * smallB, XTensor * big, int dim)
{
    TensorList smalls(2);
    smalls.Add((XTensor*)smallA);
    smalls.Add((XTensor*)smallB);

    _Concatenate(&smalls, big, dim);
}

/*
concatenate two tensors along a given dimension (return an XTensor structure).
make a new tensor to keep the result and return it.

>> smallA - one tensor for concatenation
>> smallB - the other tensor for concatenation
>> big - the resulting tensor
>> dim - which dimension we perform the concatenation
<< return - the tensor of concatenating two tensor along a given dimension
*/
XTensor Concatenate(const XTensor &smallA, const XTensor &smallB, int dim)
{
    CheckNTErrors(dim >= 0, "Illegal dimension to concatenate!");

    TensorList smalls(2);
    smalls.Add((XTensor*)&smallA);
    smalls.Add((XTensor*)&smallB);

    bool uniform = true;
    for (int i = 1; i < smalls.count; i++) {
        XTensor * a = (XTensor*)smalls.Get(i - 1);
        XTensor * b = (XTensor*)smalls.Get(i);
        CheckNTErrors((a && b), "Empty input tensors!");
        if (!_IsSameShaped(a, b))
            uniform = false;
    }
    XTensor * tensor = (XTensor*)smalls.Get(0);
    int order = tensor->order;
    int * dimSize = new int[order];

    if (uniform) {
        for (int i = 0; i < tensor->order; i++) {
            if (i != dim)
                dimSize[i] = tensor->dimSize[i];
            else
                dimSize[i] = tensor->dimSize[dim] * smalls.count;
        }

        float dr = (!tensor->isSparse) ? 1.0F : tensor->denseRatio;
        XTensor big(order, dimSize, tensor->dataType, dr, tensor->devID, tensor->mem);
        big.SetTMPFlag();

        /* call _Merge function */
        _Merge(&smalls, &big, dim);
                
        /* tensor connection */
        if (tensor->enableGrad) {
            XLink::MakeLink(&smalls, &big, SHAPE_MERGE);
            XLink::AddParamToHeadInt(&big, dim);
        }

        /* destroy variables */
        delete[] dimSize;

        return big;
    }
    else {
        for (int i = 0; i < tensor->order; i++)
            if (i != dim)
                dimSize[i] = tensor->dimSize[i];

        int catDimSize = 0;
        for (int i = 0; i < smalls.count; i++) {
            XTensor * tensor = (XTensor*)smalls.Get(i);
            catDimSize += tensor->dimSize[dim];
        }
        dimSize[dim] = catDimSize;

        float dr = (!tensor->isSparse) ? 1.0F : tensor->denseRatio;
        XTensor big(order, dimSize, tensor->dataType, dr, tensor->devID, tensor->mem);
        big.SetTMPFlag();

        /* call _ConcatenateSolely function */
        _ConcatenateSolely(&smalls, &big, dim);

        /* tensor connection */
        if (tensor->enableGrad) {
            XLink::MakeLink(&smalls, &big, SHAPE_CONCATENATE);
            XLink::AddParamToHeadInt(&big, dim);
        }

        /* destroy variables */
        delete[] dimSize;

        return big;
    }
}

} // namespace nts(NiuTrans.Tensor)