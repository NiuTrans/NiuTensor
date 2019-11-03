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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2019-10-13
 */

#include "Stack.h"
#include "IsSameShaped.h"
#include "../../XUtility.h"
#include "../../XName.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* stack small tensors into a big tensor along with a dimension */
void _Stack(const TensorList * smalls, XTensor * t, int dim)
{
    dim = (dim < 0 ? t->order - 1 : dim);
    int count = smalls->count;

    CheckNTErrors(smalls != NULL, "Invalid list!");
    CheckNTErrors(count > 0, "Empty list!");
    CheckNTErrors(dim >= 0 && dim < t->order, "Wrong range of dim");
    for (int i = 1; i < count; i++) {
        XTensor * tmp1 = smalls->GetItem(i);
        XTensor * tmp2 = smalls->GetItem(i-1);
        CheckNTErrors(_IsSameShaped(tmp1, tmp2), "The input tensor must be same size!");
    }

    int blockSize = 1;
    int blockNum = 1;
    int gridSize = 1;
    int gridNum = 1;

    XTensor * smallsItem0 = smalls->GetItem(0);
    int unitNum = smallsItem0->unitNum;
    int unitSize = smallsItem0->unitSize;
    int itemSize = unitNum * unitSize;

    for (int i = 0; i < smallsItem0->order; i++) {
        if (i >= dim)
            blockSize *= smallsItem0->dimSize[i];
        else
            blockNum *= smallsItem0->dimSize[i];
    }

    /* merging with fewer data copy operations */
    if (count * gridNum <= MIN_TENSOR_MERGE_LIST_NUM) {
        int sPitch = blockSize * unitSize;
        int tPtich = blockSize * count * unitSize;
        int mSize = blockSize * unitSize;
        int n = blockNum;
        int sStep = 0;
        int tStep = blockSize * unitSize;
        char * tData = (char*)t->data;
        for (int k = 0; k < count; k++) {
            XTensor * s = smalls->GetItem(k);
            char * sData = (char*)s->data;
            XMemCopy2D(tData + k * tStep, tPtich, t->devID,
                       sData + k * sStep, sPitch, s->devID,
                       mSize, n);
        }
    }
    else {
        ShowNTErrors("TO DO!!!");
    }
}

/* stack small tensors into a big tensor along with a dimension (return an XTensor structure) */
XTensor Stack(const TensorList &smalls, int dim)
{
    int count = smalls.count;
    CheckNTErrors(count > 0, "Empty list!");
    CheckNTErrors(dim >= 0, "Illegal dimension to concatenate!");

    XTensor * tensor = smalls.GetItem(0);
    int order = tensor->order + 1;
    int * dimSize = new int[order];

    for (int i = 0; i < order; i++) {
        if (i < dim)
            dimSize[i] = tensor->GetDim(i);
        else if (i > dim)
            dimSize[i] = tensor->GetDim(i);
        else if (i == dim)
            dimSize[i] = count;
    }

    float dr = (!tensor->isSparse) ? 1.0F : tensor->denseRatio;
    XTensor t(order, dimSize, tensor->dataType, dr, tensor->devID, tensor->mem);
    t.SetTMPFlag();

    /* destroy variables */
    delete[] dimSize;

    /* call _Stack function */
    _Stack(&smalls, &t, dim);
                
    /* tensor connection */
    for (int i = 0; i < count; i++) {
        XTensor * tmp = smalls.GetItem(i);
        if (tmp->enableGrad == false)
            return t;
    }

    XLink::MakeLink(&smalls, &t, SHAPE_STACK);
    XLink::AddParamToHeadInt(&t, dim);

    return t;
}

/* check the shape of target tensor */
bool CheckStackShape(const TensorList &smalls, XTensor &t, int dim)
{
    XTensor * tensor = (XTensor*)smalls.GetItem(0);
    int order = tensor->order;

    for (int i = 0; i < tensor->order; i++) {
        if (i < dim)
            if (t.GetDim(i) != tensor->GetDim(i)) 
                return false;
        else if (i > dim)
            if (t.GetDim(i) != tensor->GetDim(i-1)) 
                return false;
        else if (i == dim)
            if (t.GetDim(i) != smalls.count) 
                return false;
    }

    return true;
}

/* stack small tensors into a big tensor along with a dimension */
void Stack(const TensorList &smalls, XTensor &t, int dim)
{
    int count = smalls.count;
    CheckNTErrors(count > 0, "Empty list!");
    CheckNTErrors(dim >= 0, "Illegal dimension to concatenate!");

    if (!t.isInit || !CheckStackShape(smalls, t, dim)) {
        XTensor * tensor = smalls.GetItem(0);
        int order = tensor->order + 1;
        int * dimSize = new int[order];

        for (int i = 0; i < order; i++) {
            if (i < dim)
                dimSize[i] = tensor->GetDim(i);
            else if (i > dim)
                dimSize[i] = tensor->GetDim(i-1);
            else if (i == dim)
                dimSize[i] = count;
        }

        float dr = (!tensor->isSparse) ? 1.0F : tensor->denseRatio;
        InitTensorV2(&t, order, dimSize, tensor->dataType, dr, tensor->devID, tensor->mem);

        /* destroy variables */
        delete[] dimSize;
    }

    /* call _Stack function */
    _Stack(&smalls, &t, dim);
                
    /* tensor connection */
    for (int i = 0; i < count; i++) {
        XTensor * tmp = smalls.GetItem(i);
        if (tmp->enableGrad == false)
            return;
    }

    XLink::MakeLink(&smalls, &t, SHAPE_STACK);
    XLink::AddParamToHeadInt(&t, dim);
}

} // namespace nts(NiuTrans.Tensor)

