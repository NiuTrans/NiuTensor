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
#include "../../XUtility.h"
#include "../../XName.h"
#include "../shape/IsSameShaped.h"
#include "Merge.h"
#include "MakeMergeBlockIndex.h"
#include "../movement/CopyBlocksOnSite.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
transform a tensor by merging it along with a dimension.

e.g., (N/3, M, 3) -> (N, M)

>> s - the source tensor
>> t - the target tensor (for return)
>> whereToMerge - the merging operation is along with which dimension
>> leadingDim - the leading dimension of merging, take (N/3, M, 3) -> (N, M) 
   for example, whereToMerge = 0 (i.e., the dimension for "N/3")
   leadingDim = 2 (i.e., the dimension for "3")
*/
void _Merge(const XTensor * s, XTensor * t, int whereToMerge, int leadingDim)
{
    if(leadingDim < 0)
        leadingDim = 0;

    if (leadingDim >= s->order)
        leadingDim = leadingDim - s->order;

    CheckNTErrors((s != NULL && t != NULL), "Invalid tensors!");
    CheckNTErrors((s->devID == t->devID || (s->devID < 0 && t->devID < 0)),
                  "the data must be kept on the same device!");

    CheckNTErrors((s->unitNum == t->unitNum && s->unitSize == t->unitSize), "Unmatched tensors!");
    CheckNTErrors((s->order == t->order + 1), "Unmatched tensors!");
    CheckNTErrors((leadingDim < whereToMerge), "Invalid leading dimension!");

    for (int i = 0; i < s->order; i++) {
        if (i == whereToMerge) {
            
            CheckNTErrors((t->dimSize[i - 1] == s->dimSize[i] * s->dimSize[leadingDim]),
                          "Unmatched tensor sizes!");
        }
        else if (i < leadingDim){
            CheckNTErrors((s->dimSize[i] == t->dimSize[i]),
                          "Unmatched tensor sizes!");
        }
        else if (i > leadingDim) {
            CheckNTErrors((s->dimSize[i] == t->dimSize[i - 1]),
                          "Unmatched tensor sizes!");
        }
    }

    int blockSize = 1;
    int blockNum = 1;
    int gridSize = 1;
    int gridNum = 1;
    int mergedNum = s->dimSize[leadingDim];

    for (int i = 0; i < s->order; i++) {
        if (i >= leadingDim) {
            if (i >= whereToMerge)
                blockSize *= s->dimSize[i];
            else
                blockNum *= s->dimSize[i];
        }
    }

    CheckNTErrors((s->unitNum % (blockSize * blockNum) == 0), "Incorrect size!");

    /* a grid has a number of blocks. there might be several grids */
    gridSize = blockNum;
    gridNum = s->unitNum / (blockSize * blockNum);

    if (mergedNum * gridNum <= MIN_TENSOR_MERGE_NUM) {
        int sPitch = blockSize * s->unitSize;
        int tPtich = blockSize * mergedNum * t->unitSize;
        int mSize = blockSize * t->unitSize;
        int n = blockNum / mergedNum;
        int sStep = n * sPitch;
        int tStep = blockSize * t->unitSize;
        for (int g = 0; g < gridNum; g++) {
            char * tData = (char*)t->data + g * blockSize * blockNum * t->unitSize;
            char * sData = (char*)s->data + g * blockSize * blockNum * s->unitSize;
            for (int k = 0; k < mergedNum; k++) {
                XMemCopy2D(tData + k * tStep, tPtich, t->devID,
                           sData + k * sStep, sPitch, s->devID, mSize, n);
            }
        }
    }
    else {
        XMem * mem = s->mem;
        int size = s->unitNum * s->unitSize;

        bool isOnSameDevice = (s->devID < 0 && t->devID < 0) || (s->devID == t->devID);

        void * dataTMP = t->data;

        if (!isOnSameDevice)
            dataTMP = mem != NULL ? mem->AllocBuf(mem->devID, size) : XMemAlloc(mem->devID, size);

        int blockNumInMerge = s->dimSize[leadingDim];
        int splitSizeInGrid = gridSize / blockNumInMerge;
        int realBlockSize = blockSize * t->unitSize;

        int * blockIndex = (int*)(mem != NULL ?
                                  mem->AllocBuf(mem->devID, blockNum * gridNum * sizeof(int)) :
                                  XMemAlloc(s->devID, blockNum * gridNum * sizeof(int)));

        _MakeMergeBlockIndex(blockIndex, blockNum, blockNumInMerge, splitSizeInGrid, gridSize, gridNum, s->devID);

        _CopyBlocksOnSite(s->data, realBlockSize, blockNum * gridNum, dataTMP, blockIndex, s->devID);

        if (mem != NULL)
            mem->ReleaseBuf(mem->devID, blockNum * gridNum * sizeof(int));
        else
            XMemFree(s->devID, blockIndex);

        if (!isOnSameDevice) {
            XMemCopy(t->data, t->devID, dataTMP, s->devID, size);
            if (mem != NULL)
                mem->ReleaseBuf(mem->devID, size);
            else
                XMemFree(s->devID, dataTMP);
        }
    }
}

bool CheckMergeSize(const XTensor * s, const XTensor * t, int whereToMerge, int leadingDim)
{
    if (!(s && t))
        return false;

    if (!(s->dataType == t->dataType))
        return false;

    if (leadingDim < 0)
        leadingDim = 0;
    int order = s->order - 1;
    int * dimSize = new int[order];

    for (int i = 0; i < s->order; i++) {
        if (i < leadingDim)
            dimSize[i] = s->dimSize[i];
        else if (i > leadingDim) {
            if (i != whereToMerge)
                dimSize[i - 1] = s->dimSize[i];
            else
                dimSize[i - 1] = s->dimSize[i] * s->dimSize[leadingDim];
        }
    }

    for (int i = 0; i < order; i++) {
        if (dimSize[i] != t->dimSize[i])
            return false;
    }

    return true;
}


/*
transform a tensor by merging it along with a dimension (return an XTensor structure)
make a new tensor to keep the result and  return it

e.g., (N/3, M, 3) -> (N, M)

>> s - the source tensor
>> whereToMerge - the merging operation is along with which dimension
>> leadingDim - the leading dimension of merging, take (N/3, M, 3) -> (N, M) 
   for example, whereToMerge = 0 (i.e., the dimension for "N/3")
   leadingDim = 2 (i.e., the dimension for "3")
<< return - the transformed tensor by merging along with a dimension
*/
XTensor Merge(const XTensor &s, int whereToMerge, int leadingDim)
{
    CheckNTErrors(leadingDim < whereToMerge, "Invalid leading dimension!");
    
    if (leadingDim < 0)
        leadingDim = 0;
    int order = s.order - 1;
    int * dimSize = new int[order];

    for (int i = 0; i < s.order; i++) {
        if (i < leadingDim) 
            dimSize[i] = s.dimSize[i];
        else if (i > leadingDim) {
            if (i != whereToMerge)
                dimSize[i - 1] = s.dimSize[i];
            else
                dimSize[i - 1] = s.dimSize[i] * s.dimSize[leadingDim];
        }
    }

    float dr = (!s.isSparse) ? 1.0F : s.denseRatio;
    XTensor t(order, dimSize, s.dataType, dr, s.devID, s.mem);
    t.SetTMPFlag();

    /* call _Merge function */
    _Merge(&s, &t, whereToMerge, leadingDim);

    /* tensor connections */
    if (s.enableGrad) {
        XLink::MakeLink(&s, NULL, &t, SHAPE_MERGE);
        XLink::AddParamToHeadInt(&t, whereToMerge);
        XLink::AddParamToHeadInt(&t, leadingDim);
    }

    /* destroy variables */
    delete[] dimSize;

    return t;
}

void Merge(const XTensor &s, XTensor &t, int whereToMerge, int leadingDim)
{
    if (!t.isInit || !CheckMergeSize(&s, &t, whereToMerge, leadingDim)) {
        if (leadingDim < 0)
            leadingDim = 0;
        int order = s.order - 1;
        int * dimSize = new int[order];

        for (int i = 0; i < s.order; i++) {
            if (i < leadingDim)
                dimSize[i] = s.dimSize[i];
            else if (i > leadingDim) {
                if (i != whereToMerge)
                    dimSize[i - 1] = s.dimSize[i];
                else
                    dimSize[i - 1] = s.dimSize[i] * s.dimSize[leadingDim];
            }
        }

        float dr = (!s.isSparse) ? 1.0F : s.denseRatio;
        InitTensorV2(&t, order, dimSize, s.dataType, dr, s.devID, s.mem);

        /* destroy variables */
        delete[] dimSize;
    }

    /* call _Merge function */
    _Merge(&s, &t, whereToMerge, leadingDim);

    if (s.enableGrad) {
        /* tensor connections */
        XLink::MakeLink(&s, NULL, &t, SHAPE_MERGE);
        XLink::AddParamToHeadInt(&t, whereToMerge);
        XLink::AddParamToHeadInt(&t, leadingDim);
    }
}

/*
merge small tensors into a big tensor

>> smalls - the list of the small tensors
>> t - the merged tensor (for return)
>> whereToMerge - the merging operation is along with which dimension
*/
void _Merge(const TensorList * smalls, XTensor * t, int whereToMerge)
{
    whereToMerge = (whereToMerge < 0 ? t->order - 1 : whereToMerge);

    CheckNTErrors((smalls != NULL), "Invalid list!");
    CheckNTErrors((smalls->count > 0), "Empty list!");
    CheckNTErrors((whereToMerge >= 0 && whereToMerge < t->order), "Wrong range of  whereToMerge");

    bool uniform = true;

    int mergeNum = smalls->count;
    XTensor* smallsItem0 = smalls->GetItem(0);
    int itemSize = smallsItem0->unitNum * smallsItem0->unitSize;

    for (int i = 0; i < smalls->count; i++) {
        XTensor* smallsItem = smalls->GetItem(i);
        CheckNTErrors((t->unitNum == smallsItem->unitNum * mergeNum), "Unmatched tensors!");

        if (i > 0) {
            XTensor * preItem = smalls->GetItem(i - 1);
            if (smallsItem->unitNum * smallsItem->unitSize != (char*)smallsItem->data - (char*)preItem->data)
                uniform = false;
        }
    }

    int blockSize = 1;
    int blockNum = 1;
    int gridSize = 1;
    int gridNum = 1;
    int mergedNum = smalls->count;

    XTensor * s0 = smalls->GetItem(0);
    for (int i = 0; i < s0->order; i++) {
        if (i >= whereToMerge)
            blockSize *= s0->dimSize[i];
        else
            blockNum *= s0->dimSize[i];
    }

    CheckNTErrors((s0->unitNum % (blockSize * blockNum) == 0), "Incorrect size!");

    /* a grid has a number of blocks. there might be several grids */
    gridSize = blockNum;
    gridNum = s0->unitNum / (blockSize * blockNum);

    /* merging with fewer data copy operations */
    if (mergedNum * gridNum <= MIN_TENSOR_MERGE_LIST_NUM) {
        int sPitch = blockSize * s0->unitSize;
        int tPtich = blockSize * mergedNum * t->unitSize;
        int mSize = blockSize * t->unitSize;
        int n = blockNum;
        int sStep = 0;
        int tStep = blockSize * t->unitSize;
        for (int g = 0; g < gridNum; g++) {
            char * tData = (char*)t->data + g * blockSize * blockNum * t->unitSize;
            for (int k = 0; k < mergedNum; k++) {
                XTensor * s = smalls->GetItem(k);
                char * sData = (char*)s->data + g * blockSize * blockNum * s->unitSize;
                XMemCopy2D(tData + k * tStep, tPtich, t->devID,
                    sData + k * sStep, sPitch, s->devID,
                    mSize, n);
            }
        }
    }
    /* merging with fewer kernel/api calls??? (i'm not sure about it!! may remove this later) */
    else {
        int* dimSizeTMP = new int[smallsItem0->order + 1];
        for (int i = 0; i < smallsItem0->order; i++)
            dimSizeTMP[i + 1] = -smallsItem0->dimSize[i];
        dimSizeTMP[0] = -mergeNum;

        XMem * mem = smallsItem0->mem;
        XTensor * tensorTMP = new XTensor(smallsItem0->order + 1, dimSizeTMP,
                                          smallsItem0->dataType, smallsItem0->denseRatio,
                                          smallsItem0->devID, mem);
        int size = mergeNum * itemSize;

        void * dataTMP = NULL;
        if (uniform)
            dataTMP = smallsItem0->data;
        else
            dataTMP = mem != NULL ? mem->AllocBuf(mem->devID, size) : XMemAlloc(t->devID, size);

        tensorTMP->data = dataTMP;

        /* copy from source to tmp */
        if (!uniform) {
            for (int i = 0; i < mergeNum; i++) {
                XTensor* smallsItem = smalls->GetItem(i);
                XMemCopy((char*)(tensorTMP->data) + (itemSize * i), tensorTMP->devID, smallsItem->data, smallsItem->devID, itemSize);
            }
        }

        _Merge(tensorTMP, t, whereToMerge + 1);

        delete[] dimSizeTMP;

        tensorTMP->data = NULL;
        delete tensorTMP;

        if ((!uniform) && (mem != NULL))
            mem->ReleaseBuf(mem->devID, size);
        else
            XMemFree(t->devID, dataTMP);
    }
}

/*
merge small tensors into a big tensor (return an XTensor structure)
make a new tensor to keep the result and return it

>> smalls - the list of the small tensors
>> whereToMerge - the merging operation is along with which dimension
<< return - the big tensor merged by small tensors
*/
XTensor Merge(const TensorList &smalls, int whereToMerge)
{
    XTensor * tensor = smalls.GetItem(0);
    int order = tensor->order;
    int * dimSize = new int[order];
    for (int i = 0; i < tensor->order; i++) {
        if (i != whereToMerge)
            dimSize[i] = tensor->dimSize[i];
        else
            dimSize[i] = tensor->dimSize[whereToMerge] * smalls.count;
    }

    float dr = (!tensor->isSparse) ? 1.0F : tensor->denseRatio;
    XTensor big(order, dimSize, tensor->dataType, dr, tensor->devID, tensor->mem);
    big.SetTMPFlag();

    /* call _Merge function */
    _Merge(&smalls, &big, whereToMerge);
    
    /* tensor connections */
    if (tensor->enableGrad) {
        XLink::MakeLink(&smalls, &big, SHAPE_MERGE_LIST);
        XLink::AddParamToHeadInt(&big, whereToMerge);
    }

    /* destroy variables */
    delete[] dimSize;

    return big;
}

/* 
merge two tensors into a big tensor (return an XTensor structure) 
>> smalls - the list of the small tensors
>> whereToMerge - the merging operation is along with which dimension
<< return - the big tensor merged by small tensors
*/
XTensor Merge(const XTensor &smallA, const XTensor &smallB, int whereToMerge)
{
    CheckNTErrors(IsSameShaped(smallA, smallB), 
                 "The two tensors must be of the same size!");

    int order = smallA.order;
    int * dimSize = new int[order];
    for (int i = 0; i < smallA.order; i++) {
        if (i != whereToMerge)
            dimSize[i] = smallA.dimSize[i];
        else
            dimSize[i] = smallA.dimSize[whereToMerge] * 2;
    }

    float dr = (!smallA.isSparse) ? 1.0F : smallA.denseRatio;
    XTensor big(order, dimSize, smallA.dataType, dr, smallA.devID, smallA.mem);
    big.SetTMPFlag();

    TensorList smalls(2);
    smalls.Add((XTensor*)&smallA);
    smalls.Add((XTensor*)&smallB);

    /* call _Merge function */
    _Merge(&smalls, &big, whereToMerge);

    /* tensor connections */
    if (smallA.enableGrad) {
        XLink::MakeLink(&smalls, &big, SHAPE_MERGE_LIST);
        XLink::AddParamToHeadInt(&big, whereToMerge);
    }

    /* destroy variables */
    delete[] dimSize;

    return big;
}

} // namespace nts(NiuTrans.Tensor)
