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


#include "Split.h"
#include "MakeSplitBlockIndex.h"
#include "../../XName.h"
#include "../../XTensor.h"
#include "../../XDevice.h"
#include "../../XUtility.h"
#include "../movement/CopyBlocksOnSite.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
transform a tensor by splitting it, e.g., (N, M) -> (N/3, M, 3)

>> s - the source tensor
>> t - the target tensor (for return)
>> whereToSplit - which dimension of the tensor is to split
>> splitNum - how many splits
*/
void _Split(const XTensor * s, XTensor * t, int whereToSplit, int splitNum)
{
    CheckNTErrors((s && t), "Invalid tensors!");
    CheckNTErrors((s->devID == t->devID || (s->devID < 0 && t->devID < 0)),
                  "the data must be kept on the same device!");

    CheckNTErrors((s->unitNum == t->unitNum && s->unitSize == t->unitSize), "Unmatched tensors!");
    CheckNTErrors((s->order == t->order - 1), "Unmatched tensors!");
    CheckNTErrors((t->dimSize[0] == splitNum), "Incorrect tensor sizes!");

    for (int i = 0; i < s->order; i++) {
        if (i == whereToSplit) {
            CheckNTErrors((s->dimSize[i] == t->dimSize[i + 1] * splitNum),
                          "Unmatched tensor sizes!");
        }
        else {
            CheckNTErrors((s->dimSize[i] == t->dimSize[i + 1]),
                          "Unmatched tensor sizes!");
        }
    }

    /* for the case that we split the last dimension. Actually
    (N, M) and (N, M/3, 3) have the same memory layout */
    if (0 == whereToSplit) {
        XMemCopy(t->data, t->devID, s->data, s->devID, s->unitNum * s->unitSize);
        return;
    }

    int blockSize = 1;
    int blockNum = 1;
    for (int i = 0; i < s->order; i++) {
        if (i == whereToSplit) {
            blockSize *= s->dimSize[i] / splitNum;
            blockNum *= splitNum;
        }
        else if (i > whereToSplit)
            blockSize *= s->dimSize[i];
        else
            blockNum *= s->dimSize[i];
    }

    CheckNTErrors((blockNum % splitNum == 0), "Incorrect split number!");

    if (splitNum <= MIN_TENSOR_SPLIT_NUM) {
        int sPitch = blockSize * splitNum * s->unitSize;
        int tPitch = blockSize * t->unitSize;
        int mSize = blockSize * t->unitSize;
        int n = blockNum / splitNum;
        int sStep = blockSize * s->unitSize;
        int tStep = n * tPitch;
        if(t->devID < 0){
            for (int k = 0; k < splitNum; k++) {
                XMemCopy2D((char*)t->data + k * tStep, tPitch, t->devID,
                           (char*)s->data + k * sStep, sPitch, s->devID,
                            mSize, n);
            }
        }
        else{
#ifdef USE_CUDA
#ifdef STREAMED_MEMCPOPY
            XStream * stream = GDevs.GPUs[t->devID].stream;
            for (int k = 0; k < splitNum; k++) {
                XMemCopy2DAsync((char*)t->data + k * tStep, tPitch, t->devID,
                                (char*)s->data + k * sStep, sPitch, s->devID,
                                 mSize, n, stream);
            }
            stream->StreamSynchronize();
#else
            for (int k = 0; k < splitNum; k++) {
                XMemCopy2D((char*)t->data + k * tStep, tPitch, t->devID,
                           (char*)s->data + k * sStep, sPitch, s->devID,
                            mSize, n);
            }
#endif
#else
            ShowNTErrors("Please specify USE_CUDA and recompile the code!");
#endif
        }
    }
    else {
        XMem * mem = s->mem;
        int size = s->unitNum * s->unitSize;
        bool isOnSameDevice = (s->devID < 0 && t->devID < 0) || (s->devID == t->devID);

        void * dataTMP = t->data;

        if (!isOnSameDevice)
            dataTMP = mem != NULL ? mem->AllocBuf(mem->devID, size) : XMemAlloc(s->devID, size);

        int realBlockSize = blockSize * t->unitSize;
        int blockSplitSize = blockNum / splitNum;

        int * blockIndex = (int*)(mem != NULL ?
                                  mem->AllocBuf(mem->devID, blockNum * sizeof(int)) :
                                  XMemAlloc(s->devID, blockNum * sizeof(int)));

        _MakeSplitBlockIndex(blockIndex, splitNum, blockSplitSize, blockNum, s->devID);

        _CopyBlocksOnSite(s->data, realBlockSize, blockNum, dataTMP, blockIndex, s->devID);

        if (mem != NULL)
            mem->ReleaseBuf(mem->devID, blockNum * sizeof(int));
        else
            XMemFree(s->devID, blockIndex);

        /* copy from tmp to target */
        if (!isOnSameDevice) {
            XMemCopy(t->data, t->devID, dataTMP, s->devID, size);

            if (mem != NULL)
                mem->ReleaseBuf(mem->devID, size);
            else
                XMemFree(s->devID, dataTMP);
        }
    }
}

bool CheckSplitSize(const XTensor * s, const XTensor * t, int whereToSplit, int splitNum)
{
    if (!(s && t))
        return false;

    if (!(s->dataType == t->dataType))
        return false;

    int order = s->order + 1;
    int * dimSize = new int[order];

    dimSize[0] = splitNum;
    for (int i = 0; i < s->order; i++) {
        if (i == whereToSplit)
            dimSize[i + 1] = s->dimSize[i] / splitNum;
        else
            dimSize[i + 1] = s->dimSize[i];
    }

    for (int i = 0; i < order; i++) {
        if (dimSize[i] != t->dimSize[i])
            return false;
    }

    return true;
}

/*
transform a tensor by splitting it, e.g., (N, M) -> (N/3, M, 3) (return an XTensor structure)
make a new tensor to keep the result and return it

>> s - the source tensor
>> whereToSplit - which dimension of the tensor is to split
>> splitNum - how many splits
<< return - teh transformed tensor by splitting it
*/
XTensor Split(const XTensor &s, int whereToSplit, int splitNum)
{
    CheckNTErrors(&s, "Invalid tensors!");
    CheckNTErrors(s.dimSize[whereToSplit] % splitNum == 0, 
                  "The dimension cannot be splitted due to the inproper split number");

    int order = s.order + 1;
    int * dimSize = new int[order];

    dimSize[0] = splitNum;
    for (int i = 0; i < s.order; i++) {
        if (i == whereToSplit)
            dimSize[i+1] = s.dimSize[i] / splitNum;
        else
            dimSize[i+1] = s.dimSize[i];
    }
    
    float dr = (!s.isSparse) ? 1.0F : s.denseRatio;
    XTensor t(order, dimSize, s.dataType, dr, s.devID, s.mem);
    t.SetTMPFlag();

    /* call _Split function */
    _Split(&s, &t, whereToSplit, splitNum);
        
    /* tensor connections */
    if (s.enableGrad) {
        XLink::MakeLink(&s, NULL, &t, SHAPE_SPLIT);
        XLink::AddParamToHeadInt(&t, whereToSplit);
        XLink::AddParamToHeadInt(&t, splitNum);
    }

    /* destroy variables */
    delete[] dimSize;

    return t;
}

void Split(const XTensor &s, XTensor &t, int whereToSplit, int splitNum)
{
    if (!t.isInit || !CheckSplitSize(&s, &t, whereToSplit, splitNum)) {
        int order = s.order + 1;
        int * dimSize = new int[order];

        dimSize[0] = splitNum;
        for (int i = 0; i < s.order; i++) {
            if (i == whereToSplit)
                dimSize[i + 1] = s.dimSize[i] / splitNum;
            else
                dimSize[i + 1] = s.dimSize[i];
        }

        float dr = (!s.isSparse) ? 1.0F : s.denseRatio;
        InitTensorV2(&t, order, dimSize, s.dataType, dr, s.devID, s.mem);

        /* destroy variables */
        delete[] dimSize;
    }

    /* call _Split function */
    _Split(&s, &t, whereToSplit, splitNum);

    if (s.enableGrad) {
        /* tensor connections */
        XLink::MakeLink(&s, NULL, &t, SHAPE_SPLIT);
        XLink::AddParamToHeadInt(&t, whereToSplit);
        XLink::AddParamToHeadInt(&t, splitNum);
    }
}

/*
split a big tensor into small tensors

>> big - the source tensor
>> smalls - the list that keeps the resulting tensors (for return)
   NOTE that all the "small" tensors have already been placed in the list in advance.
>> whereToSplit - which dimension of the tensor is to split
>> splitNum - how many splits
*/
void _Split(const XTensor * big, TensorList * smalls, int whereToSplit, int splitNum)
{
    CheckNTErrors((smalls != NULL), "Invalid list!");
    CheckNTErrors((smalls->count == splitNum), "Unmatched tensors!");
    CheckNTErrors((smalls->count > 0), "Wrong input!");

    bool uniform = true;

    for (int i = 0; i < smalls->count; i++) {
        XTensor* smallsItem = (XTensor*)smalls->GetItem(i);
        CheckNTErrors((big->unitNum == smallsItem->unitNum * splitNum), "Unmatched tensors!");
        if (i > 0) {
            XTensor * preItem = (XTensor*)smalls->GetItem(i - 1);
            if (smallsItem->unitNum * smallsItem->unitSize != (char*)smallsItem->data - (char*)preItem->data)
                uniform = false;
        }
    }

    int blockSize = 1;
    int blockNum = 1;
    for (int i = 0; i < big->order; i++) {
        if (i == whereToSplit) {
            blockSize *= big->dimSize[i] / splitNum;
            blockNum *= splitNum;
        }
        else if (i > whereToSplit)
            blockSize *= big->dimSize[i];
        else
            blockNum *= big->dimSize[i];
    }

    CheckNTErrors((blockNum % splitNum == 0), "Incorrect split number!");

    /* splitting with fewer data copy operations */
    if (splitNum <= MIN_TENSOR_SPLIT_LIST_NUM) {
        XTensor * t0 = (XTensor*)smalls->GetItem(0);
        int sPitch = blockSize * splitNum * big->unitSize;
        int tPitch = blockSize * t0->unitSize;
        int mSize = blockSize * t0->unitSize;
        int n = blockNum / splitNum;
        int sStep = blockSize * big->unitSize;
        int tStep = 0;

        if(big->devID < 0){
            for (int k = 0; k < splitNum; k++) {
                XTensor * t = (XTensor*)smalls->GetItem(k);
                XMemCopy2D((char*)t->data + k * tStep, tPitch, t->devID,
                           (char*)big->data + k * sStep, sPitch, big->devID,
                            mSize, n);
            }
        }
        else{
#ifdef USE_CUDA
#ifdef STREAMED_MEMCPOPY
            XStream * stream = GDevs.GPUs[big->devID].stream;
            for (int k = 0; k < splitNum; k++) {
                XTensor * t = (XTensor*)smalls->GetItem(k);
                XMemCopy2DAsync((char*)t->data + k * tStep, tPitch, t->devID,
                                (char*)big->data + k * sStep, sPitch, big->devID,
                                 mSize, n, stream);
            }
            stream->StreamSynchronize();
#else
            for (int k = 0; k < splitNum; k++) {
                XTensor * t = (XTensor*)smalls->GetItem(k);
                XMemCopy2D((char*)t->data + k * tStep, tPitch, t->devID,
                           (char*)big->data + k * sStep, sPitch, big->devID,
                            mSize, n);
            }
#endif
#else
            ShowNTErrors("Please specify USE_CUDA and recompile the code!");
#endif
        }
    }
    /* splitting with fewer kernel/api calls??? (i'm not sure about it!! may remove this later) */
    else {
        int* dimSizeTMP = new int[big->order + 1];
        for (int i = 0; i < big->order; i++)
            dimSizeTMP[i + 1] = -big->dimSize[i];
        dimSizeTMP[whereToSplit + 1] /= splitNum;
        dimSizeTMP[0] = -splitNum;

        XMem * mem = big->mem;
        XTensor* tensorTMP = new XTensor(big->order + 1, dimSizeTMP, big->dataType, big->denseRatio, big->devID, mem);
        int size = big->unitNum * big->unitSize;
        void * dataTMP = NULL;

        if (uniform) {
            XTensor* first = (XTensor*)smalls->GetItem(0);
            dataTMP = first->data;
        }
        else {
            dataTMP = mem != NULL ? mem->AllocBuf(mem->devID, size) : XMemAlloc(big->devID, size);
        }

        tensorTMP->data = dataTMP;

        _Split(big, tensorTMP, whereToSplit, splitNum);

        /* copy from tmp to target */
        if (!uniform) {
            int splitSize = big->unitNum * big->unitSize / splitNum;
            for (int i = 0; i < splitNum; i++) {
                XTensor* smallsItem = (XTensor*)smalls->GetItem(i);
                XMemCopy(smallsItem->data, smallsItem->devID, (char*)(tensorTMP->data) + (splitSize * i), tensorTMP->devID, splitSize);
            }
        }

        delete[] dimSizeTMP;

        tensorTMP->data = NULL;
        delete tensorTMP;

        if ((!uniform) && (mem != NULL))
            mem->ReleaseBuf(mem->devID, size);
        else
            XMemFree(big->devID, dataTMP);
    }
}

/*
split a big tensor into small tensors

>> big - the source tensor
>> smalls - the list that keeps the resulting tensors (for return)
   NOTE that all the "small" tensors have already been placed in the list in advance.
>> whereToSplit - which dimension of the tensor is to split
>> splitNum - how many splits
*/
void Split(const XTensor &big, TensorList &smalls, int whereToSplit, int splitNum)
{
    CheckNTErrors(big.GetDim(whereToSplit) % splitNum == 0, "Wrong splitNum!");

    /* call _Split function */
    _Split(&big, &smalls, whereToSplit, splitNum);
            
    /* tensor connections */
    for(int i = 0; i < smalls.count; i++){
        XTensor * s = (XTensor*)smalls.Get(i);

        if (s->enableGrad) {
            XLink::MakeLink(&big, NULL, s, SHAPE_SPLIT_LIST);
            XLink::AddParamToHeadInt(s, whereToSplit);

            /* it is tricky here that we keep the id of each
               block, rather than the total number of the splits */
            XLink::AddParamToHeadInt(s, i);
        }
    }
}

} // namespace nts(NiuTrans.Tensor)
