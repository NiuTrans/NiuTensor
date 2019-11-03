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

#include "CopyIndexed.h"
#include "CopyIndexed.cuh"
#include "CopyBlocks.h"
#include "Gather.h"
#include "../../XName.h"
#include "../utilities/SetAscendingOrder.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
copy indexed sub-tensors

>> s - the source tensor
>> t - the target tensor
>> dim - the leading dimension to define "sub-tensors"
         e.g., for a tensor of size (4, 2, 3) and dim = 0, 
         we have 4 sub-tensors of size (2, 3)
>> srcIndex - index of the source sub-tensors
>> indexSize - length of srcIndex (and tgtIndex)
>> tgtIndex - index of the target sub-tensors
>> copyNum - number of the sub-tensors we copy for each source index, 
             e.g., for srcIndex = [0,1] and copyNum = 2,
             we actually copy the source sub-tensors 0, 1, 1 and 2
*/
void _CopyIndexed(const XTensor * s, XTensor * t, int dim, 
                  int * srcIndex, int indexSize, int * tgtIndex, 
                  int copyNum)
{
    CheckNTErrors(s && t, "Invalid tensors!");
    CheckNTErrors(s->devID == t->devID || (s->devID < 0 && t->devID < 0),
                  "the data must be kept on the same device!");
    CheckNTErrors(dim < s->order && dim < t->order, "A too larget dimension specified!");
    CheckNTErrors(s->unitSize == t->unitSize, "Unmatched tensors!");

    int blockSizeSrc = 1;
    int blockSizeTgt = 1;
    int blockNumSrc = 1;
    int blockNumTgt = 1;
    int leadDimSizeSrc = s->dimSize[dim];
    int leadDimSizeTgt = t->dimSize[dim];
    int indexOffsetNum = 1;

    for (int i = dim + 1; i < s->order; i++) {
        blockSizeSrc *= s->dimSize[i];
    }
    for (int i = dim + 1; i < t->order; i++) {
        blockSizeTgt *= t->dimSize[i];
    }
    for (int i = 0; i <= dim; i++)
    {
        blockNumSrc *= s->dimSize[i];
        blockNumTgt *= t->dimSize[i];
    }

    CheckNTErrors(blockSizeSrc == blockSizeTgt, "Unmatched tensors!");
    indexOffsetNum = blockNumSrc / s->dimSize[dim];

    int realIndexSize = indexOffsetNum * indexSize * copyNum;
    int * realSrcIndex = new int[realIndexSize];
    int * realTgtIndex = new int[realIndexSize];
    for (int i = 0; i < indexOffsetNum; i++) {
        int base = i * indexSize * copyNum;
        int baseSrc = i * leadDimSizeSrc;
        int baseTgt = i * leadDimSizeTgt;
        for (int j = 0; j < indexSize; j++) {
            int offset = base + j * copyNum;
            int * rsi = realSrcIndex + offset;
            int * rti = realTgtIndex + offset;
            for (int k = 0; k < copyNum; k++) {
                rsi[k] = baseSrc + srcIndex[j] + k;
                rti[k] = baseTgt + tgtIndex[j] + k;
                CheckNTErrors(rsi[k] < s->unitNum, "Wrong index!");
                CheckNTErrors(rti[k] < t->unitNum, "Wrong index!");
            }
        }
    }

    for (int i = 0; i < indexSize; i++) {
        CheckNTErrors(srcIndex[i] < blockNumSrc, "Index is out of scope!");
        CheckNTErrors(tgtIndex[i] < blockNumTgt, "Index is out of scope!");
    }

    _CopyBlocks(s->data, blockSizeSrc * s->unitSize, realSrcIndex, realIndexSize, t->data, realTgtIndex, s->mem, s->devID);

    delete[] realSrcIndex;
    delete[] realTgtIndex;
}

/*
copy selected sub-tensors where indeces are kept in tensors

>> s - the source tensor
>> t - the target tensor
>> dim - the leading dimension to define "sub-tensors"
         e.g., for a tensor of size (4, 2, 3) and dim = 0, 
         we have 4 sub-tensors of size (2, 3)
>> srcIndex - the tensor to save the index of the source sub-tensors
>> tgtIndex - the tensor to save the index of the target sub-tensors
>> copyNum - number of the sub-tensors we copy for each source index, 
             e.g., for srcIndex = [0,1] and copyNum = 2,
             we actually copy the source sub-tensors 0, 1, 1 and 2
*/
void _CopyIndexed(const XTensor * s, XTensor * t, int dim, 
                  const XTensor * srcIndex, const XTensor * tgtIndex, 
                  int copyNum)
{
    int order = s->order;
    int indexSize = srcIndex->unitNum;

    CheckNTErrors(indexSize != 0, "NULL index!")
    CheckNTErrors(s && t, "Invalid tensors!");
    CheckNTErrors(srcIndex && tgtIndex, "Invalid index tensors!");
    CheckNTErrors(s->devID == t->devID || (s->devID < 0 && t->devID < 0),
                  "the data must be kept on the same device!");
    CheckNTErrors(srcIndex->devID == srcIndex->devID || (s->devID < 0 && t->devID < 0),
                  "the index must be kept on the same device!");
    CheckNTErrors(s->devID == srcIndex->devID || (s->devID < 0 && t->devID < 0),
                  "the data and index must be kept on the same device!");
    CheckNTErrors(dim >= 0 && dim < order, "A too larget dimension specified!");
    CheckNTErrors(s->unitSize == t->unitSize, "Unmatched tensors!");
    CheckNTErrors(srcIndex->unitNum == tgtIndex->unitNum, "Unmatched index tensors!");

    for (int i = 0; i < order; i++) {
        if (i != dim) {
            CheckNTErrors(s->GetDim(i) == t->GetDim(i), "Unmatched dimensions");
        }
        else {
            CheckNTErrors(t->GetDim(i) == indexSize * copyNum, "Unmatched dimensions");
        }
    }

#ifdef USE_CUDA
    if (s->devID >= 0 && srcIndex->devID >= 0) {
        _CudaCopyIndexed(s, t, dim, srcIndex, tgtIndex, copyNum);
        return;
    }
#endif

    int blockNum = 1;
    int stride = 1;
    int blockSizeSrc = 1;
    int blockSizeTgt = 1;

    for (int i = 0; i < dim; i++)
        blockNum *= s->GetDim(i);
    
    for (int i = dim + 1; i < order; i++)
        stride *= s->GetDim(i);

    blockSizeSrc = stride * s->GetDim(dim);
    blockSizeTgt = stride * t->GetDim(dim);

    DTYPE * sData = (DTYPE*)s->data;
    DTYPE * tData = (DTYPE*)t->data;
    int * sIndex = (int*)srcIndex->data;
    int * tIndex = (int*)tgtIndex->data;

    for (int i = 0; i < indexSize; i++) {
        for (int c = 0; c < copyNum; c++) {
            int si = sIndex[i] + c;
            int ti = tIndex[i] + c;

            for (int j = 0; j < blockNum; j++) {
                DTYPE * sd = sData + j * blockSizeSrc + si * stride;
                DTYPE * td = tData + j * blockSizeTgt + ti * stride;
                for (int k = 0; k < stride; k++)
                    *(td + k) = *(sd + k);
            }
        
        }
    }
}

/* 
copy selected sub-tensors

>> s - the source tensor
>> t - the target tensor
>> dim - the leading dimension to define "sub-tensors"
         e.g., for a tensor of size (3, 2, 4) and dim = 2, 
         we have 4 sub-tensors of size (3, 2)
>> srcIndex - the tensor to save the index of the source sub-tensors
>> copyNum - number of the sub-tensors we copy for each source index, 
             e.g., for srcIndex = [1,4] and copyNum = 2,
             we actually copy the source sub-tensors 1, 2, 4, 5
*/
void _CopyIndexed(const XTensor * s, XTensor * t, int dim,                   
                  const XTensor * srcIndex, int copyNum)
{
    XTensor * tgtIndex = NewTensor(srcIndex);
    SetAscendingOrder(*tgtIndex, 0);

    _CopyIndexed(s, t, dim, srcIndex, tgtIndex, copyNum);
    delete tgtIndex;
}

/*
copy selected sub-tensors where indeces are kept in tensors (return an XTensor structure)
make a new tensor to keep the result and return it

>> s - the source tensor
>> dim - the leading dimension to define "sub-tensors"
         e.g., for a tensor of size (3, 2, 4) and dim = 2, 
         we have 4 sub-tensors of size (3,2)
>> srcIndex - index of the source sub-tensors
>> indexSize - length of srcIndex (and tgtIndex)
>> tgtIndex - index of the target sub-tensors
>> copyNum - number of the sub-tensors we copy for each source index, 
             e.g., for srcIndex = [1,4] and copyNum = 2,
             we actually copy the source sub-tensors 1, 2, 4, 5
<< return - the result of copying indexed sub-tensors
*/
XTensor CopyIndexed(const XTensor & s, int dim, 
                    const XTensor & srcIndex, const XTensor & tgtIndex,
                    int copyNum)
{
    CheckNTErrors(dim >= 0 && dim < s.order, "A too larget dimension specified!");

    int order = s.order;
    int * dimSize = new int[order];
    int indexSize = srcIndex.unitNum;

    for (int i = 0; i < s.order; i++) {
        if (i == dim)
            dimSize[i] = indexSize * copyNum;
        else
            dimSize[i] = s.dimSize[i];
    }
    
    float dr = (!s.isSparse) ? 1.0F : s.denseRatio;
    XTensor t(order, dimSize, s.dataType, dr, s.devID, s.mem);
    t.SetTMPFlag();

    /* call _CopyIndexed function */
    _CopyIndexed(&s, &t, dim, &srcIndex, &tgtIndex, copyNum);

    TensorList list(3);
    list.Add((XTensor*)&s);
    list.Add((XTensor*)&srcIndex);
    list.Add((XTensor*)&tgtIndex);

    /* tensor connection */
    if (s.enableGrad) {
        XLink::MakeLink(&list, &t, MOVEMENT_COPYINDEXED);
        XLink::AddParamToHeadInt(&t, dim);
        XLink::AddParamToHeadInt(&t, copyNum);
    }

    /* destroy variables */
    delete[] dimSize;

    return t;
}

/*
copy indexed sub-tensors (return a XTensor structure)
make a new tensor to keep the result and return it

>> s - the source tensor
>> dim - the leading dimension to define "sub-tensors"
         e.g., for a tensor of size (3, 2, 4) and dim = 2, 
         we have 4 sub-tensors of size (3,2)
>> srcIndex - index of the source sub-tensors
>> indexSize - length of srcIndex (and tgtIndex)
>> tgtIndex - index of the target sub-tensors
>> copyNum - number of the sub-tensors we copy for each source index, 
             e.g., for srcIndex = [1,4] and copyNum = 2,
             we actually copy the source sub-tensors 1, 2, 4, 5
<< return - the result of copying indexed sub-tensors
*/
XTensor CopyIndexed(const XTensor &s, int dim, int * srcIndex, int indexSize, int * tgtIndex, int copyNum)
{
    CheckNTErrors(dim >= 0 && dim < s.order, "A too larget dimension specified!");

    int order = s.order;
    int * dimSize = new int[order];

    for (int i = 0; i < s.order; i++) {
        if (i == dim)
            dimSize[i] = indexSize * copyNum;
        else
            dimSize[i] = s.dimSize[i];
    }
    
    float dr = (!s.isSparse) ? 1.0F : s.denseRatio;
    XTensor t(order, dimSize, s.dataType, dr, s.devID, s.mem);
    t.SetTMPFlag();

    /* call _CopyIndexed function */
    _CopyIndexed(&s, &t, dim, srcIndex, indexSize, tgtIndex, copyNum);

    /* NOTE: we must allocate a new array to save index,
             because the source index may be released. */
    int * saveSrcIndex = new int[indexSize];
    memcpy(saveSrcIndex, srcIndex, indexSize * sizeof(int));

    int * saveTgtIndex = new int[indexSize];
    memcpy(saveTgtIndex, tgtIndex, indexSize * sizeof(int));

    /* tensor connection */
    if (s.enableGrad) {
        XLink::MakeLink(&s, NULL, &t, MOVEMENT_COPYINDEXED);
        XLink::AddParamToHeadInt(&t, dim);
        XLink::AddParamToHeadPointer(&t, saveSrcIndex);
        XLink::AddParamToHeadInt(&t, indexSize);
        XLink::AddParamToHeadPointer(&t, saveTgtIndex);
        XLink::AddParamToHeadInt(&t, copyNum);
    }

    /* destroy variables */
    delete[] dimSize;

    return t;
}

} // namespace nts(NiuTrans.Tensor)
