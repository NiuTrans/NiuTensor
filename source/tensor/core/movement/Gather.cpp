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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-09-18
 */

#include "Gather.h"
#include "Gather.cuh"
#include "CopyIndexed.h"
#include "../../XUtility.h"
#include "../../XName.h"
#include "../shape/Reshape.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/*
gather indexed sub-tensors

>> s - the source tensor
>> t - the target tensor
>> srcIndex - index of the source sub-tensors
>> dim - the leading dimension to define "sub-tensors"
e.g., for a tensor of size (3, 2, 4) and dim = 2,
we have 4 sub-tensors of size (3, 2)
*/
void _Gather(const XTensor * s, XTensor * t, XTensor * srcIndex, int dim)
{
    CheckNTErrors((s && t), "Invalid tensors!");
    CheckNTErrors(s->devID == t->devID, "the data must be kept on the same device!");
    CheckNTErrors((t->unitSize == srcIndex->unitSize), "Unmatched tensors!");
    CheckNTErrors((srcIndex->dataType == X_INT), "The index tensor should be INT type!");
    CheckNTErrors((srcIndex->order == s->order), "index's order should be the same with source's");
#ifdef USE_CUDA
    if (s->devID >= 0 && t->devID >= 0) {
        _CudaGather(s, t, srcIndex, dim);
        return;
    }
#endif
    int stride = 1;
    int blockNum = 1;
    for (int i = dim + 1; i < s->order; ++i)
    {
        stride *= s->GetDim(i);
    }
    for (int i = 0; i < dim; ++i)
    {
        blockNum *= s->GetDim(i);
    }
    int indexStrideNum = srcIndex->GetDim(dim);
    int srcStrideNum = stride * s->GetDim(dim);
    int tgtBlockSize = stride * indexStrideNum;

    DTYPE * sData = (DTYPE*)s->data;
    DTYPE * tData = (DTYPE*)t->data;
    int * sIndexData = (int*)srcIndex->data;
    for (int blockIndex = 0; blockIndex < blockNum; ++blockIndex)
    {
        for (int i = 0; i < indexStrideNum; i++) {
            for (int j = 0; j < stride; j++)
            {
                int sIndex = sIndexData[i * stride + blockIndex * indexStrideNum + j] * stride + blockIndex * srcStrideNum + j;
                CheckNTErrors(sIndex < s->unitNum, "Wrong index!");
                int tIndex = i * stride + blockIndex * tgtBlockSize + j;
                tData[tIndex] = sData[sIndex];
            }
        }
    }
}

/*
gather indexed sub-tensors

>> s - the source tensor
>> t - the target tensor
>> srcIndex - the tensor to save the index of the source tensor
*/
void _Gather(const XTensor * s, XTensor * t, XTensor * srcIndex)
{
    CheckNTErrors((s && t), "Invalid tensors!");
    CheckNTErrors(s->devID == t->devID, "the data must be kept on the same device!");
    CheckNTErrors((s->unitSize == t->unitSize), "Unmatched tensors!");

    if (s->devID >= 0) {
#ifdef USE_CUDA
        _CudaGather(s, t, srcIndex);
#else
        ShowNTErrors("Plesae specify USE_CUDA and recompile the code!");
#endif
    }
    else {
        int stride = 1;
        int indexSize = 1;

        stride = s->GetDim(-1);
        indexSize = srcIndex->unitNum;

        DTYPE * sData = (DTYPE*)s->data;
        DTYPE * tData = (DTYPE*)t->data;
        int * sIndexData = (int*)srcIndex->data;

        for (int i = 0; i < indexSize; i++) {
            int sIndex = sIndexData[i] * stride;
            CheckNTErrors(sIndex < s->unitNum && sIndex >= 0, "Wrong index!");
            for (int j = 0; j < stride; j++)
                tData[i * stride + j] = sData[sIndex + j];
        }
    }
}

/*
gather indexed sub-tensors (return an XTensor structure)
make a new tensor to keep the result and return it

>> s - the source tensor(2D)
>> index - the index tensor
<< return - the result of gather indexed sub-tensors
*/
XTensor Gather(XTensor &s, XTensor &index)
{
    int dim = 0;

    CheckNTErrors(s.order == 2, "The order of the input tensor must be 2!");
 
    int order = s.order;
    int * dimSize = new int[order];

    for (int i = 0; i < s.order; i++) {
        if (i == dim)
            dimSize[i] = index.unitNum;
        else
            dimSize[i] = s.dimSize[i];
    }
    
    float dr = (!s.isSparse) ? 1.0F : s.denseRatio;
    XTensor t(order, dimSize, s.dataType, dr, s.devID, s.mem);
    t.SetTMPFlag();

    delete[] dimSize;

    _Gather(&s, &t, &index);

    /* tensor connection */
    if (s.enableGrad)
    {
        XLink::MakeLink(&s, &index, &t, MOVEMENT_GATHER);
    }

    if(index.order > 1) {
        int * dims = new int[index.order + 1];
        memcpy(dims, index.dimSize, index.order * sizeof(int));
        dims[index.order] = t.GetDim(-1);

        t.Reshape(index.order + 1, dims);
        delete[] dims;
    }
    return t;
}

} // namespace nts(NiuTrans.Tensor)