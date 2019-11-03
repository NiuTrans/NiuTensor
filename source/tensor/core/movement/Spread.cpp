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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-09-25
 */

#include "Spread.h"
#include "Spread.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
This is core assignment for spread function.

>> sData - the data pointer of the source tensor
>> cData - the data pointer of collection tensor
>> blockNum - number of data blocks
>> blockSizeSrc - size of source data block
>> blockSizeColl - size of source data block
>> stride - stride of a data block
*/
void _Assignment(DTYPE * sData, DTYPE * cData, int blockNum, 
                 int blockSizeSrc, int blockSizeColl, int stride) 
{
    for (int i = 0; i < blockNum; i++) {
        DTYPE * s = sData + blockSizeSrc * i;
        DTYPE * c = cData + blockSizeColl * i;
        for(int j = 0; j < stride; j++)
            s[j] = c[j];
    }
}

/*
spread a collection tensor to source tensor.
This is a inverse operation compared to gather.

>> source - the source tensor whose data would be modified
>> collection - the collection whose data would be spread to source tensor
>> dim - the leading dimension to define "sub-tensors"
         e.g., for a tensor of size (3, 2, 4) and dim = 2, 
         we have 4 sub-tensors of size (3, 2)
>> srcIndex - index of the source sub-tensors
>> indexSize - length of srcIndex (and collIndex)
>> collIndex - index of the gathered sub-tensors
*/
void _Spread(XTensor * source, XTensor * collection, int dim, 
             int * srcIndex, int indexSize, int * collIndex)
{
    int order = source->order;

    CheckNTErrors(source->dataType == DEFAULT_DTYPE, "TODO!");
    CheckNTErrors(dim >= 0 && dim < order, "Illegal dimension!");
    
    for(int i = 0; i < order; i++){
        if(i < dim){
            CheckNTErrors(collection->GetDim(i) == source->GetDim(i), "Illegal dimension!");
        }
        else if(i > dim){
            CheckNTErrors(collection->GetDim(i) == source->GetDim(i), "Illegal dimension!");
        }
        else{
            CheckNTErrors(collection->GetDim(i) == indexSize, "Illegal dimension!");
        }
    }

#ifdef USE_CUDA
    if(source->devID >= 0 && collection->devID >= 0) {
        _CudaSpread(source, collection, dim, srcIndex, indexSize, collIndex);
        return;
    }
#endif

    int blockSizeSrc = 1;
    int blockSizeColl = 1;
    int blockNum = 1;
    int stride = 1;

    for (int i = dim + 1; i < order; i++) {
        stride *= source->GetDim(i);
    }
    
    blockSizeSrc = stride * source->GetDim(dim);
    blockSizeColl = stride * collection->GetDim(dim);
    blockNum = source->unitNum / blockSizeSrc;

    DTYPE * sData = (DTYPE*)source->data;
    DTYPE * cData = (DTYPE*)collection->data;

    for(int i = 0; i < indexSize; i++){
        int src = srcIndex[i];
        int tgt = collIndex[i];
        DTYPE * s = sData + src * stride;
        DTYPE * c = cData + tgt * stride;
        _Assignment(s, c, blockNum, blockSizeSrc, blockSizeColl, stride);
    }
}

/*
This is core assignment for backward computation of gather function.
Care of the operator "+=" instead of "=".

>> sData - the data pointer of the source tensor
>> cData - the data pointer of collection tensor
>> blockNum - number of data blocks
>> blockSizeSrc - size of source data block
>> blockSizeColl - size of source data block
>> stride - stride of a data block
*/
void _AssignmentForGather(DTYPE * sData, DTYPE * cData, int blockNum, 
                          int blockSizeSrc, int blockSizeColl, int stride) 
{
    for (int i = 0; i < blockNum; i++) {
        DTYPE * s = sData + blockSizeSrc * i;
        DTYPE * c = cData + blockSizeColl * i;
        for(int j = 0; j < stride; j++)
            s[j] += c[j];
    }
}

/*
spread a collection tensor to source tensor.
And this is a special spread function for backward computation of CopyIndexed function.

>> s - the source tensor whose data would be modified
>> c - the collection whose data would be spread to source tensor
>> dim - the leading dimension to define "sub-tensors"
         e.g., for a tensor of size (3, 2, 4) and dim = 2, 
         we have 4 sub-tensors of size (3, 2)
>> srcIndex - the tensor to save the index of the source sub-tensors
>> collIndex - the tensor to save the index of the collection sub-tensors
>> copyNum - number of the sub-tensors we copy for each source index, 
             e.g., for srcIndex = [1,4] and copyNum = 2,
             we actually copy the source sub-tensors 1, 2, 4, 5
*/
void _SpreadForCopyIndexed(XTensor * s, XTensor * c, int dim, 
                           XTensor * srcIndex, XTensor * collIndex, 
                           int copyNum)
{
    int order = s->order;
    int indexSize = srcIndex->unitNum;

    CheckNTErrors(indexSize != 0, "NULL index!")
    CheckNTErrors((s && c), "Invalid tensors!");
    CheckNTErrors((srcIndex && collIndex), "Invalid index tensors!");
    CheckNTErrors((s->devID == c->devID || (s->devID < 0 && c->devID < 0)),
                  "the data must be kept on the same device!");
    CheckNTErrors((srcIndex->devID == srcIndex->devID || (s->devID < 0 && c->devID < 0)),
                  "the index must be kept on the same device!");
    CheckNTErrors((s->devID == srcIndex->devID || (s->devID < 0 && c->devID < 0)),
                  "the data and index must be kept on the same device!");
    CheckNTErrors((dim >= 0 && dim < s->order), "A too larget dimension specified!");
    CheckNTErrors((s->unitSize == c->unitSize), "Unmatched tensors!");
    CheckNTErrors((srcIndex->unitNum == collIndex->unitNum), "Unmatched index tensors!");

    CheckNTErrors(s->dataType == DEFAULT_DTYPE, "TODO!");
    CheckNTErrors(dim >= 0 && dim < order, "Illegal dimension!");
    
    for (int i = 0; i < order; i++) {
        if (i != dim) {
            CheckNTErrors(s->GetDim(i) == c->GetDim(i), "Unmatched dimensions");
        }
        else {
            CheckNTErrors(c->GetDim(i) == indexSize * copyNum, "Unmatched dimensions");
        }
    }

#ifdef USE_CUDA
    if(s->devID >= 0 && c->devID >= 0) {
        _CudaSpreadForCopyIndexed(s, c, dim, srcIndex, collIndex, copyNum);
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
    blockSizeTgt = stride * c->GetDim(dim);

    DTYPE * sData = (DTYPE*)s->data;
    DTYPE * cData = (DTYPE*)c->data;
    int * sIndex = (int*)srcIndex->data;
    int * cIndex = (int*)collIndex->data;

    for (int i = 0; i < indexSize; i++) {
        for (int c = 0; c < copyNum; c++) {
            int si = sIndex[i] + c;
            int ti = cIndex[i] + c;

            for (int j = 0; j < blockNum; j++) {
                DTYPE * sd = sData + j * blockSizeSrc + si * stride;
                DTYPE * td = cData + j * blockSizeTgt + ti * stride;
                for (int k = 0; k < stride; k++)
                    *(sd + k) += *(td + k);
            }
        
        }
    }
}

/*
spread a collection tensor to source tensor.
And this is a special spread function for backward computation of gather function.

>> source - the source tensor whose data would be modified
>> collection - the collection whose data would be spread to source tensor
>> index - the tensor to save the index of the collenction tensor
*/
void _SpreadForGather(XTensor * source, XTensor * collection, XTensor * index)
{
    int dim = 0;
    int order = source->order;

    CheckNTErrors(source->dataType == DEFAULT_DTYPE, "TODO!");
    CheckNTErrors(collection->GetDim(-1) == source->GetDim(-1), "Illegal dimension!");
    CheckNTErrors(collection->unitNum/collection->GetDim(-1) == index->unitNum, 
                 "Illegal dimension!");
    
    //for(int i = 0; i < order; i++){
    //    if(i == dim){
    //        CheckNTErrors(collection->GetDim(i) == index->unitNum, "Illegal dimension!");
    //    }
    //    else {
    //        CheckNTErrors(collection->GetDim(i) == source->GetDim(i), "Illegal dimension!");
    //    }
    //}

#ifdef USE_CUDA
    if(source->devID >= 0 && collection->devID >= 0) {
        _CudaSpreadForGather(source, collection, index);
        return;
    }
#endif
    
    int stride = 1;
    int indexSize = 1;

    stride = source->GetDim(-1);
    indexSize = index->unitNum;

    DTYPE * sData = (DTYPE*)source->data;
    DTYPE * cData = (DTYPE*)collection->data;
    int * sIndexData = (int*)index->data;

    for (int i = 0; i < indexSize; i++) {
        int sIndex = sIndexData[i] * stride;
        for (int j = 0; j < stride; j++)
            sData[sIndex + j] += cData[i * stride + j];
    }
}

} // namespace nts(NiuTrans.Tensor)