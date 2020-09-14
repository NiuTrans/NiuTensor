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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-19
 * It was chilly when I came into the office this morning ...
 * because i forgot to turn the air-condition off last night :(
 */

#include "XNoder.h"
#include "XBackwardShape.h"
#include "../tensor/XName.h"
#include "../tensor/XUtility.h"
#include "../tensor/core/CHeader.h"
#include "../tensor/core/getandset/SetData.h"

namespace nts{

/* compute dE/dx of a node */
void XShapeGrad::MakeGrad(XTensor * node, bool isEfficient)
{
    if (!isEfficient) {
        CheckNTErrors(node->grad != NULL, "No gradient found!");
    }
    else {
        CheckNTErrors(!node->isGrad || node->grad != NULL, "No gradient found!");
    }

    XLink &income = node->income;
    int operID = income.typeID;

    if (operID == GETANDSET_CONVERTDATATYPE)
        GradConvertDataType(node, isEfficient);
    else if (operID == MOVEMENT_COPYINDEXED)
        GradCopyIndexed(node, isEfficient);
    else if (operID == MOVEMENT_GATHER)
        GradGather(node, isEfficient);
    else if (operID == MOVEMENT_DROPOUTWITHINDEX)
        GradDropoutWithIndex(node, isEfficient);
    else if (operID == SHAPE_MERGE)
        GradMerge(node, isEfficient);
    else if (operID == SHAPE_MERGE_LIST)
        GradMergeList(node, isEfficient);
    else if (operID == SHAPE_RESHAPE)
        GradReshape(node, isEfficient);
    else if (operID == SHAPE_SPLIT)
        GradSplit(node, isEfficient);
    else if (operID == SHAPE_SPLIT_LIST)
        GradSplitList(node, isEfficient);
    else if (operID == SHAPE_TRANSPOSE)
        GradTranspose(node, isEfficient);
    else if (operID == SHAPE_UNSQUEEZE)
        GradUnsqueeze(node, isEfficient);
    else{
        ShowNTErrors("Unsupported backward computation! TODO!");
    }
}

/* indicates whether the node is for a math operation */
bool XShapeGrad::IsShapeOP(XTensor * node)
{
    XLink &income = node->income;
    return (income.typeID & DATA_BASE) != 0;
}

/* post processing of a node */
void XShapeGrad::PostProcessing(XTensor * node, int typeID, bool isEfficient)
{
    if (typeID == SHAPE_SPLIT_LIST)
        GradSplitListPost(node, isEfficient);
}

/*
gradient computation for convertdatatype
for
b = convertdatatype(a)
we have
dE/da = convertdatatype(dE/db)
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XShapeGrad::GradConvertDataType(XTensor* node, bool isEfficient)
{
    XLink& income = node->income;
    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for CopyIndexed!");

    XTensor* a = income.tails[0];

    if (!isEfficient || a->isGrad) {
        XNoder::MakeGrad(a);

        XTensor* tmp = NewTensorBufV2(a, a->devID, a->mem);
        _ConvertDataType(node->grad, tmp);
        _SumMe(a->grad, tmp);

        DelTensorBuf(tmp);
    }
}

/* 
gradient computation for copying indexed sub-tensors
for
b = copyindexed(a) 
we have
dE/da = spreadforcopyindexed(b)
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XShapeGrad::GradCopyIndexed(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum > 0, "Wrong input tensor number for CopyIndexed!");

    int dim = income.GetParamInt(0);
    int copyNum = income.GetParamInt(1);

    XTensor * input = income.tails[0];
    XTensor * srcIndex = income.tails[1];
    XTensor * tgtIndex = income.tails[2];

    if (!isEfficient || input->isGrad) {
        XNoder::MakeGrad(input);

        XTensor * tmp = NewTensorBufV2(input, input->devID, input->mem);
        _SpreadForCopyIndexed(tmp, node->grad, dim, srcIndex, tgtIndex, copyNum);
        _SumMe(input->grad, tmp);

        DelTensorBuf(tmp);
    }
}

/* 
gradient computation for gather function
for
b = gather(a) 
we have
dE/da = spreadforgather(b)
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XShapeGrad::GradGather(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum > 0, "Wrong input tensor number for Gather!");

    XTensor * input = income.tails[0];
    XTensor * index = income.tails[1];
    
    if (!isEfficient || input->isGrad) {
        XNoder::MakeGrad(input);

        XTensor * tmp = NewTensorBufV2(input, input->devID, input->mem);
        tmp->SetZeroAll();
        _SpreadForGather(tmp, node->grad, index);
        _SumMe(input->grad, tmp);

        DelTensorBuf(tmp);
    }

    node->visitMark = NODE_FINISHED;
}

/*
gradient computation for DropoutWithIndex function
*/
void XShapeGrad::GradDropoutWithIndex(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum > 0, "Wrong input tensor number for DropoutWithIndex!");

    XTensor * input = income.tails[0];
    XTensor * index = income.tails[1];
    DTYPE scale = income.GetParam(0);
    
    if (!isEfficient || input->isGrad) {
        XNoder::MakeGrad(input);

        XTensor * tmp = NewTensorBufV2(input, input->devID, input->mem);
        _CopyValues(node->grad, tmp);

        tmp->Reshape(tmp->unitNum);

        _DropoutWithIndex(node->grad, index, tmp);
        _ScaleAndShiftMe(tmp, scale);

        tmp->Reshape(input->order, input->dimSize);
        _SumMe(input->grad, tmp);

        DelTensorBuf(tmp);
    }

    node->visitMark = NODE_FINISHED;
}

/* 
gradient for merge
for 
c = merge(a_0, a_1, ...)
where a_i is the i-th block in a tensor a
we have
dE/da_0 = dE/dc_{split_0}
dE/db_1 = dE/dc_{split_1}
...
i.e.,
dE/da = split(dE/dc)
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XShapeGrad::GradMerge(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    XTensor * input = income.tails[0];

    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for MERGE!");
    CheckNTErrors(node->order == input->order - 1, "Wrong tensor orders!");

    int whereToMerge = income.GetParamInt(0);
    int leadDim = income.GetParamInt(1);

    if (!isEfficient || input->isGrad) {
        XNoder::MakeGrad(input);

        int * dims = new int[input->order];
        memset(dims, 0, sizeof(int) * input->order);
        for (int i = 0, j = 0; i < input->order; i++) {
            if (i >= leadDim) {
                dims[j++] = input->dimSize[i];
            }
        }
        dims[0] = -dims[0];
        XTensor gradInputSmall(input->order - leadDim, dims,
                               input->dataType, input->denseRatio,
                               input->devID, input->mem);

        dims[whereToMerge - leadDim] *= dims[0];
        XTensor gradNodeSmall(node->order - leadDim, dims + leadDim + 1,
                              node->dataType, node->denseRatio,
                              node->devID, node->mem);

        int blockSize = 1;
        int blockNum = 1;
        for (int i = 0; i < input->order; i++) {
            if (i < leadDim)
                blockNum *= input->dimSize[i];
        }
        blockSize = input->GetDataSizeInChar() / blockNum;

        /* we can simply split the gradient tensor
           if the input is used in merging only */
        if (input->outgo.tailNum == 1) {
            for (int i = 0; i < blockNum; i++) {
                gradNodeSmall.data = (char*)node->grad->data + i * blockSize;
                gradInputSmall.data = (char*)input->grad->data + i * blockSize;
                _Split(&gradNodeSmall, &gradInputSmall, whereToMerge - leadDim - 1, input->dimSize[leadDim]);
            }
        }

        /* a more complicated case is that the input tensor is used for
           other operations somewhere else. So we have to do gradient
           accumulation after spliting, i.e., we need an additional
           SUM operation */
        else {
            XTensor gradInputSmallBuf(&gradInputSmall);

            for (int i = 0; i < blockNum; i++) {
                gradNodeSmall.data = (char*)node->grad->data + i * blockSize;
                gradInputSmall.data = (char*)input->grad->data + i * blockSize;
                _Split(&gradNodeSmall, &gradInputSmallBuf, whereToMerge - leadDim - 1, input->dimSize[leadDim]);
                _Sum(&gradInputSmall, &gradInputSmallBuf, &gradInputSmall);
            }
        }

        gradNodeSmall.data = NULL;
        gradInputSmall.data = NULL;

        delete[] dims;
    }

    node->visitMark = NODE_FINISHED;
}

/* 
gradient for merging a list of tensors
for 
c = merge(list(a, b, ...)) 
where a, b ... are of the same size
we have
dE/da = dE/dc_{split_0}
dE/db = dE/dc_{split_1}
i.e.,
list(dE/da, dE/db, ...) = split(dE/dc)
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XShapeGrad::GradMergeList(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum > 0, "Wrong input tensor number for MERGE!");

    XTensor * last = NULL;
    TensorList smalls(income.tailNum);
    TensorList smallsGrad(income.tailNum);
    bool mergeOnly = true;

    for (int i = 0; i < income.tailNum; i++) {
        /* TODO! efficient backpropagate */
        XTensor * tail = income.tails[i];
        XNoder::MakeGrad(tail);
        smalls.Add(tail);
        smallsGrad.Add(tail->grad);
        
        if (i > 1)
            CheckNTErrors(_IsSameShaped(last, tail), "Input tensors must be of the same size!");

        if (tail->outgo.tailNum > 1)
            mergeOnly = false;

        last = tail;
    }

    int whereToMerge = income.GetParamInt(0);

    /* we can simply split the gradient tensor into the input tensors 
       if the inputs are used in merging only */
    if (mergeOnly)
        _Split(node->grad, &smallsGrad, whereToMerge, smalls.count);

    /* a more complicated case is that the input tensors are used for 
       other operations somewhere else. So we have to do gradient 
       accumulation after spliting, i.e., we need an additional 
       SUM operation */
    else{
        int * dims = new int[last->order + 1];
        dims[0] = smalls.count;
        for(int i = 0; i < last->order; i++)
            dims[i + 1] = last->dimSize[i];

        XTensor gradSplit(last->order + 1, dims, 
                          last->dataType, last->denseRatio, 
                          last->devID, last->mem);

        _Split(node->grad, &gradSplit, whereToMerge, smalls.count);

        memcpy(dims, last->dimSize, sizeof(int) * last->order);
        dims[0] = -dims[0];
        XTensor gradSmall(last->order, dims,
                          last->dataType, last->denseRatio, 
                          last->devID, last->mem);

        /* gradient accumulation for each split */
        for (int i = 0; i < smalls.count; i++) {
            XTensor * inputGrad = (XTensor*)smallsGrad.Get(i);
            gradSmall.data = (char*)gradSplit.data + i * last->unitNum * last->unitSize;
            _Sum(inputGrad, &gradSmall, inputGrad);
        }

        gradSmall.data = NULL;
        delete[] dims;
    }

    node->visitMark = NODE_FINISHED;
}

/* 
gradient computation for reshaping a tensor
for
b = reshape(a)
we have
dE/da = reshape(dE/db)
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XShapeGrad::GradReshape(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for RESHAPE!");
    
    XTensor * input = income.tails[0];

    if (!isEfficient || input->isGrad) {
        XNoder::MakeGrad(input);

        node->grad->Reshape(input->order, input->dimSize);
        _CopyValues(node->grad, input->grad);
        node->grad->Reshape(node->order, node->dimSize);
    }

    node->visitMark = NODE_FINISHED;
}

/* 
gradient computation for split: 
for
c = split(a)
we have
dE/da = merge(dE/dc)
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XShapeGrad::GradSplit(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    XTensor * input = income.tails[0];

    int whereToSplit = income.GetParamInt(0);
    int splitNum = income.GetParamInt(1);

    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for SPLIT!");
    CheckNTErrors(node->order == input->order + 1, "Wrong tensor orders!");
    CheckNTErrors(splitNum == node->dimSize[0], "Wrong split number!");

    if (!isEfficient || input->isGrad) {
        XNoder::MakeGrad(input);

        /* we can simply merge the gradient tensor
           if the input is used in spliting only */
        if (input->outgo.tailNum == 1)
            _Merge(node->grad, input->grad, whereToSplit + 1, 0);

        /* if the tensor is used somewhere else, we need another SUM
           for gradient accumulation */
        else {
            XTensor * inputGradTMP = NewTensorBufV2(input, input->devID, input->mem);

            _Merge(node->grad, inputGradTMP, whereToSplit + 1, 0);
            _Sum(input->grad, inputGradTMP, input->grad);

            DelTensorBuf(inputGradTMP);
        }
    }

    node->visitMark = NODE_FINISHED;
}

/* 
gradient computation for spliting 
where we return the list of the splits
for
list(c_1, ...) = split(a) 
we have
dE/da = merge(dE/c_1, ...)
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XShapeGrad::GradSplitList(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    //XTensor * input = income.tails[0];

    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for SPLIT!");
    //CheckNTErrors(node->order == input->order + 1, "Wrong tensor orders!");

    node->visitMark = NODE_DOING;
}

/*
gradient computation for spliting. We return 
the list of the splits : list(c_1, ...) = split(a).
this method is called only when all nodes of spliting 
have been processed. We do this in a post-processing
manner because we can fuze multiple memory copy jobs 
one time. This is good for system speed up. 
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XShapeGrad::GradSplitListPost(XTensor * node, bool isEfficient)
{
    /* we compute the gradient for current node, rather than for
       child node, i.e., we use the outgoing edge here */
    XLink &outgo = node->outgo;
    TensorList splits(outgo.tailNum);
    int whereToSplit = -1;
    int splitNum = 0;

    for (int i = 0; i < outgo.tailNum; i++) {
        XTensor * parent = (XTensor*)outgo.tails[i];
        XLink &income = parent->income;
        if (income.typeID == SHAPE_SPLIT_LIST) {
            int w = income.GetParamInt(0);
            int splitID = income.GetParamInt(1);
            
            if (whereToSplit < 0)
                whereToSplit = w;
            splitNum++;

            CheckNTErrors(whereToSplit == w, "Wrong dimension for spliting");
            CheckNTErrors(income.tailNum == 1, "Something wrong with outgoing edge!");
            CheckNTErrors(splitNum - 1 == splitID, "Wrong split id!");

            splits.Add(parent->grad);
        }
    }

    if (!isEfficient || node->isGrad) {
        XNoder::MakeGrad(node);

        /* we can simply merge the gradient tensor
           if the node is used in spliting only */
        if (outgo.tailNum == splitNum) {
            _Merge(&splits, node->grad, whereToSplit);
        }

        /* if the tensor is used as input to other nodes
           somewhere else, we need another SUM for gradient
           accumulation */
        else {
            XTensor * nodeGradTMP = NewTensorBufV2(node, node->devID, node->mem);

            _Merge(&splits, nodeGradTMP, whereToSplit + 1);
            _Sum(node->grad, nodeGradTMP, node->grad);

            DelTensorBuf(nodeGradTMP);
        }
    }
}

/*
gradient for transposing a tensor
for
c = Transpose(a)
we have
dE/da = Transpose(dE/dc)
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XShapeGrad::GradTranspose(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for TRANSPOSE!");

    XTensor * output = node;
    XTensor * input = income.tails[0];


    if (!isEfficient || input->isGrad) {
        XNoder::MakeGrad(input);

        int i = income.GetParamInt(0);
        int j = income.GetParamInt(1);

        CheckNTErrors(input->order > i && i >= 0, "index of dimension is out of scope!");
        CheckNTErrors(input->order > j && j >= 0, "index of dimension is out of scope!");

        XTensor * tmp = NewTensorBufV2(input, input->devID, input->mem);
        _Transpose(output->grad, tmp, i, j);
        _Sum(input->grad, tmp, input->grad);

        DelTensorBuf(tmp);
    }

    node->visitMark = NODE_FINISHED;
}

/* 
gradient for unsqueezing a tensor
for
c = unsqueeze(a) 
we have
dE/da = reduecesum(dE/dc)
>> node - the node (c) for backward computation
>> isEfficient - indicates whether the computation is in
                 an efficient manner
*/
void XShapeGrad::GradUnsqueeze(XTensor * node, bool isEfficient)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for UNSQUEEZE!");

    XTensor * output = node;
    XTensor * input = income.tails[0];

    int dim = income.GetParamInt(0);
    int dSize = income.GetParamInt(1);

    CheckNTErrors(dSize == output->GetDim(dim), "Wrong dim size for UNSQUEEZE!");
    CheckNTErrors(output->unitNum == input->unitNum * dSize, "Wrong tensor size!");
    
    if (!isEfficient || input->isGrad) {
        XNoder::MakeGrad(input);

        XTensor * tmp = NewTensorBufV2(input->grad, input->devID, input->mem);

        _ReduceSum(output->grad, tmp, dim);
        _Sum(input->grad, tmp, input->grad);

        DelTensorBuf(tmp);
    }

    node->visitMark = NODE_FINISHED;
}

}