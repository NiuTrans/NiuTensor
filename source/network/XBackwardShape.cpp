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
void XShapeGrad::MakeGrad(XTensor * node, bool isEfficent)
{
    CheckNTErrors(node->grad != NULL, "No gradient found!");

    XLink &income = node->income;
    int operID = income.typeID;

    if(operID == MOVEMENT_COPYINDEXED)
        GradCopyIndexed(node, isEfficent);
    else if(operID == MOVEMENT_GATHER)
        GradGather(node, isEfficent);
    else if (operID == MOVEMENT_DROPOUTWITHINDEX)
        GradDropoutWithIndex(node, isEfficent);
    else if(operID == SHAPE_MERGE)
        GradMerge(node, isEfficent);
    else if(operID == SHAPE_MERGE_LIST)
        GradMergeList(node, isEfficent);
    else if(operID == SHAPE_RESHAPE)
        GradReshape(node, isEfficent);
    else if(operID == SHAPE_SPLIT)
        GradSplit(node, isEfficent);
    else if(operID == SHAPE_SPLIT_LIST)
        GradSplitList(node, isEfficent);
    else if (operID == SHAPE_TRANSPOSE)
        GradTranspose(node, isEfficent);
    else if(operID == SHAPE_UNSQUEEZE)
        GradUnsqueeze(node, isEfficent);
    else{
        ShowNTErrors("TODO!");
    }
}

/* indicates whether the node is for a math operation */
bool XShapeGrad::IsShapeOP(XTensor * node)
{
    XLink &income = node->income;
    return (income.typeID & DATA_BASE) != 0;
}

/* post processing of a node */
void XShapeGrad::PostProcessing(XTensor * node, int typeID, bool isEfficent)
{
    if(typeID == SHAPE_SPLIT_LIST)
        GradSplitListPost(node, isEfficent);
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
void XShapeGrad::GradCopyIndexed(XTensor * node, bool isEfficent)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum > 0, "Wrong input tensor number for CopyIndexed!");

    int dim = income.GetParamInt(0);
    int copyNum = income.GetParamInt(1);

    XTensor * input = income.tails[0];
    XTensor * srcIndex = income.tails[1];
    XTensor * tgtIndex = income.tails[2];

    XNoder::MakeGrad(input);
    _SpreadForCopyIndexed(input->grad, node->grad, dim, srcIndex, tgtIndex, copyNum);
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
void XShapeGrad::GradGather(XTensor * node, bool isEfficent)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum > 0, "Wrong input tensor number for Gather!");

    XTensor * input = income.tails[0];
    XTensor * index = income.tails[1];
    XNoder::MakeGrad(input);

    _SpreadForGather(input->grad, node->grad, index);

    node->visitMark = NODE_FINISHED;
}

/*
gradient computation for DropoutWithIndex function
*/
void XShapeGrad::GradDropoutWithIndex(XTensor * node, bool isEfficent)
{
    XLink &income = node->income;
    CheckNTErrors(income.tailNum > 0, "Wrong input tensor number for DropoutWithIndex!");

    XTensor * input = income.tails[0];
    XTensor * index = income.tails[1];
    DTYPE scale = income.GetParam(0);
    XNoder::MakeGrad(input);

    //_Identity(node->grad, input->grad);
    _CopyValues(node->grad, input->grad);

    int order = node->grad->order;
    int * dimSize = new int[order];

    for (int i = 0; i < order; i++) {
        dimSize[i] = node->grad->dimSize[i];
    }

    int order1 = 1;
    int * dimSize1 = new int[order1];
    dimSize1[0] = input->grad->unitNum;
    
    input->grad->Reshape(order1, dimSize1);

    _DropoutWithIndex(node->grad, index, input->grad);
    _ScaleAndShiftMe(input->grad, scale);

    input->grad->Reshape(order, dimSize);

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
void XShapeGrad::GradMerge(XTensor * node, bool isEfficent)
{
    XLink &income = node->income;
    XTensor * input = income.tails[0];

    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for MERGE!");
    CheckNTErrors(node->order == input->order - 1, "Wrong tensor orders!");

    int whereToMerge = income.GetParamInt(0);
    int leadDim = income.GetParamInt(1);

    int blockSize = 1;
    int blockNum = 1;
    for(int i = 0; i < input->order; i++){
        if(i < leadDim)
            blockNum *= input->dimSize[i];
    }
    blockSize = input->GetDataSizeInChar() / blockNum;

    XNoder::MakeGrad(input);

    int * dims = new int[input->order];
    memset(dims, 0, sizeof(int) * input->order);
    for(int i = 0, j = 0; i < input->order; i++){
        if(i >= leadDim){
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

    /* we can simply split the gradient tensor 
       if the input is used in merging only */
    if(input->outgo.tailNum == 1){
        for(int i = 0; i < blockNum; i++){
            gradNodeSmall.data = (char*)node->grad->data + i * blockSize;
            gradInputSmall.data = (char*)input->grad->data + i * blockSize;
            _Split(&gradNodeSmall, &gradInputSmall, whereToMerge - leadDim - 1, input->dimSize[leadDim]);
        }
    }

    /* a more complicated case is that the input tensor is used for 
       other operations somewhere else. So we have to do gradient 
       accumulation after spliting, i.e., we need an additional 
       SUM operation */
    else{
        XTensor gradInputSmallBuf(&gradInputSmall);

        for(int i = 0; i < blockNum; i++){
            gradNodeSmall.data = (char*)node->grad->data + i * blockSize;
            gradInputSmall.data = (char*)input->grad->data + i * blockSize;
            _Split(&gradNodeSmall, &gradInputSmallBuf, whereToMerge - leadDim - 1, input->dimSize[leadDim]);
            _Sum(&gradInputSmall, &gradInputSmallBuf, &gradInputSmall);
        }
    }

    gradNodeSmall.data = NULL;
    gradInputSmall.data = NULL;

    delete[] dims;

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
    for(int i = 0; i < income.tailNum; i++){
        XTensor * tail = income.tails[i];
        XNoder::MakeGrad(tail);
        smalls.Add(tail);
        smallsGrad.Add(tail->grad);
        
        if(i > 1){
            CheckNTErrors(_IsSameShaped(last, tail), 
                         "Input tensors must be of the same size!");
        }

        if(tail->outgo.tailNum  > 1)
            mergeOnly = false;

        last = tail;
    }

    int whereToMerge = income.GetParamInt(0);

    /* we can simply split the gradient tensor into the input tensors 
       if the inputs are used in merging only */
    if(mergeOnly)
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
        for(int i = 0; i < smalls.count; i++){
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
void XShapeGrad::GradReshape(XTensor * node, bool isEfficent)
{
    XLink &income = node->income;
    XTensor * input = income.tails[0];
    XNoder::MakeGrad(input);

    CheckNTErrors(income.tailNum == 1, "Wrong input tensor number for MERGE!");

    node->grad->Reshape(input->order, input->dimSize);
    _CopyValues(node->grad, input->grad);
    node->grad->Reshape(node->order, node->dimSize);

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

    XNoder::MakeGrad(input);

    /* we can simply merge the gradient tensor 
       if the input is used in spliting only */
    if(input->outgo.tailNum == 1)
        _Merge(node->grad, input->grad, whereToSplit + 1, 0);

    /* if the tensor is used somewhere else, we need another SUM
       for gradient accumulation */
    else{
        XTensor * inputGradTMP = NewTensorBufV2(input, input->devID, input->mem);

        _Merge(node->grad, inputGradTMP, whereToSplit + 1, 0);
        _Sum(input->grad, inputGradTMP, input->grad);
        
        DelTensorBuf(inputGradTMP);
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

    for(int i = 0; i < outgo.tailNum; i++){
        XTensor * parent = (XTensor*)outgo.tails[i];
        XLink &income = parent->income;
        if(income.typeID == SHAPE_SPLIT_LIST){
            int w = income.GetParamInt(0);
            int splitID = income.GetParamInt(1);
            
            if(whereToSplit < 0)
                whereToSplit = w;
            splitNum++;

            CheckNTErrors(whereToSplit == w, "Wrong dimension for spliting");
            CheckNTErrors(income.tailNum == 1, "Something wrong with outgoing edge!");
            CheckNTErrors(splitNum - 1 == splitID, "Wrong split id!");

            splits.Add(parent->grad);
        }
    }

    XNoder::MakeGrad(node);

    /* we can simply merge the gradient tensor 
       if the node is used in spliting only */
    if(outgo.tailNum == splitNum){
        _Merge(&splits, node->grad, whereToSplit);
    }

    /* if the tensor is used as input to other nodes
       somewhere else, we need another SUM for gradient 
       accumulation */
    else{
        XTensor * nodeGradTMP = NewTensorBufV2(node, node->devID, node->mem);

        _Merge(&splits, nodeGradTMP, whereToSplit + 1);
        _Sum(node->grad, nodeGradTMP, node->grad);
        
        DelTensorBuf(nodeGradTMP);
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
    XTensor * b = NewTensorBufV2(input, input->devID, input->mem);
    XNoder::MakeGrad(input);

    int i = income.GetParamInt(0);
    int j = income.GetParamInt(1);

    CheckNTErrors(input->order > i && i >= 0, "index of dimension is out of scope!");
    CheckNTErrors(input->order > j && j >= 0, "index of dimension is out of scope!");

    _Transpose(output->grad, b, i, j);
    _Sum(input->grad, b, input->grad);
    
    DelTensorBuf(b);

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
    XNoder::MakeGrad(input);

    int dim = income.GetParamInt(0);
    int dSize = income.GetParamInt(1);

    CheckNTErrors(dSize == output->GetDim(dim), "Wrong dim size for UNSQUEEZE!");
    CheckNTErrors(output->unitNum = input->unitNum * dSize, "Wrong tensor size!");
    
    XTensor * g = NewTensorBufV2(input->grad, input->devID, input->mem);
    
    _ReduceSum(output->grad, g, dim);
    _Sum(input->grad, g, input->grad);
    
    DelTensorBuf(g);

    node->visitMark = NODE_FINISHED;
}

}