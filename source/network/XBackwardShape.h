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
 * backward computation for shaping and data movement
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-18
 */

#include "../tensor/XTensor.h"
#include "../tensor/function/FHeader.h"

#ifndef __XBACKWARDSHAPE_H__
#define __XBACKWARDSHAPE_H__

namespace nts{

/* this class computes the gradient for tensor shaping and movement given a node */
class XShapeGrad
{
public:
    /* compute dE/dx of a node */
    static
    void MakeGrad(XTensor * node, bool isEfficent);

    /* indicates whether the node is for a shaping operation */
    static
    bool IsShapeOP(XTensor * node);

    /* post processing of a node */
    static
    void PostProcessing(XTensor * node, int typeId, bool isEfficent);

private:
    
    /* gradient computation for copying indexed sub-tensors: b = copyindexed(a, srcIndex, indexSize, tgtIndex, copyNum) */
    static
    void GradCopyIndexed(XTensor * node, bool isEfficent);
        
    /* gradient computation for copying indexed sub-tensors: b = gather(a, index) */
    static
    void GradGather(XTensor * node, bool isEfficent);

    /* gradient computation for dropout with index: b = dropoutwithindex(a, index) */
    static
    void GradDropoutWithIndex(XTensor * node, bool isEfficent);

    /* gradient computation for merge: c = merge(a, b, ...) */
    static
    void GradMerge(XTensor * node, bool isEfficent);

    /* gradient computation for merging a list of tensors : c = merge(list(a, b, ...)) */
    static
    void GradMergeList(XTensor * node, bool isEfficent);
    
    /* gradient computation for transposing a tensor : b = transpose(a) */
    static
    void GradTranspose(XTensor * node, bool isEfficent);

    /* gradient computation for reshaping a tensor: c = reshape(a) */
    static
    void GradReshape(XTensor * node, bool isEfficent);

    /* gradient computation for split: c = split(a) */
    static
    void GradSplit(XTensor * node, bool isEfficent);

    /* gradient computation for spliting. we return the list of the splits : list(c_1, ...) = split(a) */
    static
    void GradSplitList(XTensor * node, bool isEfficent);

    /* gradient computation for spliting. we return the list of the splits : list(c_1, ...) = split(a).
       this method is called only when all nodes of spliting have been processed. We do this in a post-processing
       manner because we can fuze multiple memory copy jobs one time. This is good for system speed up. */
    static
    void GradSplitListPost(XTensor * node, bool isEfficent);

    /* gradient computation for unsqueezing a tensor : c = unsqueeze(a) */
    static
    void GradUnsqueeze(XTensor * node, bool isEfficent);

};

}

#endif