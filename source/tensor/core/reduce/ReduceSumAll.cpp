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
 * $Created by: LI Yinqqiao (email: li.yin.qiao.2012@hotmail.com) 2020-01-09
 */

#include "ReduceSumAll.h"
#include "ReduceSum.h"
#include "../../XName.h"
#include "../movement/CopyValues.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

int * getDimSize(const XTensor * tensor, int n)
{
    int order = tensor->order;
    int * dimSize = new int[order - 1];

    for (int i = 0; i < order; i++) {
        if(i < n)
            dimSize[i] = tensor->dimSize[i];
        else if(i > n)
            dimSize[i - 1] = tensor->dimSize[i];
    }
    return dimSize;
}

/*
sum all the items of the tensor (It should be optimized!)
>> source - the inpute tensor
<< target - the total summation
*/
void _ReduceSumAll(const XTensor * source, XTensor * target)
{
    CheckNTErrors((source->devID == target->devID || (source->devID < 0 && target->devID < 0)),
                  "This code must be run on the same device!");
    CheckNTErrors((source && target), "Empty input or output tensors!");
    CheckNTErrors((target->order == 0), "Incorrect target tensor sizes!");
    CheckNTErrors((target->unitNum == 1), "Illegal dimension to reduce!");
    CheckNTErrors((source->dataType == target->dataType), "Unmatched data types!");

    int dims[1] = {source->unitNum};

    XTensor * all = NewTensorBufV2(1, dims, source->dataType, source->denseRatio, source->devID, source->mem);

    _CopyValues(source, all);
    _ReduceSum(all, target, 0);

    DelTensorBuf(all);
}

/*
sum all the items of the tensor (It should be optimized!)
>> source - the inpute tensor
<< value - the total summation
*/
void _ReduceSumAll(const XTensor * source, DTYPE * value)
{
    int * dimSize = new int[MAX_TENSOR_DIM_NUM];
    float dr = (!source->isSparse) ? 1.0F : source->denseRatio;
    XTensor * target = NewTensorBufV2(0, dimSize, source->dataType, source->denseRatio, source->devID, source->mem);
    target->SetTMPFlag();

    /* call _ReduceSum function */
    _ReduceSumAll(source, target);
    *value = target->Get0D();

    delete[] dimSize;
    DelTensorBuf(target);
}

/*
sum all the items of the tensor
>> source - the inpute tensor
<< return - the total summation
*/
XTensor ReduceSumAll(const XTensor & source)
{
    int * dimSize = new int[MAX_TENSOR_DIM_NUM];
    float dr = (!source.isSparse) ? 1.0F : source.denseRatio;
    XTensor target(0, dimSize, source.dataType, dr, source.devID, source.mem);
    target.SetTMPFlag();

    /* call _ReduceSum function */
    _ReduceSumAll(&source, &target);

    /* tensor connection */
    if (source.enableGrad) {
        XLink::MakeLink(&source, NULL, &target, REDUCE_REDUCESUMALL);
    }

    /* destroy variables */
    delete[] dimSize;

    return target;
}

/*
sum all the items of the tensor
>> source - the inpute tensor
<< return - the total summation   
*/
DTYPE ReduceSumAllValue(const XTensor & source)
{
    XTensor target;
    target = ReduceSumAll(source);
    return target.Get0D();
}

} // namespace nts(NiuTrans.Tensor)