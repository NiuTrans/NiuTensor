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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-09-27
 */

#include "Squeeze.h"
#include "../movement/CopyValues.h"
#include "../shape/IsSameShaped.h"
#include "../../XName.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/*
squeeze the tensor along the specified dimension 
>> source - the input tensor
>> target - the output tensor
>> leadingDim - the dimension that we would squeeze
                if leadingDim = -1, squeeze all dimensions that are 1
                else, squeeze the specified dimension
*/
void _Squeeze(XTensor * source, XTensor * target, int leadingDim)
{
    int order = target->order;

    CheckNTErrors(_IsSameShaped(source, target), 
                 "The source and target tensor must be of the same size!");
    CheckNTErrors(leadingDim >= -1 && leadingDim < order,
                  "Wrong leading dimension");

    _CopyValues(source, target);

    if(leadingDim < 0) {
        int * newDimSize = new int[order];
        int newOrder = 0;
        for(int i = 0; i < order; i++) {
            int dim = source->GetDim(i);
            if(dim > 1) {
                newDimSize[newOrder] = dim;
                newOrder += 1;
            }
        }
        target->Reshape(newOrder, newDimSize);
        delete[] newDimSize;
    }
    else {
        if(source->GetDim(leadingDim) > 1) 
            return;

        int newOrder = order - 1;
        int * newDimSize = new int[newOrder];
        for(int i = 0; i < order; i++)
            if(i < leadingDim)
                newDimSize[i] = source->GetDim(i);
            else if(i > leadingDim)
                newDimSize[i - 1] = source->GetDim(i);

        target->Reshape(newOrder, newDimSize);
        delete[] newDimSize;
    }
}

/*
squeeze the tensor along the specified dimension  (do it on site)
keep the result in the input tensor a and return nothing

>> source - the input tensor
>> leadingDim - the dimension that we would squeeze
                if leadingDim = -1, squeeze all dimensions that are 1
                else, squeeze the specified dimension
*/
void _SqueezeMe(XTensor * source, int leadingDim)
{
    _Squeeze(source, source, leadingDim);
}

/*
squeeze the tensor along the specified dimension  (do it on site)
keep the result in the input tensor a and return nothing

>> source - the input tensor
>> leadingDim - the dimension that we would squeeze
                if leadingDim = -1, squeeze all dimensions that are 1
                else, squeeze the specified dimension
*/
void SqueezeMe(XTensor& source, int leadingDim)
{
    _Squeeze(&source, &source, leadingDim);
}

/*
squeeze the tensor along the specified dimension (return an XTensor structure)
make a new tensor to keep the result and return it

>> source - the input tensor
>> leadingDim - the dimension that we would squeeze
                if leadingDim = -1, squeeze all dimensions that are 1
                else, squeeze the specified dimension
<< return - the output tensor after squeeze operation
*/
XTensor Squeeze(XTensor & source, int leadingDim)
{
    XTensor target(&source);
    target.SetTMPFlag();

    /* call _Squeeze function */
    _Squeeze(&source, &target, leadingDim);

    /* tensor connections */
    if (source.enableGrad) {
        XLink::MakeLink(&source, NULL, &target, SHAPE_SQUEEZE);
    }

    return target;
}

void Squeeze(XTensor & source, XTensor & target, int leadingDim)
{
    if (!target.isInit || !IsSameShaped(source, target)) {
        InitTensorV2(&target, &source);
    }

    /* call _Squeeze function */
    _Squeeze(&source, &target, leadingDim);

    if (source.enableGrad) {
        /* tensor connections */
        XLink::MakeLink(&source, NULL, &target, SHAPE_SQUEEZE);
    }
}

} // namespace nts(NiuTrans.Tensor)