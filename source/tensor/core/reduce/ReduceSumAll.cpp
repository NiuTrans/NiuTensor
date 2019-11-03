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

#include "ReduceSumAll.h"
#include "ReduceSum.h"
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
<< return - the total summation
*/
DTYPE _ReduceSumAll(const XTensor * source)
{
    int dims[2] = {1, source->unitNum};
    int one = 1;

    XTensor * all = NewTensorBufV2(2, dims, source->dataType, source->denseRatio, source->devID, source->mem);
    XTensor * result = NewTensorBufV2(1, &one, source->dataType, 1.0F, source->devID, source->mem);
    
    _CopyValues(source, all);
    _ReduceSum(all, result, 1);
    
    DTYPE r = result->Get1D(0);
    
    DelTensorBuf(result);
    DelTensorBuf(all);
    
    return r;

    /*int order = source->order;
    DTYPE summation;

    XTensor * big = NewTensor(source);
    _CopyValues(source, big);
    for(int i = order - 1; i >= 0; i--) {
        if(i == 0)
            big->Reshape(1, big->unitNum);

        int leadingDim = big->order - 1;
        int * dimSize;
        dimSize = getDimSize(big, leadingDim);
        XTensor * little = NewTensorV2(big->order - 1, dimSize, source->dataType, source->denseRatio, 
                                     source->devID, source->mem);
        
        _ReduceSum(big, little, leadingDim);

        delete big;
        delete dimSize;

        big = NewTensor(little);
        _CopyValues(little, big);

        delete little;
    }
    summation = big->Get1D(0);
    delete big;

    return summation;*/
}

/*
sum all the items of the tensor
>> source - the inpute tensor
<< return - the total summation   
*/
DTYPE ReduceSumAll(const XTensor & source)
{
    return _ReduceSumAll(&source);
}

} // namespace nts(NiuTrans.Tensor)