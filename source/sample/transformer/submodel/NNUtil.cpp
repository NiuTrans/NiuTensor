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
 * $Created by: HU Chi (huchinlp@foxmail.com) 2020-03-21
 */

#include "NNUtil.h"

/* the nmt namespace */
namespace nmt
{

/* 
a wrapper for the gather function 
>> src - the input tensor
>> index - the index tensor
<< res - the output tensor
*/
XTensor AutoGather(XTensor& src, XTensor& index)
{

    if (src.order == 2)
        return Gather(src, index);
    else {
        CheckNTErrors(src.order == 3, "the source must be 3d");

        int order = src.order;
        int dimSize[MAX_TENSOR_DIM_NUM];
        for (int i = 0; i < src.order; i++) {
            dimSize[i] = src.dimSize[i];
        }

        src.Reshape(src.dimSize[0], src.dimSize[1] * src.dimSize[2]);
        XTensor res = Gather(src, index);

        src.Reshape(order, dimSize);

        dimSize[0] = index.dimSize[0];
        dimSize[1] = res.unitNum / (dimSize[0] * dimSize[2]);

        res.Reshape(order, dimSize);
        return res;
    }
}

} /* end of the nmt namespace */