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
* $Created by: Jiang Yufan (email: jiangyufan2018@outlook.com) 2019-03-20
*/

#include "DropoutWithIndex.h"
#include "DropoutWithIndex.cuh"
#include "../core/CHeader.h"
#include "../XName.h"
#include "Identity.h"

namespace nts {

/*
This is a special implementation of "dropout" to reduce memory with maskIndex.

>> x - input tensor
>> maskIndex - mask index tensor
>> c - output tensor
*/
void _DropoutWithIndex(const XTensor * x, XTensor * maskIndex, XTensor * c)
{
    CheckNTErrors(maskIndex->order == 1, "Illegal tensor order!");

#ifdef USE_CUDA
    if (maskIndex->devID >= 0 || x->devID >= 0 || c->devID >= 0) {
        _CudaDropoutWithIndex(x, maskIndex, c);
        return;
    }
#endif

    // TODO!!
    ShowNTErrors("TODO!");
}

/*
This is a special implementation of "dropout" to reduce memory with maskIndex.

>> x - input tensor
>> maskIndex - mask index tensor
>> c - output tensor
>> scale - scale factor
*/
XTensor DropoutWithIndex(const XTensor &x, XTensor &maskIndex, DTYPE scale)
{
    XTensor c;

    int order = x.order;
    int * dimSize = new int[order];

    for (int i = 0; i < order; i++) {
        dimSize[i] = x.dimSize[i];
    }

    InitTensor1DV2(&c, x.unitNum, x.dataType, x.devID, x.mem);

    _SetDataFixedFloat(&c, 1.0F);

    _DropoutWithIndex(&x, &maskIndex, &c);

    c.Reshape(order, dimSize);

    _MultiplyMe(&c, &x);

    _ScaleAndShiftMe(&c, scale);

    /* tensor connections */
    if (x.enableGrad) {
        XLink::MakeLink(&x, &maskIndex, &c, MOVEMENT_DROPOUTWITHINDEX);
        XLink::AddParamToHead(&c, scale);
    }

    return c;
}

}// namespace nts(NiuTrans.Tensor)