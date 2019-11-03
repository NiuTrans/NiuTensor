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

#include "../../XTensor.h"
#include "../../XName.h"
#include "../movement/CopyValues.h"
#include "../shape/IsSameShaped.h"
#include "Reshape.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* 
reshape the tensor 
>> s - the input tensor
>> order - order of the tensor
>> dimSize - the size of each dimension
<< return - the output tensor
*/
XTensor Reshape(XTensor &s, int order, int * dimSize)
{
    XTensor t(&s);
    t.SetTMPFlag();
    _CopyValues(&s, &t);

    /* call Reshape function */
    t.Reshape(order, dimSize);

    /* tensor connections */
    if (s.enableGrad) {
        XLink::MakeLink(&s, NULL, &t, SHAPE_RESHAPE);
    }

    return t;
}

void Reshape(XTensor &s, XTensor &t, int order, int * dimSize)
{
    if (!t.isInit || !IsSameShaped(t, s)) {
        InitTensorV2(&t, &s);
    }

    /* call Reshape function */
    t.Reshape(order, dimSize);

    if (s.enableGrad) {
        /* tensor connections */
        XLink::MakeLink(&s, NULL, &t, SHAPE_RESHAPE);
    }
}

} // namespace nts(NiuTrans.Tensor)
