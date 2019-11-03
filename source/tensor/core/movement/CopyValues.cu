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
* $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-04-24
*/

#include "CopyValues.h"
#include "CopyValues.cuh"
#include "../../XUtility.h"
#include "../../XDevice.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/*
copy a range of elements from a source vector to a target vector
>> s - source matrix
>> t - target matrix
>> stream - the stream for creating the job pipeline
<< return - succeed or not
*/
void _CudaCopyValues(const XTensor * s, XTensor * t, XStream * stream)
{
    CheckNTErrors(s != NULL && t != NULL, "The input tensor and output tensor must be nonempty!");
    CheckNTErrors(s->dataType == t->dataType, "Unmatched data type!");
    CheckNTErrors(s->unitSize == t->unitSize, "Incompatible data types in value copy.");
    CheckNTErrors(s->unitNum == t->unitNum, "The data items are be the same.");
    CheckNTErrors(s->denseRatio <= t->denseRatio, "Incompatible vectors in value copy.");

    /* dense -> dense */
    if (!s->isSparse && !t->isSparse) {
        if (stream == NULL)
            XMemCopy(t->data, t->devID, s->data, s->devID, s->unitSize * s->unitNum);
        else
            XMemCopyAsync(t->data, t->devID, s->data, s->devID, s->unitSize * s->unitNum, stream->stream, stream->devID);
    }
    /* dense -> sparse */
    else if (!s->isSparse && t->isSparse &&
              s->dataType == DEFAULT_DTYPE &&
              t->dataType == DEFAULT_DTYPE)
    {
        ShowNTErrors("TODO!");
    }
    /* sparse -> dense */
    else if (s->isSparse && !t->isSparse &&
             s->dataType == DEFAULT_DTYPE &&
             t->dataType == DEFAULT_DTYPE)
    {
        ShowNTErrors("TODO!");
    }
    /* sparse -> sparse */
    else if (s->isSparse && t->isSparse &&
             s->dataType == DEFAULT_DTYPE &&
             t->dataType == DEFAULT_DTYPE)
    {
        int num = s->unitNumNonZero;
        int size = sizeof(int) + num * (s->unitSize + sizeof(int));

        if (stream == NULL)
            XMemCopy(t->data, t->devID, s->data, s->devID, size);
        else
            XMemCopyAsync(t->data, t->devID, s->data, s->devID, size, stream->stream, stream->devID);

        t->unitNumNonZero = num;
    }
    else {
        ShowNTErrors("TODO!");
    }
}


#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)