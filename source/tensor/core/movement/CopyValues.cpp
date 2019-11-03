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

#include "../../XName.h"
#include "../../XUtility.h"
#include "CopyValues.h"
#include "CopyValues.cuh"
#include "../getandset/ConvertDataType.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
copy s to t

>> s - source
>> t - target
>> stream - the stream for creating the job pipeline
*/
void _CopyValues(const XTensor * s, XTensor * t, XStream * stream)
{
    if(s->data == NULL && t->data == NULL)
        return;

    CheckNTErrors(s != NULL && t != NULL, "The input tensor and output tensor must be nonempty!");
    CheckNTErrors(s->data != NULL, "Cannot copy an empty data array!");
    CheckNTErrors(t->data != NULL, "Cannot copy to an empty data array!");
    CheckNTErrors(s->unitSize == t->unitSize, "Incompatible data types in value copy.");
    CheckNTErrors(s->unitNum == t->unitNum, "The data items are be the same.");

    if ((s->dataType == X_FLOAT16 && t->dataType == X_FLOAT) ||
        (s->dataType == X_FLOAT && t->dataType == X_FLOAT16)) {
        CheckNTErrors((s->devID < 0 && t->devID < 0) || s->devID == t->devID,
                       "The code must be run on the same device!");
        CheckNTErrors(s->isSparse || t->isSparse, "TODO!");
        ConvertDataType(s->devID, s->data, s->dataType, t->data, t->dataType, s->unitNum);
    }

#ifdef USE_CUDA
    if (s->devID >= 0 || t->devID >= 0) {
        _CudaCopyValues(s, t, stream);
        return;
    }
#endif

    if (!s->isSparse && !t->isSparse) {
        memcpy((char*)t->data, (char*)s->data, s->unitSize * s->unitNum);
    }
    else if (s->isSparse && t->isSparse) {
        int d = s->unitNumNonZero;
        t->Resize(s);
        t->unitNumNonZero = d;
        memcpy((char*)t->data, (char*)s->data, sizeof(int) + d *(sizeof(int) + t->unitSize));
    }
    else {
        ShowNTErrors("TODO!");
    }
}

/*
copy s to t

>> s - source
>> sBeg - begining of the segment 
>> sLen - length of the segment
>> t - target
>> tBeg - beginning of the segment on the target side
>> stream - the stream for creating the job pipeline
*/
void _CopyValues(const XTensor * s, const int sBeg, const int sLen, XTensor * t, const int tBeg, XStream * stream)
{
    if(s->data == NULL && t->data == NULL)
        return;

    CheckNTErrors(s != NULL && t != NULL, "The input tensor and output tensor must be nonempty!");
    CheckNTErrors(s->data != NULL, "Cannot copy an empty data array!");
    CheckNTErrors(t->data != NULL, "Cannot copy to an empty data array!");
    CheckNTErrors(s->unitSize == t->unitSize, "Incompatible data types in value copy.");
    CheckNTErrors(sBeg >= 0 && sBeg + sLen <= s->unitNum, "Wrong segment of the source data array");
    CheckNTErrors(tBeg >= 0 && tBeg + sLen <= t->unitNum, "Wrong segment of the target data array");

    if (!s->isSparse && !t->isSparse) {
        XMemCopy((char*)t->data + tBeg * t->unitSize, t->devID,
                 (char*)s->data + sBeg * s->unitSize, s->devID,
                  s->unitSize * sLen);
    }
    else {
        ShowNTErrors("TODO!");
    }
}
    
/*
copy s to t (rename _CopyValues)
 >> s - source
 >> t - target
 >> stream - the stream for creating the job pipeline
*/
void CopyValues(const XTensor &s, XTensor &t, XStream * stream)
{
    _CopyValues(&s, &t, stream);
}

/*
copy s to t (return an XTensor structure)
make a new tensor to keep the result and return it

>> s - source
>> stream - the stream for creating the job pipeline
<< return - the copyed tensor t
*/
XTensor CopyValues(const XTensor &s, XStream * stream)
{
    XTensor t(&s);
    t.SetTMPFlag();

    /* call _CopyValues function */
    _CopyValues(&s, &t, stream);
        
    /* tensor connection */
    if (s.enableGrad) {
        XLink::MakeLink(&s, NULL, &t, MOVEMENT_COPYVALUES);
    }

    return t;
}

} // namespace nts(NiuTrans.Tensor)
