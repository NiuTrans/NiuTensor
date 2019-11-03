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

#include "../../XTensor.h"
#include "../../XName.h"
#include "../../XUtility.h"
#include "../shape/IsSameShaped.h"
#include "ScaleAndShift.h"
#include "ScaleAndShift.cuh"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/* 
scale and shift all tensor entires

b = a * scale + shift

>> a - the input tensor
>> b - the output tensor
>> scale - the scaler factor
>> shift - the shift factor
*/
void _ScaleAndShift(const XTensor * a, XTensor * b, DTYPE scale, DTYPE shift)
{
#ifdef USE_CUDA
    /* run it on GPUs */
    if(a->devID >= 0){
        _CudaScaleAndShift(a, b, scale, shift);
        return;
    }
#endif

    CheckNTErrors((a->dataType == DEFAULT_DTYPE), "The tensor is not in the default data type!");

    /* sparse tensor */
    if(a->isSparse){
        int num = a->unitNumNonZero;
        char * d = (char*)a->data + sizeof(int);
        char * f = d + (sizeof(int) + sizeof(DTYPE)) * 0 + sizeof(int);
        char * db = (char*)b->data + sizeof(int);
        char * fb = db + (sizeof(int) + sizeof(DTYPE)) * 0 + sizeof(int);
        for(int i = 0; i < num; i++){
            DTYPE * v = (DTYPE*)f;
            DTYPE * vb = (DTYPE*)fb;
            *vb = *v * scale + shift;
            f += sizeof(int) + sizeof(DTYPE);
            fb += sizeof(int) + sizeof(DTYPE);
        }
    }
    /* dense tensor */
    else{
        DTYPE * va = (DTYPE*)a->data;
        DTYPE * vb = (DTYPE*)b->data;
        for(int i = 0; i < b->unitNum; i++){
            *vb = *va * scale + shift;
            va++;
            vb++;
        }
    }
}

/* 
scale and shift all tensor entires (do it on site)
keep the result in the input tensor a and return nothing

a = a * scale + shift

>> a - the input/output tensor
>> scale - the scaler factor
>> shift - the shift factor
*/
void _ScaleAndShiftMe(XTensor * a, DTYPE scale, DTYPE shift)
{
    _ScaleAndShift(a, a, scale, shift);
}

/* 
scale and shift all tensor entires (do it on site)
keep the result in the input tensor a and return nothing

a = a * scale + shift

>> a - the input/output tensor
>> scale - the scaler factor
>> shift - the shift factor
*/
void ScaleAndShiftMe(XTensor& a, DTYPE scale, DTYPE shift)
{
    _ScaleAndShift(&a, &a, scale, shift);
}

/* 
scale and shift all tensor entires (return an XTensor structure)
make a new tensor to keep the result and return it

b = a * scale + shift

>> a - the input tensor
>> scale - the scaler factor
>> shift - the shift factor
<< return - the result of scaling and shifting all tensor entires
*/
XTensor ScaleAndShift(const XTensor &a, DTYPE scale, DTYPE shift)
{
    XTensor b(&a);
    b.SetTMPFlag();
    
    /* call _ScaleAndShift function */
    _ScaleAndShift(&a, &b, scale, shift);
    
    /* tensor connections */
    if (a.enableGrad) {
        XLink::MakeLink(&a, NULL, &b, MATH_SCALEANDSHIFT);
        XLink::AddParamToHead(&b, scale);
        XLink::AddParamToHead(&b, shift);
    }
    
    return b;
}

/* 
scale and shift all tensor entires

b = a * scale + shift

>> a - the input tensor
>> b - the output tensor
>> scale - the scaler factor
>> shift - the shift factor
*/
void ScaleAndShift(const XTensor & a, XTensor & b, DTYPE scale, DTYPE shift)
{
    if (!b.isInit || !IsSameShaped(a, b)) {
        InitTensorV2(&b, &a);
    }

    /* call _ScaleAndShift function */
    _ScaleAndShift(&a, &b, scale, shift);

    if (a.enableGrad) {
        /* tensor connections */
        XLink::MakeLink(&a, NULL, &b, MATH_SCALEANDSHIFT);
        XLink::AddParamToHead(&b, scale);
        XLink::AddParamToHead(&b, shift);
    }
}


} // namespace nts(NiuTrans.Tensor)
