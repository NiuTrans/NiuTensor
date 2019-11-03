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
* $Created by: Lin Ye (email: linye2015@outlook.com) 2018-08-03
*/

#include "../../XTensor.h"
#include "../../XName.h"
#include "../shape/IsSameShaped.h"
#include "Clip.h"
#include "Clip.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
set every entry to its clip value
>> a - input tensor we are processing
>> b - output tensor we are processing
>> lower - the lower border
>> upper - the upper border
*/
void _Clip(const XTensor * a, XTensor * b, DTYPE lower, DTYPE upper)
{
#ifdef USE_CUDA
    /* run it on GPUs */
    if (a->devID >= 0) {
        _CudaClip(a, b, lower, upper);
        return;
    }
#endif

    CheckNTErrors((_IsSameShaped(a, b)), "Input tensors should have the same type!");
    CheckNTErrors((a->dataType == DEFAULT_DTYPE), "TODO!");

    DTYPE * d = (DTYPE*)a->data;
    DTYPE * db = (DTYPE*)b->data;
    for (int i = 0; i < a->unitNum; i++) {
        if (d[i] > upper)
            db[i] = upper;
        else if (d[i] < lower)
            db[i] = lower;
        else
            db[i] = d[i];
    }
}

/*
set every entry to its clip value (do it on site)
keep the result in the input tensor a and return nothing
>> a - the tensor we are processing
>> lower - the lower border
>> upper - the upper border
*/
void _ClipMe(XTensor * a, DTYPE lower, DTYPE upper)
{
	_Clip(a, a, lower, upper);
}

/*
set every entry to its clip value (do it on site)
keep the result in the input tensor a and return nothing
>> a - the tensor we are processing
>> lower - the lower border
>> upper - the upper border
*/
void ClipMe(XTensor& a, DTYPE lower, DTYPE upper)
{
    _Clip(&a, &a, lower, upper);
}

/*
set every entry to its clip value (return an XTensor structure)
make a new tensor to keep the result and return it
>> a - input tensor we are processing
>> lower - the lower border
>> upper - the upper border
<< return - the clip value of the input tensor
*/
XTensor Clip(const XTensor & a, DTYPE lower, DTYPE upper)
{
	XTensor b(&a);
	b.SetTMPFlag();

	/* call _Clip function */
	_Clip(&a, &b, lower, upper);

	/* tensor connections */
	if (a.enableGrad) {
	    XLink::MakeLink(&a, NULL, &b, MATH_CLIP);
	    XLink::AddParamToHead(&b, lower);
	    XLink::AddParamToHead(&b, upper);
	}

	return b;
}

void Clip(const XTensor & a, XTensor & b, DTYPE lower, DTYPE upper)
{
    if (!b.isInit || !IsSameShaped(a, b)) {
        InitTensorV2(&b, &a);
    }

    /* call _Clip function */
    _Clip(&a, &b, lower, upper);

    /* tensor connections */
    if (a.enableGrad) {
        XLink::MakeLink(&a, NULL, &b, MATH_CLIP);
        XLink::AddParamToHead(&b, lower);
        XLink::AddParamToHead(&b, upper);
    }
}

/*
backward computation

dE/dx = dE/dy * dy/dx

hard tanh: y =  upper    if x > upper
x    if lower <= x <= upper
lower    if x< lower

and dy/dx =  1    if lower <= x <= upper
0    otherwise

>> gold - gold standard to measure error (or loss)
>> y - output of the function
>> x - input of the function
>> dedy - dE/dy
>> dedx - dE/dx
>> lossName - type of loss function, e.g., cross entropy
*/
void _ClipBackward(XTensor * y, XTensor * x, XTensor * dedy, XTensor * dedx, DTYPE lower, DTYPE upper) 
{
    
#ifdef USE_CUDA
    if (x->devID >= 0) {
        _CudaClipBackward(y, x, dedy, dedx, lower, upper);
        return;
}
#endif

    if (x->dataType == DEFAULT_DTYPE && y->dataType == DEFAULT_DTYPE) {
        DTYPE * dedyp = (DTYPE*)dedy->data;
        DTYPE * dedxp = (DTYPE*)dedx->data;
        DTYPE * ip = (DTYPE*)x->data;
        int size = y->unitNum;

        /* dE/dx = dE/dy * dy/dx */
        for (int i = 0; i < size; i++) {
            DTYPE s = ip[i];
            if (s > upper || s < lower)
                dedxp[i] = 0;
            else
                dedxp[i] = dedyp[i];
        }
    }
    else
        ShowNTErrors("TODO!");
}

} // namespace nts(NiuTrans.Tensor)