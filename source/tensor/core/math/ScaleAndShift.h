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

#ifndef __SCALEANDSHIFT_H__
#define __SCALEANDSHIFT_H__

#include "../../XTensor.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

#define _Linear   _ScaleAndShift   
#define _LinearMe _ScaleAndShiftMe
#define  Linear    ScaleAndShift

/* 
scale and shift all tensor entires 
b = a * scale + shift 
*/
void _ScaleAndShift(const XTensor * a, XTensor * b, DTYPE scale, DTYPE shift = 0);

/*
scale and shift all tensor entires
keep the result in the input tensor a and return nothing
a = a * scale + shift 
*/
void _ScaleAndShiftMe(XTensor * a, DTYPE scale, DTYPE shift = 0);

/*
scale and shift all tensor entires
keep the result in the input tensor a and return nothing
a = a * scale + shift 
*/
void ScaleAndShiftMe(XTensor & a, DTYPE scale, DTYPE shift = 0);

/*
scale and shift all tensor entires
make a new tensor to keep the result and return it
b = a * scale + shift 
*/
XTensor ScaleAndShift(const XTensor &a, DTYPE scale, DTYPE shift = 0);

/* 
scale and shift all tensor entires 
b = a * scale + shift 
*/
void ScaleAndShift(const XTensor &a, XTensor &b, DTYPE scale, DTYPE shift = 0);

} // namespace nts(NiuTrans.Tensor)

#endif // __SCALEANDSHIFT_H__