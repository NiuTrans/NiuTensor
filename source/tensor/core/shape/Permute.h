/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2018, Natural Language Processing Lab, Northestern University.
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
 * $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-07-05
 */

#ifndef __PERMUTE_H__
#define __PERMUTE_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#define permute _Permute_

/* 
generate the tensor with permuted dimensions.
b = permuted(a) 
*/
void _Permute(XTensor * a, XTensor * b, int * dimPermute);
    
/* 
permute the tensor dimensions (do it on site).
keep the result in the input tensor and return nothing.
a = permuted(a) 
*/
void _PermuteMe(XTensor * a, int * dimPermute);

/*
permute the tensor dimensions (do it on site).
keep the result in the input tensor and return nothing.
a = permuted(a)
*/
void PermuteMe(XTensor  &a, int * dimPermute);

/* 
make a tensor with permuted dimensions (return an XTensor structure).
make a new tensor to keep the result and return it.
b = permuted(a)
*/
XTensor Permute(XTensor &a, int * dimPermute);

    
} // namespace nts(NiuTrans.Tensor)

#endif // __PERMUTE_H__

