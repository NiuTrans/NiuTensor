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
* $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-07-04
*/

#ifndef __SELECT_H__
#define __SELECT_H__

#include "../../XTensor.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/* generate a tensor with selected data c = select(a) */
void _Select(const XTensor * a, XTensor * c, int* index, int dim);

/* generate a tensor with selected data c = select(a) */
void _Select(const XTensor * a, XTensor * c, XTensor* index, int dim);

/* 
generate a tensor with selected data c = select(a) (returna a XTensor structure)
make a new tensor to keep the result and return it
*/
XTensor Select(const XTensor &a, XTensor &index, int dim);

/* 
generate a tensor with selected data in range[low,high] along the given dimension 
c = select(a)
*/
void _SelectRange(const XTensor * a, XTensor * c, int dim, int low, int high);

/* 
generate a tensor with selected data in range[low,high] along the given dimension (return an XTensor structure)
make a new tensor to keep the result and return it
c = select(a)
*/
XTensor SelectRange(const XTensor &a, int dim, int low, int high);

} // namespace nts(NiuTrans.Tensor)

#endif // __SELECT_H__
