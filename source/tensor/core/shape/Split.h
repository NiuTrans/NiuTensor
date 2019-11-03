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

#ifndef __SPLIT_H__
#define __SPLIT_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#define STREAMED_MEMCPOPY

/* 
transform a tensor by splitting it 
e.g., (M, N) -> (M, N/3, 3) 
*/
void _Split(const XTensor * s, XTensor * t, int whereToSplit, int splitNum);

/* 
transform a tensor by splitting it (return an XTensor structure)
make a new tensor to keep the result and return it
e.g., (M, N) -> (M, N/3, 3) 
*/
XTensor Split(const XTensor &s, int whereToSplit, int splitNum);

void Split(const XTensor &s, XTensor &t, int whereToSplit, int splitNum);

/* split a big tensor into small tensors */
void _Split(const XTensor * big, TensorList * smalls, int whereToSplit, int splitNum);

/* 
split a big tensor into small tensors (return a TensorList structure)
make a new list to keep the result and return it
*/
void Split(const XTensor &big, TensorList &smalls, int whereToSplit, int splitNum);

} // namespace nts(NiuTrans.Tensor)

#endif // __SPLIT_H__