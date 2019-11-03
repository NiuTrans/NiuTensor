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

#ifndef __CONCATENATE_H__
#define __CONCATENATE_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
concatenate a list of tensors along a given dimension
Note that this is actually a wrapper that selects 
"ConcatenateSolely" or "Merge" by means of the tensor shapes 
*/
void _Concatenate(const TensorList * smalls, XTensor * big, int dim);

/*
concatenate a list of tensors along a given dimension (return an XTensor structure)
make a new tensor to keep the result and return it
Note that this is actually a wrapper that selects 
"ConcatenateSolely" or "Merge" by means of the tensor shapes 
*/
XTensor Concatenate(const TensorList &smalls, int dim);

void Concatenate(const TensorList & smalls, XTensor & big, int dim);

/* concatenate two tensors along a given dimension */
void _Concatenate(const XTensor * smallA, const XTensor * smallB, XTensor * big, int dim);

/* 
concatenate two tensors along a given dimension (return an XTensor structure)
make a new tensor to keep the result and return it
*/
XTensor Concatenate(const XTensor &smallA, const XTensor &smallB, int dim);

} // namespace nts(NiuTrans.Tensor)

#endif // __CONCATENATE_H__