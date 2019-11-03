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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-08-15
 */

#ifndef __DIVDIM_H__
#define __DIVDIM_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)
    
/* 
tensor division of two tensors:
c(i) = a/b + \alpha * c
where the size of b is equal to the n-th dimension of a, 
i.e., a is divided with b by broadcasting 
*/
void _DivDim(const XTensor * a, const XTensor * b, XTensor * c, int n, DTYPE alpha = (DTYPE)0.0);

/*
tensor division of two tensors:
c(i) = a/b + \alpha * c
where the size of b is equal to the n-th dimension of a, 
i.e., a is divided with b by broadcasting 
we keep the result in the input tensor a and return nothing
*/
void _DivDim(XTensor * a, const XTensor * b, int n, DTYPE alpha = (DTYPE)0.0);
    

/*
tensor division of two tensors:
c(i) = a/b + \alpha * c
where the size of b is equal to the n-th dimension of a, 
i.e., a is divided with b by broadcasting 
we make a new tensor c to keep the result and return it
*/
XTensor DivDim(const XTensor &a, const XTensor &b, int n, DTYPE alpha = (DTYPE)0.0);

/* 
tensor division of two tensors:
c(i) = a/b + \alpha * c
where the size of b is equal to the n-th dimension of a, 
i.e., a is divided with b by broadcasting 
*/
void DivDim(const XTensor &a, const XTensor &b, XTensor &c, int n, DTYPE alpha = (DTYPE)0.0);
    
} // namespace nts(NiuTrans.Tensor)

#endif // __DIVDIM_H__
