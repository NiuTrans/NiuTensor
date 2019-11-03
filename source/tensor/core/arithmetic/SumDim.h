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
 * $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-07-29
 * It reached to 39 centigrade around 3:00 pm in Shenyang
 * &Updated by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-12-26
 * Add summation by broadcasting.
 * Four of my master students graduated. Good luck to them for their future work!
 */

#ifndef __SUMDIM_H__
#define __SUMDIM_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)
    
/* tensor summation c = a + b * \beta where the size of b is equal to the n-th dimension of a, 
   i.e., a is summed with b by broadcasting */
void _SumDim(const XTensor * a, const XTensor * b, XTensor * c, int n, DTYPE beta = (DTYPE)1.0);
    
/* tensor summation c = a + b * \beta where the size of b is equal to the n-th dimension of a, 
   i.e., a is summed with b by broadcasting. we keep the result in the input tensor a and return nothing */
void _SumDim(XTensor * a, const XTensor * b, int n, DTYPE beta = (DTYPE)1.0);
    
/* tensor summation c = a + b * \beta where the size of b is equal to the n-th dimension of a, 
   i.e., a is summed with b by broadcasting. We make a new tensor c to keep the result and return it */
XTensor SumDim(const XTensor &a, const XTensor &b, int n, DTYPE beta = (DTYPE)1.0);

/* tensor summation c = a + b * \beta where the size of b is equal to the n-th dimension of a, 
   i.e., a is summed with b by broadcasting */
void SumDim(const XTensor &a, const XTensor &b, XTensor &c, int n, DTYPE beta = (DTYPE)1.0);

/* tensor broadcast summation c = a + b * \beta where some of dimensions of b can be of size 1 */
void _SumBroadcast(const XTensor * a, const XTensor * b, XTensor * c, DTYPE beta = (DTYPE)1.0);

/* tensor broadcast summation c = a + b * \beta where some of dimensions of b can be of size 1.
   we return the resulting tensor here */
XTensor SumBroadcast(const XTensor &a, const XTensor &b, DTYPE beta = (DTYPE)1.0);

/* tensor broadcast summation c = a + b * \beta where some of dimensions of b can be of size 1 */
void SumBroadcast(const XTensor &a, const XTensor &b, XTensor &c, DTYPE beta = (DTYPE)1.0);
    
} // namespace nts(NiuTrans.Tensor)

#endif // __SUMDIM_H__
