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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-09-12
 */

#ifndef __DROPOUT_H__
#define __DROPOUT_H__

#include "../XTensor.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/* generate a random bernoulli number */
inline DTYPE RandomBernoulli(DTYPE dropProb, DTYPE value)
{
    return (DTYPE)rand()/(DTYPE)RAND_MAX >= dropProb ? (DTYPE)value : 0;
}

/* dropout function */
void _Dropout(const XTensor * x, XTensor * y, unsigned int seed, DTYPE dropProb, int leadingDim = -1);

/* de/dx */
void _DropoutBackward(const XTensor * y, const XTensor * x, 
                      const XTensor * dedy, XTensor * dedx, 
                      unsigned int seed, DTYPE dropProb, int leadingDim = -1);

/* dropout function */
XTensor Dropout(const XTensor &x, DTYPE dropProb, int leadingDim = -1, int leadingDim2 = -1);
    
/* dropout function without broadcast */
XTensor DropoutWithoutBroadcast(const XTensor &x, DTYPE dropProb);

} // namespace nts(NiuTrans.Tensor)

#endif // __DROPOUT_H__