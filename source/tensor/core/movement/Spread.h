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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-09-25
 */

#ifndef __SPREAD_H__
#define __SPREAD_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* spread a collection tensor to source tensor */
void _Spread(XTensor * source, XTensor * collection, int dim, 
             int * srcIndex, int indexSize, int * collIndex);

/* spread a collection tensor to source tensor (return an XTensor structure)
   make a new tensor to keep the result and return it */
void Spread(XTensor * source, XTensor * collection, 
            XTensor * srcIndex, XTensor * collIndex,
            int dim);

/* special spread function for backward computation of CopyIndexed function */
void _SpreadForCopyIndexed(XTensor * source, XTensor * collection, int dim, 
                           XTensor * srcIndex, XTensor * collIndex, 
                           int copyNum);

/* special spread function for backward computation of gather function */
void _SpreadForGather(XTensor * source, XTensor * collection, XTensor * index);

} // namespace nts(NiuTrans.Tensor)

#endif // __SPREAD_H__