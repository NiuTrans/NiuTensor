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

#ifndef __COPYINDEXED_H__
#define __COPYINDEXED_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* copy selected sub-tensors */
void _CopyIndexed(const XTensor * s, XTensor * t, int dim, 
                  int * srcIndex, int indexSize, int * tgtIndex,
                  int copyNum = 1);

/* copy selected sub-tensors */
void _CopyIndexed(const XTensor * s, XTensor * t, int dim, 
                  const XTensor * srcIndex, const XTensor * tgtIndex, 
                  int copyNum = 1);

/* copy selected sub-tensors */
void _CopyIndexed(const XTensor * s, XTensor * t, int dim,                   
                  const XTensor * srcIndex, int copyNum = 1);

/*
copy selected sub-tensors where indeces are kept in tensors (return an XTensor structure)
make a new tensor to keep the result and return it
*/
XTensor CopyIndexed(const XTensor & s, int dim, 
                    const XTensor & srcIndex, const XTensor & tgtIndex,
                    int copyNum = 1);

} // namespace nts(NiuTrans.Tensor)

#endif // __COPYINDEXED_H__