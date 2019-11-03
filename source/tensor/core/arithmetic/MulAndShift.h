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
* $Created by: JIANG Yufan (email: jiangyufan2018@outlook.com) 2019-02-27
*/

#ifndef __MULANDSHIFT_H__
#define __MULANDSHIFT_H__

#include "../../XTensor.h"
#include "../CHeader.h"

namespace nts { // namespace nts(NiuTrans.Tensor)


XTensor MulAndShift(const XTensor &x, const XTensor &w, const XTensor &b,
                    DTYPE alpha = (DTYPE)1.0, XPRunner * parallelRunner = NULL);

XTensor MulAndShift(const XTensor &x, MATRIX_TRANS_TYPE transposedA, 
                    const XTensor &w, MATRIX_TRANS_TYPE transposedB, 
                    const XTensor &b, DTYPE alpha = (DTYPE)1.0, XPRunner * parallelRunner = NULL);

} // namespace nts(NiuTrans.Tensor)

#endif // __OPERATION_H__
