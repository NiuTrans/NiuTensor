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
* $Created by: Jiang Yufan (email: jiangyufan2018@outlook.com) 2019-03-20
*/

#ifndef __DROPOUTWITHINDEX_H__
#define __DROPOUTWITHINDEX_H__

#include "../XTensor.h"

namespace nts {

void _DropoutWithIndex(const XTensor * x, XTensor * maskIndex, XTensor * c);

XTensor DropoutWithIndex(const XTensor &x, XTensor &mask, DTYPE scale);

} // namespace nts(NiuTrans.Tensor)

#endif // !__DROPOUTWITHINDEX_H__

