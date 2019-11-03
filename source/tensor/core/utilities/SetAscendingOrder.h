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
* $Created by: LI Yinqiao (email: li.yin.qiao.2012@hotmail.com) 2019-10-23
*/

#ifndef __SETASCENDINGORDER_H__
#define __SETASCENDINGORDER_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)
    
/* set the cell to the ascending order along a given dimension */
void SetAscendingOrder(XTensor & tensor, int dim);

} // namespace nts(NiuTrans.Tensor)

#endif // __SETASCENDINGORDER_H__