/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2018, Natural Language Processing Lab, Northeastern University.
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
  * $Created by: Bei Li (libei_neu@outlook.com) 2020-02-03
  */

#ifndef __COMMONMODULE_H__
#define __COMMONMODULE_H__

#include "LayerNorm.h"
#include "CommonModules.h"

using namespace nts;

/* the nmt namespace */
namespace nmt
{

/* the layer normalization module to control pre-norm or post-norm*/
XTensor LN(XTensor& input, LayerNorm& ln, bool prenorm, bool before, bool after);

} /* end of the nmt namespace */

#endif