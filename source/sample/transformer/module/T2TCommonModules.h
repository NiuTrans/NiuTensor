/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2020, Natural Language Processing Lab, Northestern University. 
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

#include "T2TLayerNormal.h"
#include "T2TCommonModules.h"

using namespace nts;

namespace transformer
{

/* the layer normalization module to control pre-norm or post-norm*/
XTensor LayerNorm(XTensor& input, T2TLN& ln, bool prenorm, bool before, bool after);

}

#endif