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
 *
 * An impelementation of the transformer system. See more details 
 * about FNNLM in 
 * "Attention Is All You Need" by Vaswani et al.
 * https://arxiv.org/pdf/1706.03762.pdf
 *
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-31
 * I start writing the code related to NMT - a long time since my last coding 
 * work on MT
 */

#ifndef __TRANSFORMER_H__
#define __TRANSFORMER_H__

#include "../../tensor/XGlobal.h"
#include "../../tensor/XTensor.h"
#include "../../tensor/core/CHeader.h"

namespace transformer
{

/* entrance of the program */
int TransformerMain(int argc, const char ** argv);

}

#endif