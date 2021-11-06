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
 * $Created by: Bei Li (libei_neu@outlook.com) 2020-02-05
 * This file includes some common modules of the Transformer model
 */

#include "CommonModules.h"
#include "../../../tensor/core/CHeader.h"
#include "../../../tensor/function/FHeader.h"

/* the nmt namespace */
namespace nmt
{

/* 
flexible layer normalization for the Transformer 
>> input - input tensor
>> ln - the layernorm network
>> prenorm - whether we use prenorm or not
>> before - whether we use layernorm before attention/ffn
>> after - whether we use layernorm after attention/ffn
*/
XTensor LN(XTensor& input, LayerNorm& ln, bool prenorm, bool before, bool after)
{
    if (after ^ prenorm)
        return ln.Run(input);
    else
        return input;
}

} /* end of the nmt namespace */