/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2019, Natural Language Processing Lab, Northestern University. 
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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2019-04-08
 * Start of a new week - I just finished several documents.
 * Writing document is harder than writing code :)
 */

#ifndef __T2TLENGTHPENALTY_H__
#define __T2TLENGTHPENALTY_H__

#include "../../tensor/XTensor.h"

using namespace nts;

namespace transformer
{

/* We intend to penalize short sequences because they have higher score
   in product of a sequence of probability-like terms and have more chances
   to beat others in search. */
class T2TLengthPenalizer
{
public:
    /* GNMT-like length penalty: pl = ((5 + n)/(5 + 1))^\alpha 
       where n = length of the sequence */
    static
    XTensor GNMT(const XTensor & length, float alpha);
};

}

#endif
