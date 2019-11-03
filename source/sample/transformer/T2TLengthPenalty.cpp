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

#include "../../tensor/core/CHeader.h"
#include "T2TLengthPenalty.h"

using namespace nts;

namespace transformer
{

/* 
GNMT-like length penalty: pl = ((5 + n)/(5 + 1))^\alpha 
where n = length of the sequence 
>> length - length of the sequence (for each entry)
>> alpha - the parameter controls the length preference
<< return - length penaltyof the sequence (for each entry)
*/
XTensor T2TLengthPenalizer::GNMT(const XTensor & length, float alpha)
{
    XTensor base;
    XTensor lp;

    //base = ScaleAndShift(ScaleAndShift(length, 0, 5.0F), 1.0F/(5 + 1));
    base = (length + 5)/(1 + 5);

    lp = Power(base, alpha);
    
    return lp;
}

}
