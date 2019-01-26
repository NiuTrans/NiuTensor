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

#include "../../XTensor.h"
#include "CopyData2D.h"
#include "../../XUtility.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
copy data blocks by 2d layout
>> s - array of pointers to source data blocks
>> sPitch - source pitch size
>> t - array of pointers to target data blocks
>> tPitch - target pitch size
>> blockNum - number of source blocks
>> mSize - width of each data block to copy
>> n - height of each block
>> myMem - the memory pool
*/
void _CopyData2D(void ** s, int sPitch, void ** t, int tPitch, int blockNum, int mSize, int n, XMem * myMem)
{
    int devID = myMem != NULL ? myMem->devID : -1;

    for (int i = 0; i < blockNum; i++) {
        XMemCopy2D(t[i], tPitch, devID, s[i], sPitch, devID, mSize, n);
    }
}

} // namespace nts(NiuTrans.Tensor)