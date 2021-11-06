/*
* NiuTrans.Tensor - an open-source tensor library
* Copyright (C) 2016-2021
* Natural Language Processing Lab, Northeastern University
* and
* NiuTrans Research
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
* A record that keeps some information in running and training neural networks
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-03-06
* I will climb mountains with my wife and son this afternoon, hahaha :)
*/

#include "XNNRecord.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* constructor */
XNNRecord::XNNRecord()
{
	Clear();
    MUTEX_INIT(mutex);
}

/* de-constructor */
XNNRecord::~XNNRecord()
{
    MUTEX_DELE(mutex);
}

/* clear it */
void XNNRecord::Clear()
{
	lossAll = 0;
    sampleNum = 0;
	predictNum = 0;
	state = XWORKER_UNSTARTED;
}

/* update me with another record */
void XNNRecord::Update(XNNRecord & record)
{
	lossAll += record.lossAll;
    sampleNum += record.sampleNum;
	predictNum += record.predictNum;

}

}
