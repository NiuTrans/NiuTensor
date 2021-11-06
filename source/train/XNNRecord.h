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

#ifndef __XNNRECORD_H__
#define __XNNRECORD_H__

#include "XWorker.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* a record of keeping some stuff during training */
class XNNRecord
{
public:
	/* loss over all samples */
	float lossAll;
    
    /* sample number */
    int sampleNum;

	/* prediction number */
	int predictNum;

	/* state */
	XWORKER_STATE state;

    /* mutex */
    MUTEX_HANDLE mutex;

public:
	/* constructor */
	XNNRecord();

	/* de-constructor */
	~XNNRecord();

	/* clear it */
	void Clear();

	/* update me with another record */
	void Update(XNNRecord & record);
};
}

#endif
