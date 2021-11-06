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
* We define various template classes here. They will be overloaded and used
* in applications.
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-02-25
*/

#include "XBaseTemplate.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/******************************* 
 * data loader template 
 *******************************/

/* constructor */
DataDistributeBase::DataDistributeBase()
{
    MUTEX_INIT(loadMutex);
}

/* de-constructor */
DataDistributeBase::~DataDistributeBase()
{
    MUTEX_DELE(loadMutex);
}

/* * start the job (e.g., open the file) */
bool DataDistributeBase::Start()
{
    ShowNTErrors("DataDistributeBase::Start must be overloaded!");
    return true;
}

/* end the job (e.g., close the file) */
bool DataDistributeBase::End()
{
    ShowNTErrors("DataDistributeBase::End must be overloaded!");
    return true;
}

/* 
get a batch of samples 
>> inputs - inputs of the model
>> golds - gold standards
*/
bool DataDistributeBase::GetBatchSimple(XList * inputs, XList * golds)
{
    return false;
}

/* get a batch of samples */
bool DataDistributeBase::GetBatch(XList * args)
{
    CheckNTErrors(args->count >= 2, "More input arguments are required!");

    XList * input = (XList*)args->GetItem(0);
    XList * gold = (XList*)args->GetItem(1);

    if (GetBatchSimple(input, gold))
        return true;

    ShowNTErrors("You must be overload one of these: DataDistributeBase::GetBatchSimple ... !");
    return false;
}

/* get a batch of samples (for multi-threading) */
bool DataDistributeBase::GetBatchSafe(XList * args)
{
    bool r;

    MUTEX_LOCK(loadMutex);
    r = GetBatch(args);
    MUTEX_UNLOCK(loadMutex);

    return r;
}

}
