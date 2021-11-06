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
* This class maintains the parameters (and other stuff) for training. It
* could be used to manage the parameter copy and update in training. E.g.,
* one can use this class to keep the parameters on the server side, or
* treat it as an individual model on the worker side.
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-02-24
*/

#include "XModel.h"

/* the nts (NiuTrans.Tensor) namespace */
namespace nts {

/* constructor */
XModel::XModel()
{
    params = NULL;
    paramNum = 0;
    MUTEX_INIT(modelMutex);
}

/* de-constructor */
XModel::~XModel()
{
    Clear();
    MUTEX_DELE(modelMutex);
}

/* clear the model */
void XModel::Clear()
{
    delete[] params;
    paramNum = 0;
}

/* 
clone the model (would be overloaded) 
>> devID - the device on that we keep the model
<< return - a cloned model
*/
XModel * XModel::Clone(int devID)
{
    ShowNTErrors("XModel::Clone() should be overloaded!");
    return NULL;
}

/* 
run the neural network 
>> inputs - inputs of the model
>> outputs - outputs of the model
>> golds - gold standards
>> losses - losses of the input with respect to the gold standards
*/
bool XModel::RunSimple(XList * inputs, XList * outputs, XList * golds, XList * losses)
{
    return false;
}

/* 
run the neural network 
>> args - the arguments
*/
bool XModel::RunMe(XList * args)
{
    CheckNTErrors(args->count >= 3, "More arguments are required!");

    XList * inputs = (XList*)args->GetItem(0);
    XList * outputs = (XList*)args->GetItem(1);
    XList * golds = (XList*)args->GetItem(2);
    XList* losses = (XList*)args->GetItem(3);

    if (RunSimple(inputs, outputs, golds, losses))
        return true;

    ShowNTErrors("You must overload one of these: XModel::RunSimple ... !");
    return false;
}

/* 
add a parameter tensor 
>> param - add a 
*/
void XModel::AddParam(XTensor* param)
{
    param->SetVarFlag();

    XTensorKeeper * newParams = new XTensorKeeper[paramNum + 1];

    for (int i = 0; i < paramNum; i++) {
        newParams[i].tensor = params[i].tensor;
        newParams[i].flag = params[i].flag;
    }

    newParams[paramNum].tensor = param;
    newParams[paramNum].flag = PARAM_STATE_NOT_READY;

    delete[] params;
    params = newParams;
    paramNum++;
}

/* check if the parameters are well-defined for training */
bool XModel::CheckParam()
{
    for (int i = 0; i < paramNum; i++) {
        XTensor * param = params[i].tensor;
        if (!param->isGrad)
            return false;
    }

    return true;
}
    
/* initial model for running the it */
void XModel::InitForRun()
{
    RefreshMe();
}

/* lock the parameter states (wait for unlocking them when
   a run of training is finished) */
void XModel::LockParamsForTraining()
{
    for (int i = 0; i < paramNum; i++) {
        params[i].trainFlag = PARAM_STATE_NOT_READY;
        MUTEX_LOCK(params[i].trainLock);

        /* where is UNLOCK? We will do this when the training (a step)
           is finsished. Then, WaitForUnlockedParams() can continue. In
           such a way, we implement a START-WAIT process in each run
           of training (a step) */
    }
}

/* unlock the parameter states */
void XModel::WaitForUnlockedParams()
{
    for (int i = 0; i < paramNum; i++) {
        /* the lock proceeds only when the trainLock is unlocked 
           in training. In this way, we are actually waiting for
           the FINISHED signal from other workers/threads. */
        MUTEX_LOCK(params[i].trainLock);

        CheckNTErrors(params[i].trainFlag == PARAM_STATE_UPDATED,
                      "the state of the parameter is wrong!");
        MUTEX_UNLOCK(params[i].trainLock);
    }
}

/* refresh the model */
void XModel::RefreshMe()
{
    for (int i = 0; i < paramNum; i++) {
        params[i].tensor->isGradFinished = false;
        params[i].flag = PARAM_STATE_NOT_READY;
        params[i].trainFlag = PARAM_STATE_NOT_READY;
    }
}

/* wrapper of RefreshMe */
void XModel::Refresh(XList * args)
{
    CheckNTErrors(args != NULL || args->count == 0, "no arguments for XModel::Refresh");
    XModel * model = (XModel*)args->GetItem(0);
    model->RefreshMe();
}

/* wrapper of Run() */
bool XModel::Run(XList * args)
{
    CheckNTErrors(args != NULL || args->count == 0, "no arguments for XModel::Refresh");
    XModel * model = (XModel*)args->GetItem(0);
    XList newArgs;
    
    for (int i = 1; i < args->count; i++) {
        void * arg = args->GetItem(i);
        newArgs.Add(arg);
    }

    return model->RunMe(&newArgs);
}

} /* end of the nts (NiuTrans.Tensor) namespace */
