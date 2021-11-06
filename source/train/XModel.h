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
* I created more than one file today, hahaha
*/

#ifndef __XMODEL_H__
#define __XMODEL_H__

#include "XTensorKeeper.h"
#include "../network/XNet.h"
#include "../tensor/XQueue.h"
#include "../tensor/XList.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* a model template for training */
class XModel
{
protected:
    /* mutex of the model */
    MUTEX_HANDLE modelMutex;

public:
    /* the list of model parameters */
    XTensorKeeper * params;

    /* parameter number */
    int paramNum;

public:

    /* constructor */
    XModel();

    /* de-constructor */
    ~XModel();

    /* clear the model (would be overloaded) */
    virtual
    void Clear();

    /* clone the model (would be overloaded) */
    virtual
    XModel * Clone(int devID);

    /* run the neural network */
    virtual
    bool RunSimple(XList * inputs, XList * outputs, XList * golds, XList * losses);

protected:
    /* run the neural network */
    bool RunMe(XList * args);

public:
    /* add a parameter tensor */
    void AddParam(XTensor * param);

    /* check if the parameters are well-defined for training */
    bool CheckParam();

    /* lock the parameter states (wait for unlocking them when
       a run of training is finished) */
    void LockParamsForTraining();

    /* wait for unlocked the parameter states */
    void WaitForUnlockedParams();
    
    /* initial model for running the it */
    void InitForRun();

    /* refresh the model */
    void RefreshMe();

    /* wrapper of RefreshMe() */
    static
    void Refresh(XList * args);

    /* wrapper of Run() */
    static
    bool Run(XList * args);

};

}

#endif // __XMODEL_H__
