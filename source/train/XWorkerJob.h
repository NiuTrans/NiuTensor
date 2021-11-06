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
* The worker of running the neural network.
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-02-24
* My son had new glasses yesterday.
*/

#ifndef __XWORDERJOB_H__
#define __XWORDERJOB_H__

#include "XWorker.h"
#include "XModel.h"
#include "XNNRecord.h"
#include "XBaseTemplate.h"
#include "../tensor/XList.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* a model template for training */
class XWorkerJob : public XWorker
{
protected:
    /* the model */
    XModel * model;

    /* the input tensors of the model */
    XList inputs;

    /* the output tensors of the model */
    XList outputs;

    /* the gold standard  */
    XList golds;

    /* the loss */
    XList losses;

    /* record the information in running the neural network */
    XNNRecord record;
    
public:

    /* constructor */
    XWorkerJob();

    /* de-constructor */
    ~XWorkerJob();

    /* set the parameter keeper */
    void SetModel(XModel * myModel);

    /* get the parameter keeper */
    XModel * GetModel();

    /* set the state of the worker */
    void SetState(XWORKER_STATE myState);

    /* clear the worker */
    void Clear();

    /* get the input list */
    XList * GetInput();

    /* get the output list */
    XList * GetOutput();
    
    /* get the gold standard */
    XList * GetGold();

    /* get the loss */
    XList * GetLoss();

    /* get the record of the run */
    XNNRecord * GetRecord();

    /* record some stuff */
    void RecordMe();

    /* get the sum of losses over samples */
    float GetLossAll();
    
    /* get the number of samples */
    int GetSampleNum();

    /* get the number of outputs (predictoins) */
    int GetPredictNum();

    /* add a new job of model refreshment */
    bool AddJobRefresh(XModel * myModel);

    /* add a new job of neural network forward and backward computation (with the input) */
    bool AddJobNeuralNet(XModel * myModel, XList * inputs, XList * outputs, XList * golds, XList * losses);

    /* add a new job of recording the running of the nerual network */
    bool AddJobRecord(XNNRecord * serverRecord);

private:
    /* wrapper of RecordMe */
    static
    void RecordMeStatic(XList * args);
};

}

#endif
