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
* This class organizes the training process of neural models, e.g., nmt and lm models
* Distributed training is supported.
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-02-23
* I start coding in 2021 after one year since I typed last line of C code.
* BUT i was a GOOD tex writter in 2020 :)
*/

#ifndef __XTRAINER_H__
#define __XTRAINER_H__

#include "XLeaderPS.h"
#include "XLeaderAllReduce.h"
#include "../network/XNet.h"
#include "../tensor/XQueue.h"
#include "../tensor/XConfig.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#define MAX_DEVICE_NUM_TRAINING 128

/* 
Training of neural networks with gradient methods. Here we suppose that we 
are training NLP models. The routine could be:
   
1). initialize all we need
2). data preparation
3). loop until convergence
    a). read a batch of samples from the input file
    b). reset the worker
    c). forward computation with the input
    d). backward computation with respect to the loss
    e). collect the gradient (neccessary when several workers are available)
    f). update the model (on the server end)
    g). distribute the new model to each worker

Here a worker processes a batch of samples one time, and works with
other workers independently. The server is the origanizer. It distriute
the job to the workers and maintain the model.
*/
class XTrainer
{
public:
    /* constructor */
    XTrainer();

    /* de-constructor */
    ~XTrainer();

protected:
    /* get the device ids of the jobs */
    void GetDevIDs(XConfig * config, int * ids, int & num, int maxDevNum);

public:
    /* run the leader (this is the core process) */
    virtual
    void Run(XConfig * config, DataDistributeBase * dataDistributor, 
             XModel * model, XOptimizer * optimizer);

    /* show settings of training */
    void ShowSettings(XConfig * config);
};
}
#endif // __XTRAINER_H__