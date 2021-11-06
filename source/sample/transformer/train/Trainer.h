/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2018, Natural Language Processing Lab, Northeastern University.
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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-08-02
 */

#ifndef __TRAINER_H__
#define __TRAINER_H__

#include "../Model.h"
#include "TrainDataSet.h"
#include "../../../train/XLearningRate.h"

using namespace nts;

/* the nmt namespace */
namespace nmt
{

/* trainer of the  model */
class Trainer
{
public:
    /* parameters of adam */
    float adamBeta1T;
    float adamBeta2T;

    /* the model for training */
    NMTModel* model;

    /* configurations */
    NMTConfig* config;

    /* list of the moment of the parameters */
    TensorList moments;

    /* list of the 2nd order moment of the parameters */
    TensorList moments2nd;

    /* used for loading batches for training */
    TrainDataSet trainBatchLoader;

    /* used for loading batches for validation */
    TrainDataSet validBatchLoader;

    /* the learning rate scheduler */
    XLearningRate LRScheduler;

public:
    /* constructor */
    Trainer();

    /* de-constructor */
    ~Trainer();

    /* initialize the trainer */
    void Init(NMTConfig& myConfig, NMTModel& myModel);

    /* train the model for several epochs */
    void Run();

    /* train the model for a step */
    void RunStep();

    /* test the model */
    void Validate();

    /* make a checkpoint */
    void MakeCheckpoint(const char* label, int id);

    /* update the model by delta rule */
    void Update(const float lr);

    /* prepare model for training */
    void PrepareModel();
};

} /* end of the nmt namespace */

#endif /* end of __TRAINER_H__ */