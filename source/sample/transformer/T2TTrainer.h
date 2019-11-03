/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2018, Natural Language Processing Lab, Northestern University. 
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

#ifndef __T2TTRAINER_H__
#define __T2TTRAINER_H__

#include "T2TModel.h"
#include "T2TBatchLoader.h"
#include "../../tensor/function/FHeader.h"

using namespace nts;

namespace transformer
{

/* trainer of the T2T model */
class T2TTrainer
{
public:
    /* paramter number */
    int argNum;

    /* parameter array */
    char ** argArray;
    
    /* dimension size of each inner layer */
    int d;
    
    /* step number of warm-up for training */
    int nwarmup;

    /* vocabulary size of the source side */
    int vSize;

    /* vocabulary size of the target side */
    int vSizeTgt;

    /* learning rate */
    float lrate;
    
    /* the parameter that controls the maximum learning rate in training */
    float lrbias;

    /* sentence batch size */
    int sBatchSize;

    /* word batch size */
    int wBatchSize;

    /* training epoch number */
    int nepoch;

    /* traing step number */
    int nstep;

    /* indicates whether we use adam */
    bool useAdam;

    /* hyper parameters of adam*/
    float adamBeta1;
    float adamBeta2;
    float adamDelta;
    float adamBeta1T;
    float adamBeta2T;

    /* list of the moment of the parameter matrics */
    TensorList moments;

    /* list of the 2nd order moment of the parameter matrics */
    TensorList moments2nd;

    /* indicates whether the data file is shuffled for training */
    bool isShuffled;
    
    /* the factor of label smoothing */
    DTYPE labelSmoothingP;

    /* number of steps after which we make a checkpoint */
    int nStepCheckpoint;

    /* indicates whether we make a checkpoint after each traing epoch */
    bool useEpochCheckpoint;
    
    /* number of batches on which we do model update */
    int updateStep;

    /* indicates whether we intend to debug the net */
    bool isDebugged;

    /* indicates whether the sequence is sorted by length */
    bool isLenSorted;

    /* for batching */
    T2TBatchLoader batchLoader;

public:
    /* constructor */
    T2TTrainer();

    /* de-constructor */
    ~T2TTrainer();

    /* initialize the trainer */
    void Init(int argc, char ** argv);

    /* train the model */
    void Train(const char * fn, const char * validFN, const char * modelFN, T2TModel * model);

    /* test the model */
    void Test(const char * fn, const char * ofn, T2TModel * model);

    /* make a checkpoint */
    void MakeCheckpoint(T2TModel * model, const char * validFN, const char * modelFN, const char * label, int id);
    
    /* get word probabilities for a batch of sequences */
    float GetProb(XTensor * output, XTensor * gold, XTensor * wordProbs);

    /* update the model by delta rule */
    void Update(T2TModel * model, const float lr);

    /* prepare model for training */
    void PrepareModel(T2TModel * model);

    /* do padding on the output */
    void PadOutput(XTensor * output, XTensor * gold, XTensor * padding);
    
    /* recale the output and gold tensors for normalized loss */
    void RescaleOutput(XTensor * output, XTensor * gold, XTensor * padding);
    
    /* perform label smoothing */
    void LabelSmooth(XTensor * gold, XTensor * smoothed, DTYPE p);
};


}

#endif
