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
* We test XTrain here. It is simple, we design a simple task in that we
* make the model to predict an integer D (0-100) from three input integers 
* A, B and C (0-100). We generate a number of samples with different values
* of A, B and C. The gold standard is 
*     
*          D = (int)(sqrt(A * B) + C)/2
* 
* Our model is a two-layer feed-forward neural network. It can be treated
* as a classifier rather than a regression model.
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-03-03
* The express train was updated this year. It just takes me two hours and
* a half from Shenyang to Beijing.
*/

#ifndef __TTRAIN_H__
#define __TTRAIN_H__

#include <stdio.h>
#include <stdlib.h>
#include "XTrainer.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#define MAX_SAMPLE_NUM_IN_TTRAIN 200000
#define MAX_INT_IN_TTRAIN 100
#define MAX_SAMPLE_LINE_LENGTH 128
#define MAX_SAMPLE_SIZE 4
#define TT_BATCH_SIZE 256
#define TT_EMBEDDING_SIZE 128
#define TT_HIDDEN_SIZE 512

extern XTensor * tmpTT;

/* genreate the training data file */
void GeneateTTrainData(const char * fileName);

/* run the test */
extern
void TestTrain();

/* data loader */
class TTDataLoader : public DataDistributeBase
{
protected:
    /* file name */
    char * fileName;

    /* file handle */
    FILE * file;

    /* batch size */
    int batchSize;

public:
    /* constructor */
    TTDataLoader();

    /* de-constructor */
    ~TTDataLoader();

    /* set file name */
    void SetFileName(const char * myFileName);

    /* set batch size */
    void SetBatchSize(int myBatchSize);

    /* start the process */
    bool Start();

    /* end the process */
    bool End();

    /* get a batch of samples */
    bool GetBatchSimple(XList * inputs, XList * golds);
};

/* the model */
class TTModel : public XModel
{
protected:
    /* device id */
    int devID;

    /* configuration */
    XConfig config;

    /* embedding matrix of the input */
    XTensor embeddingW;

    /* parameter matrix of the hidden layer */
    XTensor hiddenW;

    /* parameter matrix of the output layer */
    XTensor outputW;

    /* vocabulary size */
    int vSize;

    /* embedding size */
    int eSize;

    /* hidden layer size */
    int hSize;

public:
    /* constructor */
    TTModel();

    /* de-constructor */
    ~TTModel();

    /* config it */
    void SetConfig(XConfig &myConfig);

    /* initialize the parameters */
    void Init(XConfig &myConfig, int myDevID);

    /* create the model */
    void Forward(int devID, XTensor * input, XTensor * output);

    /* clear the model */
    void Clear();

    /* clone the model */
    XModel * Clone(int devID);

    /* run the neural network */
    bool RunSimple(XList * inputs, XList * outputs, XList * golds, XList * losses);
};

/*  */

}

#endif