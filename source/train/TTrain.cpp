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
* make the model to predict an integer D (0-100) from four input integers
* A, B, C and D (0-100). We generate a number of samples with different values
* of A, B, C and D. The gold standard is
*
*          D = (int)(sqrt(A * B) + abs(C - D))/2
*
* Our model is a two-layer feed-forward neural network. It can be treated
* as a classifier rather than a regression model.
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-03-03
*/

#include "TTrain.h"
#include "../tensor/core/CHeader.h"
#include "../tensor/function/FHeader.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

XTensor * tmpTT = NULL;

/* genreate the training data file */
void GeneateTTrainData(const char * fileName)
{
    FILE * file = fopen(fileName, "wb");
    CheckNTErrors(file, "Cannot open the file");

    XPRINT(1, stderr, "[INFO] Generating data ... ");

    int sampleNum = MAX_SAMPLE_NUM_IN_TTRAIN;
    int range = MAX_INT_IN_TTRAIN;

    fprintf(file, "%d\n", sampleNum);

    srand(1);

    for (int i = 0; i < sampleNum; i++) {
        int A = (int)(((float)rand() / RAND_MAX) * range);
        int B = (int)(((float)rand() / RAND_MAX) * range);
        int C = (int)(((float)rand() / RAND_MAX) * range);
        int D = (int)(((float)rand() / RAND_MAX) * range);
        int E = (int)((sqrt(A * B) + abs(C - D)) / 2);
        fprintf(file, "%d %d %d %d %d\n", A, B, C, D, E);
    }

    XPRINT2(1, stderr, "%d samples in \"%s\" [DONE]\n", sampleNum, fileName);
    
    fclose(file);
}

/* run the test */
void TestTrain()
{
    GeneateTTrainData("ttrain.txt");

    XConfig config;
    //config.Add("dev", -1);
    config.Add("lrate", 0.1F);
    config.Add("nstep", 100000);
    config.Add("nepoch", 5);
    config.Add("jobdev0", -1);
    config.Add("jobdev1", -1);
    config.Add("jobdev2", -1);
    config.Add("jobdev3", -1);
    //config.Add("jobdev4", -1);

    int serverDevID = config.GetInt("jobdev0", -1);

    TTDataLoader loader;
    loader.SetFileName("ttrain.txt");
    loader.SetBatchSize(config.GetInt("batchsize", TT_BATCH_SIZE));

    TTModel model;
    model.Init(config, serverDevID);

    XOptimizer optimizer;
    optimizer.Init(config);

    XTrainer trainer;
    trainer.Run(&config, &loader, &model, &optimizer);
}

/*****************************
* data loader
******************************/

/* constructor */
TTDataLoader::TTDataLoader()
{
    fileName = new char[MAX_FILE_NAME_LENGTH];
    file = NULL;
    batchSize = TT_BATCH_SIZE;
}

/* de-constructor */
TTDataLoader::~TTDataLoader()
{
    delete[] fileName;
}

/* set file name */
void TTDataLoader::SetFileName(const char * myFileName)
{
    strcpy(fileName, myFileName);
}

/* set batch size */
void TTDataLoader::SetBatchSize(int myBatchSize)
{
    batchSize = myBatchSize;
}

/* start the process */
bool TTDataLoader::Start()
{
    file = fopen(fileName, "rb");
    CheckNTErrors(file != NULL, "Cannot open the file");

    /* skip the first line */
    char * line = new char[MAX_SAMPLE_LINE_LENGTH];
    fgets(line, MAX_SAMPLE_LINE_LENGTH, file);
    delete[] line;

    return true;
}

/* end the process */
bool TTDataLoader::End()
{
    fclose(file);

    return true;
}

/* 
get a batch of samples 
>> inputs - inputs of the model
>> golds - gold standards
*/
bool TTDataLoader::GetBatchSimple(XList * inputs, XList * golds)
{
    CheckNTErrors(file != NULL, "No input file specificed!");
    CheckNTErrors(inputs != NULL && inputs->count >= 1, "Wrong argument!");
    CheckNTErrors(golds != NULL && golds->count >= 1, "Wrong argument!");

    XTensor * input = (XTensor*)inputs->GetItem(0);
    XTensor * gold = (XTensor*)golds->GetItem(0);

    int count = 0;
    int sampleSize = MAX_SAMPLE_SIZE;
    char * line = new char[MAX_SAMPLE_LINE_LENGTH];
    int * inputBatch = new int[batchSize * sampleSize];
    int * goldBatch = new int[batchSize];
    int A, B, C, D, E;
    
    while (fgets(line, MAX_SAMPLE_LINE_LENGTH, file)) {

        if (count == batchSize)
            break;

        if (sscanf(line, "%d %d %d %d %d", &A, &B, &C, &D, &E) < sampleSize + 1) {
            ShowNTErrors("Wrong format in the training file!");
        }

        inputBatch[count * sampleSize] = A;
        inputBatch[count * sampleSize + 1] = B;
        inputBatch[count * sampleSize + 2] = C;
        inputBatch[count * sampleSize + 3] = D;
        goldBatch[count] = E;

        count++;
    }

    if (count > 0) {
        InitTensor2D(input, count, 4, X_INT);
        InitTensor2D(gold, count, 1, X_INT);

        input->SetData(inputBatch, count * 4);
        gold->SetData(goldBatch, count);
    }

    delete[] line;
    delete[] inputBatch;
    delete[] goldBatch;

    if (count > 0)
        return true;
    else
        return false;
}

/*****************************
* the neural model
******************************/

/* constructor */
TTModel::TTModel()
{
    devID = -1;
    vSize = 0;
    eSize = 0;
    hSize = 0;
}

/* de-constructor */
TTModel::~TTModel()
{
}

/* config it */
void TTModel::SetConfig(XConfig &myConfig)
{
    config.CreateFromMe(myConfig);
}

/* 
initialize the model 
>> myConfig - configuration
>> devID - device id
*/
void TTModel::Init(XConfig &myConfig, int myDevID)
{
    Clear();
    SetConfig(myConfig);

    devID = myDevID;

    vSize = MAX_INT_IN_TTRAIN + 1;
    eSize = config.GetInt("esize", TT_EMBEDDING_SIZE);
    hSize = config.GetInt("hsize", TT_HIDDEN_SIZE);

    InitTensor2D(&embeddingW, vSize, eSize, X_FLOAT, devID);
    InitTensor2D(&hiddenW, MAX_SAMPLE_SIZE * eSize, hSize, X_FLOAT, devID);
    InitTensor2D(&outputW, hSize, vSize, X_FLOAT, devID);

    embeddingW.SetName("embeddingw");
    hiddenW.SetName("hiddenw");
    outputW.SetName("outputw");

    embeddingW.SetDataRand(-0.1F, 0.1F);
    hiddenW.SetDataRand(-0.1F, 0.1F);
    outputW.SetDataRand(-0.1F, 0.1F);
    
    AddParam(&embeddingW);
    AddParam(&hiddenW);
    AddParam(&outputW);
}

/* 
create the model 
>> devID - device id
>> input - as it is
>> output - as it is
*/
void TTModel::Forward(int devID, XTensor * input, XTensor * output)
{
    XTensor embedding;
    XTensor embeddingCat;
    XTensor hidden;

    /* [e_0, e_1, e_2] = w_e * input(one-hot) */
    embedding = Gather(embeddingW, *input);

    /* e = merge(e_0, e_1, e_2) */
    embeddingCat = Merge(embedding, embedding.order - 1, embedding.order - 2);

    /* h = hardtanh(e * w_h) */
    hidden = HardTanH(MMul(embeddingCat, hiddenW));

    /* output = Softmax(h * w_o) */
    *output = Softmax(MMul(hidden, outputW), -1);
}

/* clear the model */
void TTModel::Clear()
{
    config.Clear();
}

/* 
clone the model 
>> devID - device id
*/
XModel * TTModel::Clone(int devID)
{
    TTModel * model = new TTModel();
    model->SetConfig(config);
    model->Init(config, devID);

    CopyValues(embeddingW, model->embeddingW);
    CopyValues(hiddenW, model->hiddenW);
    CopyValues(outputW, model->outputW);

    return model;
}

/* 
run the neural network
>> inputs - inputs of the model
>> outputs - outputs of the model
>> golds - gold standards
>> losses - losses of the output respect to the gold standards
*/
bool TTModel::RunSimple(XList * inputs, XList * outputs, XList * golds, XList* losses)
{
    //fprintf(stderr, "run simple 0\n");
    CheckNTErrors(inputs != NULL && inputs->count >= 1, "Wrong arguments!");
    CheckNTErrors(outputs != NULL && outputs->count >= 1, "Wrong arguments!");
    CheckNTErrors(golds != NULL && golds->count >= 1, "Wrong arguments!");
    CheckNTErrors(losses != NULL && losses->count >= 1, "Wrong arguments!");

    XTensor * input = (XTensor*)inputs->GetItem(0);
    XTensor * output = (XTensor*)outputs->GetItem(0);
    XTensor * gold = (XTensor*)golds->GetItem(0);
    XTensor * loss = (XTensor*)losses->GetItem(0);
    XTensor goldOneHot;

    /* place all input data on the correct device */
    input->FlushToDevice(devID);
    output->FlushToDevice(devID);
    gold->FlushToDevice(devID);

    XNet net;

    /* create the neural network and run it */
    Forward(devID, input, output);

    /* gold standard in ong-hot representaiton */
    goldOneHot = IndexToOnehot(*gold, vSize, 0.0F);

    int * dims = new int[goldOneHot.order];
    for (int i = 0; i < goldOneHot.order - 2; i++)
        dims[i] = goldOneHot.GetDim(i);
    dims[goldOneHot.order - 2] = goldOneHot.GetDim(goldOneHot.order - 1);
    goldOneHot.Reshape(goldOneHot.order - 1, dims);

    /* loss */
    *loss = CrossEntropy(*output, goldOneHot);

    /* back-propagation */
    net.Backward(*loss);

    delete[] dims;
    
    //fprintf(stderr, "run simple 1\n");

    return true;
}

}
