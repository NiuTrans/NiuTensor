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
 *
 * This is a simple impelementation of the feed-forward network-baesd language
 * model (FNNLM). See more details about FNNLM in
 * "A Neural Probabilistic Language Model" by Bengio et al.
 * Journal of Machine Learning Research 3 (2003) 1137-1155
 *
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-06-22
 */

#include <math.h>
#include "FNNLM.h"
#include "../../tensor/XGlobal.h"
#include "../../tensor/XUtility.h"
#include "../../tensor/XDevice.h"
#include "../../tensor/function/FHeader.h"
#include "../../network/XNet.h"

namespace fnnlm
{

#define MAX_NAME_LENGTH 1024
#define MAX_LINE_LENGTH_HERE 1024 * 32

char trainFN[MAX_NAME_LENGTH] = "";   // file name of the training data
char modelFN[MAX_NAME_LENGTH] = "";   // file name of the FNN model
char testFN[MAX_NAME_LENGTH] = "";    // file name of the test data
char outputFN[MAX_NAME_LENGTH] = "";  // file name of the result data
    
float learningRate = 0.01F;           // learning rate
int nStep = 10000000;                   // max learning steps (or model updates)
int nEpoch = 10;                      // max training epochs
float minmax = 0.08F;                 // range [-p,p] for parameter initialization
int sentBatch = 0;                    // batch size at the sentence level
int wordBatch = 1;                    // batch size at the word level
bool shuffled = false;                // shuffled the training data file or not
bool autoDiff = false;                // indicator of automatic differentiation

void LoadArgs(int argc, const char ** argv, FNNModel &model);
void Init(FNNModel &model);
void Check(FNNModel &model);
void Copy(FNNModel &tgt, FNNModel &src);
void Clear(FNNModel &model, bool isNodeGrad);
void InitModelTensor1D(XTensor &tensor, int num, FNNModel &model);
void InitModelTensor2D(XTensor &tensor, int rowNum, int colNum, FNNModel &model);
void Train(const char * train, bool isShuffled, FNNModel &model);
void Update(FNNModel &model, FNNModel &grad, float epsilon, bool isNodeGrad);
float GetProb(XTensor &output, XTensor &gold, XTensor * wordProbs = NULL);
void Dump(const char * fn, FNNModel &model);
void Read(const char * fn, FNNModel &model);
void Test(const char * test, const char * result, FNNModel &model);
int  LoadNGrams(FILE * file, int n, NGram * ngrams, int sentNum, int wordNum);
void InitZeroOneTensor2D(XTensor &tensor, int rowNum, int colNum, int * rows, int * cols, 
                         int itemNum, int devID);
void MakeWordBatch(XTensor &batch, NGram * ngrams, int ngramNum, int n, int vSize, int devID);
void Forward(XTensor inputs[], XTensor &output, FNNModel &model, FNNNet &net);
void Backward(XTensor inputs[], XTensor &output, XTensor &gold, LOSS_FUNCTION_NAME loss, 
              FNNModel &model, FNNModel &grad, FNNNet &net);
void ForwardAutoDiff(XTensor inputs[], XTensor &output, FNNModel &model);
void ForwardAutoDiff(NGram * ngrams, int batch, XTensor &output, FNNModel &model);

/* 
entry of the program 
>> argc - number of the arguments
>> argv - pointers to the arguments
<< return - error code

arguments:
 -train S: specify training data file name
 -model S: specify model file name
 -test S: specify test data file name
 -output S: specify result data file name
 -n D: order of the language model
 -eSize D: embedding size
 -vSize D: vocabulary size
 -hdepth D: number of stacked hidden layers
 -hsize D: size of each hidden layer
 -lrate F: learning rate
 -nstep D: maximum number of model updates
 -nepoch D: maximum number of training epochs
 -batch D: batch size (how many sentences)
 -wbatch D: batch size at the word level
            (how many words)
 -shuffle: shuffle the training data
 -devid D: the id of the device used
           -1: CPU, >=0: GPUs
 -mempool: use memory pools for memory management
 -autodiff: use automatic differentiation for training
 
 where S=string, D=integer and F=float.
 All words in the training and test data files
 are encoded as thire indeces in the vocabulary.
 E.g.,
 0 29 2 11 1
 might be a line of the file.
*/
int FNNLMMain(int argc, const char ** argv)
{
    if(argc == 0)
        return 1;

    FNNModel model;

    /* load arguments */
    LoadArgs(argc, argv, model);

    /* check the setting */
    Check(model);

    /* initialize model parameters */
    Init(model);

    /* learn model parameters */
    if(strcmp(trainFN, ""))
        Train(trainFN, shuffled, model);

    /* save the final model */
    if(strcmp(modelFN, "") && strcmp(trainFN, ""))
        Dump(modelFN, model);

    /* load the model if neccessary */
    if(strcmp(modelFN, ""))
        Read(modelFN, model);

    /* test the model on the new data */
    if(strcmp(testFN, "") && strcmp(outputFN, ""))
        Test(testFN, outputFN, model);

    return 0;
}

/* 
load arguments 
>> argc - number of the arguments
>> argv - pointers to the arguments
>> model - the fnn model
*/
void LoadArgs(int argc, const char ** argv, FNNModel &model)
{
    fprintf(stderr, "args:\n");
    for(int i = 0; i < argc; i++){
        if(!strcmp(argv[i], "-train") && i + 1 < argc){
            strcpy(trainFN, argv[i + 1]);
            fprintf(stderr, " -train=%s\n", argv[i + 1]);
        }
        if(!strcmp(argv[i], "-model") && i + 1 < argc){
            strcpy(modelFN, argv[i + 1]);
            fprintf(stderr, " -model=%s\n", argv[i + 1]);
        }
        if(!strcmp(argv[i], "-test") && i + 1 < argc){
            strcpy(testFN, argv[i + 1]);
            fprintf(stderr, " -test=%s\n", argv[i + 1]);
        }
        if(!strcmp(argv[i], "-output") && i + 1 < argc){
            strcpy(outputFN, argv[i + 1]);
            fprintf(stderr, " -output=%s\n", argv[i + 1]);
        }
        if(!strcmp(argv[i], "-n") && i + 1 < argc){
            model.n = atoi(argv[i + 1]);
            fprintf(stderr, " -n=%d\n", model.n);
        }
        if(!strcmp(argv[i], "-esize") && i + 1 < argc){
            model.eSize = atoi(argv[i + 1]);
            fprintf(stderr, " -esize=%d\n", model.eSize);
        }
        if(!strcmp(argv[i], "-vsize") && i + 1 < argc){
            model.vSize = atoi(argv[i + 1]);
            fprintf(stderr, " -vsize=%d\n", model.vSize);
        }
        if(!strcmp(argv[i], "-hdepth") && i + 1 < argc){
            model.hDepth = atoi(argv[i + 1]);
            fprintf(stderr, " -hdepth=%d\n", model.hDepth);
        }
        if(!strcmp(argv[i], "-hsize") && i + 1 < argc){
            model.hSize = atoi(argv[i + 1]);
            fprintf(stderr, " -hsize=%d\n", model.hSize);
        }
        if(!strcmp(argv[i], "-lrate") && i + 1 < argc){
            learningRate = (float)atof(argv[i + 1]);
            fprintf(stderr, " -lrate=%f\n", learningRate);
        }
        if(!strcmp(argv[i], "-nstep") && i + 1 < argc){
            nStep = atoi(argv[i + 1]);
            fprintf(stderr, " -nstep=%d\n", nStep);
        }
        if(!strcmp(argv[i], "-nepoch") && i + 1 < argc){
            nEpoch = atoi(argv[i + 1]);
            fprintf(stderr, " -nepoch=%d\n", nEpoch);
        }
        if(!strcmp(argv[i], "-minmax") && i + 1 < argc){
            minmax = (float)fabs(atof(argv[i + 1]));
            fprintf(stderr, " -minmax=%f\n", minmax);
        }
        if(!strcmp(argv[i], "-batch") && i + 1 < argc){
            sentBatch = atoi(argv[i + 1]);
            fprintf(stderr, " -batch=%d\n", sentBatch);
        }
        if(!strcmp(argv[i], "-wbatch") && i + 1 < argc){
            wordBatch = atoi(argv[i + 1]);
            fprintf(stderr, " -wbatch=%d\n", wordBatch);
        }
        if(!strcmp(argv[i], "-shuffle")){
            shuffled = true;
            fprintf(stderr, " -shuffle=true\n");
        }
        if(!strcmp(argv[i], "-autodiff")){
            autoDiff = true;
            fprintf(stderr, " -autodiff=true\n");
        }
        if(!strcmp(argv[i], "-dev") && i + 1 < argc){
            model.devID = atoi(argv[i + 1]);
            fprintf(stderr, " -dev=%d\n", model.devID);
        }
    }
}

/* check model settings */
void Check(FNNModel &model)
{
    CheckErrors(model.n > 0 && model.n <= MAX_N_GRAM, "The LM order is out of range (use -n)!");
    CheckErrors(model.vSize > 0, "no vocabulary size found (use -vsize)!");
    CheckErrors(model.eSize > 0, "no embedding size found (use -esize)!");
}

/* make a hard copy of the fnn model */
void Copy(FNNModel &tgt, FNNModel &src)
{
    InitTensor(&tgt.embeddingW, &src.embeddingW);
    for(int i = 0; i < MAX_HIDDEN_NUM; i++){
        InitTensor(&tgt.hiddenW[i], &src.hiddenW[i]);
        InitTensor(&tgt.hiddenB[i], &src.hiddenB[i]);
    }
    InitTensor(&tgt.outputW, &src.outputW);
    InitTensor(&tgt.outputB, &src.outputB);

    tgt.n = src.n;
    tgt.eSize = src.eSize;
    tgt.hDepth = src.hDepth;
    tgt.hSize = src.hSize;
    tgt.vSize = src.vSize;
    tgt.devID = src.devID;
    tgt.useMemPool = src.useMemPool;
}

/* 
reset model parameters 
>> model - the model whose parameter (gradient) is set to 0
>> isNodeGrad - indicates whether the tensor node keeps the 
                gradient information
*/
void Clear(FNNModel &model, bool isNodeGrad)
{
    if (isNodeGrad) {
        if(model.embeddingW.grad != NULL)
            model.embeddingW.grad->SetZeroAll();
        for (int i = 0; i < MAX_HIDDEN_NUM; i++) {
            if(model.hiddenW[i].grad != NULL)
                model.hiddenW[i].grad->SetZeroAll();
            if(model.hiddenB[i].grad != NULL)
                model.hiddenB[i].grad->SetZeroAll();
        }
        if(model.outputW.grad != NULL)
            model.outputW.grad->SetZeroAll();
        if(model.outputB.grad != NULL)
            model.outputB.grad->SetZeroAll();
    }
    else {
        model.embeddingW.SetZeroAll();
        for (int i = 0; i < MAX_HIDDEN_NUM; i++) {
            model.hiddenW[i].SetZeroAll();
            model.hiddenB[i].SetZeroAll();
        }
        model.outputW.SetZeroAll();
        model.outputB.SetZeroAll();
    }
}

/* 
initialize a 1d tensor using the fnn model setting 
>> tensor - the tensor to initialize
>> num - number of items
>> model - the fnn model
*/
void InitModelTensor1D(XTensor &tensor, int num, FNNModel &model)
{
    InitTensor1D(&tensor, num, X_FLOAT, model.devID);
}

/* 
initialize a 2d tensor using the fnn model setting 
>> tensor - the tensor to initialize
>> rowNum - number of rows
>> colNum - number of columns
>> model - the fnn model
*/
void InitModelTensor2D(XTensor &tensor, int rowNum, int colNum, FNNModel &model)
{
    InitTensor2D(&tensor, rowNum, colNum, X_FLOAT, model.devID);
}


/* initialize the model */
void Init(FNNModel &model)
{
    /* create embedding parameter matrix: vSize * eSize */
    InitModelTensor2D(model.embeddingW, model.vSize, model.eSize, model);
    model.embeddingW.SetVarFlag();
    
    /* create hidden layer parameter matrics */
    for(int i = 0; i < model.hDepth; i++){
        /* hidden layer parameter matrix: (n-1)eSize * hsize if it is the first layer
                                           hsize * hsize otherwise */
        if(i == 0)
            InitModelTensor2D(model.hiddenW[i], (model.n - 1) * model.eSize, model.hSize, model);
        else
            InitModelTensor2D(model.hiddenW[i], model.hSize, model.hSize, model);
        model.hiddenW[i].SetVarFlag();

        /* bias term: a row vector of hSize entries */
        InitModelTensor1D(model.hiddenB[i], model.hSize, model);
        model.hiddenB[i].SetVarFlag();
    }
    
    /* create the output layer parameter matrix and bias term */
    int iSize = model.hDepth == 0 ? (model.n - 1) * model.eSize : model.hSize;
    InitModelTensor2D(model.outputW, iSize, model.vSize, model);
    InitModelTensor1D(model.outputB, model.vSize, model);
    model.outputW.SetVarFlag();
    model.outputB.SetVarFlag();
    
    /* then, we initialize model parameters using a uniform distribution in range
       of [-minmax, minmax] */
    model.embeddingW.SetDataRand(-minmax, minmax);
    model.outputW.SetDataRand(-minmax, minmax);
    for(int i = 0; i < model.hDepth; i++)
        model.hiddenW[i].SetDataRand(-minmax, minmax);
    
    /* all bias terms are set to zero */
    model.outputB.SetZeroAll();
    for(int i = 0; i < model.hDepth; i++)
        model.hiddenB[i].SetZeroAll();
}
    
/*
 shuffle lines of the file
 >> srcFile - the source file to shuffle
 >> tgtFile - the resulting file
 */
void Shuffle(const char * srcFile, const char * tgtFile)
{
    char * line = new char[MAX_LINE_LENGTH_HERE];
#ifndef WIN32
    sprintf(line, "shuf %s > %s", srcFile, tgtFile);
    system(line);
#else
    ShowErrors("Cannot shuffle the file on WINDOWS systems!");
#endif
    delete[] line;
    
}
    
char lineBuf[MAX_LINE_LENGTH_HERE];
int wordBuf[MAX_LINE_LENGTH_HERE];

/* 
train the model with the standard SGD method
>> train - training data file
>> isShuffled - shuffle the data file or not
>> model - the fnn model
*/
void Train(const char * train, bool isShuffled, FNNModel &model)
{
    char name[MAX_NAME_LENGTH];
    
    /* shuffle the data */
    if(isShuffled){
        sprintf(name, "%s-tmp", train);
        Shuffle(train, name);
    }
    else
        strcpy(name, train);
    
    int epoch = 0;
    int step = 0;
    int wordCount = 0;
    int wordCountTotal = 0;
    int ngramNum = 1;
    float loss = 0;
    bool isEnd = false;
    
    NGram * ngrams = new NGram[MAX_LINE_LENGTH_HERE];

    /* make a model to keep gradients */
    FNNModel grad;
    Copy(grad, model);

    /* XNet for automatic differentiation */
    XNet autoDiffer;

    double startT = GetClockSec();
    
    /* iterate for a number of epochs */
    for(epoch = 0; epoch < nEpoch; epoch++){

        /* data file */
        FILE * file = fopen(name, "rb");
        CheckErrors(file, "Cannot open the training file");

        wordCount = 0;
        loss = 0;
        ngramNum = 1;

        while(ngramNum > 0){
            
            /* load a minibatch of ngrams */
            ngramNum = LoadNGrams(file, model.n, ngrams, sentBatch, wordBatch);

            if (ngramNum <= 0)
                break;

            /* previous n - 1 words */
            XTensor inputs[MAX_N_GRAM];

            /* the predicted word */
            XTensor output;

            /* the gold standard */
            XTensor gold;

            /* the loss tensor */
            XTensor lossTensor;

            /* make the input tensor for position i */
            for(int i = 0; i < model.n - 1; i++)
                MakeWordBatch(inputs[i], ngrams, ngramNum, i, model.vSize, model.devID);

            /* make the gold tensor */
            MakeWordBatch(gold, ngrams, ngramNum, model.n - 1, model.vSize, model.devID);

            if(!autoDiff){
                /* prepare an empty network for building the fnn */
                FNNNet net;

                /* gradident = 0 */
                Clear(grad, false);

                /* forward computation */
                Forward(inputs, output, model, net);

                /* backward computation to obtain gradients */
                Backward(inputs, output, gold, CROSSENTROPY, model, grad, net);

                /* update model parameters */
                Update(model, grad, learningRate, false);

                /* get probabilities */
                float prob = GetProb(output, gold);
                loss -= prob;
            }
            else{
                /* gradient = 0 */
                Clear(model, true);

                /* forward + backward process */
                
                /* this is implemented by gather function */
                ForwardAutoDiff(ngrams, ngramNum, output, model);
                
                /* this is implemented by multiply function */
                lossTensor = CrossEntropy(output, gold);

                /* automatic differentiation */
                autoDiffer.Backward(lossTensor);

                /* update model parameters */
                Update(model, grad, learningRate, true);

                /* get probabilities */
                float prob = ReduceSumAll(lossTensor);
                loss += prob;
            }

            wordCount += ngramNum;
            wordCountTotal += ngramNum;
            
            if(++step >= nStep){
                isEnd = true;
                break;
            }

            if (step % 100 == 0) {
                double elapsed = GetClockSec() - startT;
                XPRINT5(0, stderr, "[INFO] elapsed=%.1fs, step=%d, epoch=%d, ngram=%d, ppl=%.3f\n",
                           elapsed, step, epoch + 1, wordCountTotal, exp(loss / wordCount));
            }
        }

        fclose(file);
        
        if(isEnd)
            break;

        Test(testFN, outputFN, model);
    }

    double elapsed = GetClockSec() - startT;
    
    XPRINT5(0, stderr, "[INFO] elapsed=%.1fs, step=%d, epoch=%d, ngram=%d, ppl=%.3f\n", 
               elapsed, step, epoch, wordCountTotal, exp(loss / wordCount));
    XPRINT3(0, stderr, "[INFO] training finished (took %.1fs, step=%d and epoch=%d)\n", 
               elapsed, step, epoch);
    
    delete[] ngrams;
}

/* 
update the model parameters using the delta rule
>> model - the model to update
>> grad - gradients
>> epsilon - learning rate
>> isNodeGrad - indicates whether the gradient is associated with the node
*/
void Update(FNNModel &model, FNNModel &grad, float epsilon, bool isNodeGrad)
{
    TensorList paraList(10);
    TensorList gradList(10);

    paraList.Add(&model.outputW);
    paraList.Add(&model.outputB);

    for (int i = 0; i < model.hDepth; i++) {
        paraList.Add(&model.hiddenW[i]);
        paraList.Add(&model.hiddenB[i]);
    }

    paraList.Add(&model.embeddingW);

    if(!isNodeGrad){
        gradList.Add(&grad.outputW);
        gradList.Add(&grad.outputB);

        for (int i = 0; i < model.hDepth; i++) {
            gradList.Add(&grad.hiddenW[i]);
            gradList.Add(&grad.hiddenB[i]);
        }
;
        gradList.Add(&grad.embeddingW);
    }
    else{
        gradList.Add(model.outputW.grad);
        gradList.Add(model.outputB.grad);

        for (int i = 0; i < model.hDepth; i++) {
            gradList.Add(model.hiddenW[i].grad);
            gradList.Add(model.hiddenB[i].grad);
        }

        gradList.Add(model.embeddingW.grad);
    }

    for (int i = 0; i < paraList.count; i++) {
        XTensor * para = (XTensor*)paraList.GetItem(i);
        XTensor * paraGrad = (XTensor*)gradList.GetItem(i);

        /* the delta rule */
        _Sum(para, paraGrad, para, -epsilon);
    }
}
  
/*
get prediction probabilites of the gold words
>> output - output probabilities
>> gold - gold standard
>> wordPobs - probability of each word
<< return - probability of the batch
*/
float GetProb(XTensor &output, XTensor &gold, XTensor * wordProbs)
{
    XTensor probs;
    InitTensor(&probs, &output);
    
    /* probs[i,j] = output[i,j] * gold[i,j] */
    Multiply(output, gold, probs);

    /* probability of each word */
    XTensor wprobs;
    InitTensor1D(&wprobs, output.GetDim(0), output.dataType, output.devID);
    ReduceSum(probs, wprobs, 1);
    if(wordProbs != NULL)
        CopyValues(wprobs, *wordProbs);

    /* reshape the tensor to fit it into the reduce procedure 
       TODO: XTensor supports scalars */
    int dims[2];
    dims[0] = 1;
    dims[1] = probs.unitNum;
    probs.Reshape(2, dims);
 
    /* probability for the batch */
    XTensor result;
    InitTensor1D(&result, 1, X_FLOAT, output.devID);
    ReduceSum(probs, result, 1);
    
    return result.Get1D(0);
}

int pin = 0;
int wordBufCount = 0;

/*
load a minibatch of ngrams
>> file - data file
>> n - order of the language model
>> ngrams - the loaded ngrams
>> sentNum - maximum sentences kept in the minibatch
>> wordNum - maximum words kept in the minibatch
*/
int LoadNGrams(FILE * file, int n, NGram * ngrams, int sentNum, int wordNum)
{
    int num = 0;
    int lineNum = 0;
    while(pin > 0 || fgets(lineBuf, MAX_LINE_LENGTH_HERE - 1, file)){
        if(pin <= 0){
            int len = (int)strlen(lineBuf);

            while(lineBuf[len - 1] == '\r' || lineBuf[len - 1] == '\n'){
                lineBuf[len - 1] = 0;
                len--;
            }

            len = (int)strlen(lineBuf);
            if(len == 0)
                continue;
        
            /* how many characters are in a word */
            int wSize = 0;
        
            /* how many words are in the sentence */
            int wNum = 0;
            int i = 0;

            for(i = pin; i < len; i++){
                /* load word (id) seperated by space or tab */
                if((lineBuf[i] == ' ' || lineBuf[i] == '\t') && wSize > 0){
                    lineBuf[i] = 0;
                    wordBuf[wNum++] = atoi(lineBuf + i - wSize);
                    wSize = 0;
                }
                else
                    wSize++;
            }

            if(wSize > 0)
                wordBuf[wNum++] = atoi(lineBuf + i - wSize);

            wordBufCount = wNum;
            lineNum++;
        }
        else
            lineNum = 1;

        int i = -MAX_INT;

        /* create ngrams */
        for(i = MAX(pin, n - 1); i < wordBufCount - 1; i++){
            memcpy(ngrams[num++].words, wordBuf + i - n + 1, sizeof(int) * n);
            if(num >= wordNum)
                break;
        }

        /* set a finished flag if we reach the end of the sentence*/
        if(i >= wordBufCount - 1){
            pin = 0;
            wordBufCount = 0;
        }
        /* record where to start next time if we break in the middle */
        else{
            pin = i + 1;
        }
        
        if((sentNum > 0 && lineNum >= sentNum) || num >= wordNum)
            break;
    }
    
    return num;
}

/*
make a 2d tensor in zero-one representation
The indexed cell is set to 1, and 0 otherwise.
>> tensor - the tensor to initialize
>> rowNum - number of rows
>> colNum - number of columns
>> rows - row index
>> cols - column index
>> itemNum - number of non-zero items
>> devID - device id
*/
void InitZeroOneTensor2D(XTensor &tensor, int rowNum, int colNum, int * rows, int * cols, 
                         int itemNum, int devID)
{
    InitTensor2D(&tensor, rowNum, colNum, X_FLOAT, devID);

    tensor.SetZeroAll();

    /* set none-zero cells */
    for(int i = 0; i < itemNum; i++)
        tensor.Set2D(1.0F, rows[i], cols[i]);
}

/*
make a tensor that encodes a batch of words
>> batch - the tensor encoding a batch of words
>> ngrams - the ngram batch
>> ngramNum - batch size
>> n - indicate which word is encode for each ngram
>> vSize - vocabulary size
>> devID - device id
*/
void MakeWordBatch(XTensor &batch, NGram * ngrams, int ngramNum, int n, int vSize, int devID)
{
    int * rows = new int[ngramNum];
    int * cols = new int[ngramNum];

    for(int i = 0; i < ngramNum; i++){
        rows[i] = i;
        cols[i] = ngrams[i].words[n];
    }

    InitZeroOneTensor2D(batch, ngramNum, vSize, rows, cols, ngramNum, devID);

    delete[] rows;
    delete[] cols;
}

/*
forward procedure
>> inputs - input word representations
>> output - output probability
>> model - the fnn model
>> net - the network that keeps the internal tensors generated in the process
*/
void Forward(XTensor inputs[], XTensor &output, FNNModel &model, FNNNet &net)
{
    int batchSize = -1;
    int n = model.n;
    int depth = model.hDepth;
    TensorList eList(n - 1);

    /* previoius n - 1 words */
    for(int i = 0; i < n - 1; i++){
        XTensor &input = inputs[i];
        XTensor &w = model.embeddingW;
        XTensor &embedding = net.embeddings[i];

        if(batchSize == -1)
            batchSize = input.dimSize[0];
        else{
            CheckErrors(batchSize == input.dimSize[0], "Wrong input word representations!");
        }

        /* embedding output tensor of position i */
        InitModelTensor2D(embedding, batchSize, model.eSize, model);

        /* generate word embedding of position i:
           embedding = input * w   */
        MatrixMul(input, X_NOTRANS, w, X_NOTRANS, embedding);

        eList.Add(&net.embeddings[i]);
    }

    /* concatenate word embeddings
       embeddingcat = cat(embedding_0...embedding_{n-1}) */
    InitModelTensor2D(net.embeddingCat, batchSize, (n - 1) * model.eSize, model);
    Concatenate(eList, net.embeddingCat, 1);

    /* go over each hidden layer */
    for(int i = 0; i < depth; i++){
        XTensor &h_pre = i == 0 ? net.embeddingCat : net.hiddens[i - 1];
        XTensor &w = model.hiddenW[i];
        XTensor &b = model.hiddenB[i];
        XTensor &h = net.hiddens[i];
        XTensor &s = net.hiddenStates[i];

        InitModelTensor2D(h, batchSize, model.hSize, model);
        InitModelTensor2D(s, batchSize, model.hSize, model);

        /* generate hidden states of layer i: 
           s = h_pre * w    */
        MatrixMul(h_pre, X_NOTRANS, w, X_NOTRANS, s);

        /* make a 2d tensor for the bias term */
        XTensor b2D;
        InitTensor(&b2D, &s);
        Unsqueeze(b, b2D, 0, batchSize);

        /* introduce bias term:
           s = s + b
           NOTE: the trick here is to extend b to a 2d tensor
                 to fit into the 2d representation in tensor summation */
        Sum(s, b2D, s);

        /* pass the state through the hard tanh function:
           h = tanh(s) */
        HardTanH(s, h);
    }

    /* generate the output Pr(w_{n-1}|w_0...w_{n-2}):
       y = softmax(h_last * w) 
       Note that this is the implementation as that in Bengio et al.' paper.
       TODO: we add bias term here */
    {
        XTensor &h_last = depth > 0 ? net.hiddens[depth - 1] : net.embeddingCat;
        XTensor &w = model.outputW;
        XTensor &b = model.outputB;
        XTensor &s = net.stateLast;
        XTensor &y = output;

        InitModelTensor2D(s, batchSize, model.vSize, model);
        InitModelTensor2D(y, batchSize, model.vSize, model);

        /* s = h_last * w  */
        MatrixMul(h_last, X_NOTRANS, w, X_NOTRANS, s);

        XTensor b2D;
        InitTensor(&b2D, &s);
        Unsqueeze(b, b2D, 0, batchSize);

        Sum(s, b2D, s);

        /* y = softmax(s) */
        LogSoftmax(s, y, 1);
    }
}

/*
backward procedure
>> inputs - input word representations
>> output - output probability
>> gold - gold standard
>> loss - loss function name
>> model - the fnn model
>> grad - the model that keeps the gradient information
>> net - the network that keeps the internal tensors generated in the process
*/
void Backward(XTensor inputs[], XTensor &output, XTensor &gold, LOSS_FUNCTION_NAME loss, 
              FNNModel &model,  FNNModel &grad, FNNNet &net)
{
    int batchSize = output.GetDim(0);
    int n = model.n;
    int depth = model.hDepth;

    /* back-propagation for the output layer */
    XTensor &y = output;
    XTensor &s = net.stateLast;
    XTensor &x = depth > 0 ? net.hiddens[depth - 1] : net.embeddingCat;
    XTensor &w = model.outputW;
    XTensor &dedw = grad.outputW;
    XTensor &dedb = grad.outputB;
    XTensor deds(&y);
    XTensor dedx(&x);

    /* for y = softmax(s), we get dE/ds
        where E is the error function (define by loss) */
    _LogSoftmaxBackward(&gold, &y, &s, NULL, &deds, NULL, 1, loss);

    /* for s = x * w, we get 
       dE/w_{i,j} = dE/ds_j * ds/dw_{i,j} 
                  = dE/ds_j * x_{i}
       (where i and j are the row and column indices, and
        x is the top most hidden layer)
       so we know 
       dE/dw = x^T * dE/ds */
    MatrixMul(x, X_TRANS, deds, X_NOTRANS, dedw);

    /* gradient of the bias: dE/db = dE/ds * 1 = dE/ds
    specifically dE/db_{j} = \sum_{i} dE/ds_{i,j} */
    ReduceSum(deds, dedb, 0);

    /* then, we compute 
       dE/dx_{j} = \sum_j' (dE/ds_{j'} * ds_{j'}/dx_j) 
                 = \sum_j' (dE/ds_{j'} * w_{j, j'})
       i.e., 
       dE/dx = dE/ds * w^T */
    MatrixMul(deds, X_NOTRANS, w, X_TRANS, dedx);

    XTensor &gradPassed = dedx;
    XTensor dedsHidden;
    XTensor dedxBottom;
    if (depth > 0)
        InitTensor(&dedsHidden, &dedx);
    InitTensor(&dedxBottom, &net.embeddingCat);

    /* back-propagation from top to bottom in the stack of hidden layers
       for each layer, h = f(s)
                       s = x * w + b */
    for (int i = depth - 1; i >= 0; i--) {
        XTensor &h = net.hiddens[i];
        XTensor &s = net.hiddenStates[i];
        XTensor &x = i == 0 ? net.embeddingCat : net.hiddenStates[i - 1];
        XTensor &w = model.hiddenW[i];
        XTensor &dedh = gradPassed;  // gradient passed though the previous layer
        XTensor &dedx = i == 0 ? dedxBottom : dedh;
        XTensor &deds = dedsHidden;
        XTensor &dedw = grad.hiddenW[i];
        XTensor &dedb = grad.hiddenB[i];
        
        /* backpropagation through the activation fucntion: 
           dE/ds = dE/dh * dh/ds */
        _HardTanHBackward(&h, &s, &dedh, &deds);

        /* gradient of the weight: dE/dw = x^T * dE/ds   */
        MatrixMul(x, X_TRANS, deds, X_NOTRANS, dedw);

        /* gradient of the bias: dE/db = dE/ds * 1 = dE/ds
           specifically dE/db_{j} = \sum_{i} dE/ds_{i,j} */
        ReduceSum(deds, dedb, 0);

        /* gradient of the input: dE/dx = dE/ds * w^T    */
        MatrixMul(deds, X_NOTRANS, w, X_TRANS, dedx);

        if (i > 0)
            CopyValues(dedx, gradPassed);
    }

    TensorList eList(n - 1);

    /* back-propagation for the embedding layer */
    for (int i = 0; i < n - 1; i++) {
        XTensor * dedy = NewTensor2D(batchSize, model.eSize, X_FLOAT, model.devID);
        eList.Add(dedy);
    }

    /* gradient of the concatenation of the embedding layers */
    XTensor &dedyCat = depth > 0 ? dedxBottom : dedx;

    /* split the concatenation of gradients of the embeddings */
    Split(dedyCat, eList, 1, n - 1);

    /* go over for each word */
    for (int i = 0; i < n - 1; i++) {
        XTensor * dedy = (XTensor*)eList.GetItem(i);
        XTensor &x = inputs[i];
        XTensor &dedw = grad.embeddingW;

        /* gradient of the embedding weight: dE/dw += x^T * dE/dy 
           NOTE that we accumulate dE/dw here because the matrix w
           is shared by several layers (or words) */
        MatrixMul(x, X_TRANS, *dedy, X_NOTRANS, dedw, 1.0F, 1.0F);

        delete dedy;
    }
}

/*
forward process (with tensor connections) (this is implemented by gather function)
>> ngrams - the loaded ngrams
>> batch - the tensor encoding a batch of words
>> output - output probability
>> model - the fnn model
*/
void ForwardAutoDiff(NGram * ngrams, int batch, XTensor &output, FNNModel &model)
{
    int n = model.n;
    int depth = model.hDepth;

    XTensor words;
    XTensor embeddingBig;
    XTensor hidden;
    XTensor b;

    int size = batch * (n-1);
    int * index = new int[size];

    for(int i = 0; i < batch; i++){
        for (int j = 0; j < n-1; j++){
            int a = i * (n - 1) + j;
            index[a] = ngrams[i].words[j];
        }
    }

    InitTensor1D(&words, size, X_INT, model.devID);
    words.SetData(index, size);

    embeddingBig = Gather(model.embeddingW, words);

    delete[] index;

    int dimSize[2];
    dimSize[0] = embeddingBig.GetDim(0) / (n - 1);
    dimSize[1] = embeddingBig.GetDim(1) * (n - 1);

    hidden = Reshape(embeddingBig, embeddingBig.order, dimSize);

    /* hidden layers */
    for(int i = 0; i < depth; i++)
        hidden = HardTanH(MMul(hidden, model.hiddenW[i]) + model.hiddenB[i]);

    /* output layer */
    //output = LogSoftmax(MMul(hidden, model.outputW) + model.outputB, 1);
    output = Softmax(MMul(hidden, model.outputW) + model.outputB, 1);
}

/*
forward process (with tensor connections) (this is implemented by multiply function)
>> inputs - input word representations
>> output - output probability
>> model - the fnn model
*/
void ForwardAutoDiff(XTensor inputs[], XTensor &output, FNNModel &model)
{
    int n = model.n;
    int depth = model.hDepth;

    XTensor words;
    XTensor embeddingBig;
    XTensor hidden;
    XTensor b;

    TensorList inputList(n - 1);
    for(int i = 0; i < n - 1; i++)
        inputList.Add(inputs + i);

    /* represent n - 1 words in one tensor */
    words = Merge(inputList, 0);

    /* word embedding */
    embeddingBig = MMul(words, model.embeddingW);

    /* input of the first hidden layer */
    hidden = Split(embeddingBig, 0, n - 1);
    hidden = Merge(hidden, 2, 0);

    /* hidden layers */
    for(int i = 0; i < depth; i++)
        hidden = MMul(hidden, model.hiddenW[i]) + model.hiddenB[i];

    /* output layer */
    output = LogSoftmax(MMul(hidden, model.outputW) + model.outputB, 1);

}

/* 
dump the model to the disk space
>> fn - where to keep the model
>> model - the fnn model
*/
void Dump(const char * fn, FNNModel &model)
{
    FILE * file = fopen(fn, "wb");
    CheckErrors(file, "Cannot open the model file");

    model.embeddingW.Dump(file, "embedding w:");
    for (int i = 0; i < model.hDepth; i++) {
        char name[MAX_NAME_LENGTH];
        sprintf(name, "hidden %d w:", i);
        model.hiddenW[i].Dump(file, name);
        sprintf(name, "hidden %d b:", i);
        model.hiddenB[i].Dump(file, name);
    }

    model.outputW.Dump(file, "output w:");
    model.outputB.Dump(file, "output b:");

    fclose(file);

    XPRINT(0, stderr, "[INFO] model saved\n");
}

/* 
read the model from the disk space
>> fn - where to keep the model
>> model - the fnn model
*/
void Read(const char * fn, FNNModel &model)
{
    FILE * file = fopen(fn, "rb");
    CheckErrors(file, "Cannot open the model file");

    model.embeddingW.Read(file, "embedding w:");
    for (int i = 0; i < model.hDepth; i++) {
        char name[MAX_NAME_LENGTH];
        sprintf(name, "hidden %d w:", i);
        model.hiddenW[i].Read(file, name);
        sprintf(name, "hidden %d b:", i);
        model.hiddenB[i].Read(file, name);
    }

    model.outputW.Read(file, "output w:");
    model.outputB.Read(file, "output b:");

    fclose(file);

    XPRINT(0, stderr, "[INFO] model loaded\n");
}

/* 
test the model
>> test - test data file
>> result - where to keep the result
>> model - the fnn model
*/
void Test(const char * test, const char * result, FNNModel &model)
{
    int wordCount = 0;
    int sentCount = 0;
    float loss = 0;

    NGram * ngrams = new NGram[MAX_LINE_LENGTH_HERE];

    double startT = GetClockSec();

    /* data files */
    FILE * file = fopen(test, "rb");
    CheckErrors(file, "Cannot read the test file");
    FILE * ofile = fopen(result, "wb");
    CheckErrors(ofile, "Cannot open the output file");

    int ngramNum = 1;
    while (ngramNum > 0) {

        /* load a minibatch of ngrams */
        ngramNum = LoadNGrams(file, model.n, ngrams, 1, MAX_INT);

        if (ngramNum <= 0)
            break;

        /* previous n - 1 words */
        XTensor inputs[MAX_N_GRAM];

        /* the predicted word */
        XTensor output;

        /* the gold standard */
        XTensor gold;
        
        /* make the input tensor for position i */
        for (int i = 0; i < model.n - 1; i++)
            MakeWordBatch(inputs[i], ngrams, ngramNum, i, model.vSize, model.devID);

        /* make the gold tensor */
        MakeWordBatch(gold, ngrams, ngramNum, model.n - 1, model.vSize, model.devID);

        if (!autoDiff) {
            /* prepare an empty network for building the fnn */
            FNNNet net;

            /* forward computation */
            Forward(inputs, output, model, net);
        }
        else {            
            /* this is implemented by gather function */
            ForwardAutoDiff(ngrams, ngramNum, output, model);
            output = Log(output);
				
			/* this is implemented by multiply function */
			//ForwardAutoDiff(inputs, output, model);
        }

        /* prediction probabilities */
        XTensor probs;
        InitTensor1D(&probs, ngramNum);

        /* get probabilities */
        float prob = GetProb(output, gold, &probs);

        /* dump the test result */
        for (int i = 0; i < model.n - 1; i++)
            fprintf(ofile, "%d ", ngrams[0].words[i]);
        for (int i = 0; i < ngramNum; i++)
            fprintf(ofile, "%d ", ngrams[i].words[model.n - 1]);
        fprintf(ofile, "||| ");
        for (int i = 0; i < model.n - 1; i++)
            fprintf(ofile, "<s> ");
        for (int i = 0; i < ngramNum; i++)
            fprintf(ofile, "%f ", probs.Get1D(i));
        fprintf(ofile, "||| %f\n", prob);

        loss += -prob;
        wordCount += ngramNum;
        sentCount += 1;
    }

    fclose(file);
    fclose(ofile);

    double elapsed = GetClockSec() - startT;

    XPRINT1(0, stderr, "[INFO] ppl=%.2f\n", exp(loss/wordCount));
    XPRINT3(0, stderr, "[INFO] test finished (took %.1fs, sentence=%d and ngram=%d)\n", 
               elapsed, sentCount, wordCount);

    delete[] ngrams;
}

};
