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

#include <math.h>
#include "T2TTrainer.h"
#include "T2TUtility.h"
#include "../../tensor/XUtility.h"
#include "../../tensor/core/CHeader.h"
#include "../../tensor/loss/LHeader.h"
#include "../../network/XNoder.h"

#ifndef WIN32
#include <sys/time.h>
#include <unistd.h>
#endif

namespace transformer
{

/* constructor */
T2TTrainer::T2TTrainer()
{
    argNum = 0;
    argArray = NULL;
}

/* de-constructor */
T2TTrainer::~T2TTrainer()
{
    for(int i = 0; i < moments.count; i++){
        XTensor * m = (XTensor*)moments.Get(i);
        delete m;
    }

    for(int i = 0; i < moments2nd.count; i++){
        XTensor * m = (XTensor*)moments2nd.Get(i);
        delete m;
    }

    for(int i = 0; i < argNum; i++)
        delete[] argArray[i];
    delete[] argArray;

}

/* 
initialization 
>> argc - number of arguments
>> argv - list of pointers to the arguments
*/
void T2TTrainer::Init(int argc, char ** argv)
{
    argNum = argc;
    argArray = new char*[argc];
    for(int i = 0; i < argNum; i++){
        argArray[i] = new char[strlen(argv[i]) + 1];
        strcpy(argArray[i], argv[i]);
    }

    LoadParamFloat(argc, argv, "lrate", &lrate, 1.0F);
    LoadParamFloat(argc, argv, "lrbias", &lrbias, 0);
    LoadParamInt(argc, argv, "sbatch", &sBatchSize, 1);
    LoadParamInt(argc, argv, "wbatch", &wBatchSize, 1);
    LoadParamInt(argc, argv, "nepoch", &nepoch, 1);
    LoadParamInt(argc, argv, "nstep", &nstep, 1);
    LoadParamInt(argc, argv, "d", &d, 512);
    LoadParamInt(argc, argv, "nwarmup", &nwarmup, 4000);
    LoadParamInt(argc, argv, "vsize", &vSize, 1);
    LoadParamInt(argc, argv, "vsizetgt", &vSizeTgt, vSize);
    LoadParamBool(argc, argv, "adam", &useAdam, false);
    LoadParamFloat(argc, argv, "adambeta1", &adamBeta1, 0.9F);
    LoadParamFloat(argc, argv, "adambeta2", &adamBeta2, 0.98F);
    LoadParamFloat(argc, argv, "adamdelta", &adamDelta, 1e-9F);
    LoadParamBool(argc, argv, "shuffled", &isShuffled, false);
    LoadParamFloat(argc, argv, "labelsmoothing", &labelSmoothingP, 0);
    LoadParamInt(argc, argv, "nstepcheckpoint", &nStepCheckpoint, -1);
    LoadParamBool(argc, argv, "epochcheckpoint", &useEpochCheckpoint, false);
    LoadParamInt(argc, argv, "updatestep", &updateStep, 1);
    LoadParamBool(argc, argv, "debug", &isDebugged, false);
    LoadParamBool(argc, argv, "sorted", &isLenSorted, false);

    adamBeta1T = 1.0F;
    adamBeta2T = 1.0F;

    batchLoader.Init(argc, argv);
}

int tc = 0;

/* 
train the model
>> fn - training data file
>> validFN - validation data file
>> modelFN - where we keep the model
>> model - model to train
*/
void T2TTrainer::Train(const char * fn, const char * validFN, const char * modelFN, T2TModel * model)
{
    int step = 0;
    int wc = 0;
    int ws =0;
    int wordCount = 0;
    int wordCountTotal = 0;
    int wordCountBatch = 0;
    bool isEnd = false;
    float loss = 0;
    float lr = 0;
    int nStepCheck = 0;
    int nCheckpoint = 0;
    int nSkipped = 0;
    int gradStep = 0;
    int validStep = 0;
    int epoch = 0;

    char * trainFN = new char[(int)strlen(fn) + 10];
    strcpy(trainFN, fn);

#ifndef WIN32
    if(isShuffled)
        sprintf(trainFN, "%s.random", fn);
#endif

    int devID = model->devID;
    XNet net;

    if(isDebugged)
        net.SetGradEfficientFlag(false);
    
    PrepareModel(model);

    double startT = GetClockSec();
    
    for(epoch = 1; epoch <= nepoch; epoch++){
#ifndef WIN32
        if(isShuffled)
            batchLoader.Shuffle(fn, trainFN);
#endif
        
        FILE * file = fopen(trainFN, "rb");
        CheckNTErrors(file, "cannot open training file!");
        
        wordCount = 0;
        loss = 0;
        
        /* batch of sequences (on the encoder and decoder sides) */
        XTensor batchEnc;
        XTensor batchDec;

        /* labels */
        XTensor label;

        /* padding */
        XTensor paddingEnc;
        XTensor paddingDec;

        /* gold standard */
        XTensor gold;
        
        /* label smoothed gold standard (if needed) */
        XTensor goldSmoothed;
        
        while (batchLoader.LoadBatch(file, model->isLM, 
                                     &batchEnc, &paddingEnc, &batchDec, &paddingDec, &gold, &label,
                                     NULL, vSize, vSizeTgt,
                                     sBatchSize, wBatchSize, isLenSorted, ws, wc, devID, true)) 
        {

            CheckNTErrors(batchEnc.order == 2, "wrong tensor order of the sequence batch");

            /* output probabilities */
            XTensor output;

            /* make the network */
            if(model->isLM)
                model->MakeLM(batchEnc, output, paddingEnc, true);
            else if(model->isMT)
                model->MakeMT(batchEnc, batchDec, output, paddingEnc, paddingDec, true);
            else{
                ShowNTErrors("Illegal model type!");
            }

            /* back-propagation for obtaining gradients */
            //if (labelSmoothingP > 0)
            //    LabelSmooth(&gold, &goldSmoothed, labelSmoothingP);

            XTensor labelOnehot;

            labelOnehot = IndexToOnehot(label, vSizeTgt, labelSmoothingP);
            
            /* make paddings for the output */
            //if (output.GetDim(0) > 0)
                //PadOutput(&output, &labelOnehot, &paddingDec);

            /* get probabilities */
            //float prob = GetProb(&output, &labelOnehot, NULL);
            XTensor lossTensor;
            lossTensor = CrossEntropy(output, labelOnehot, paddingDec);
            float prob = ReduceSumAll(lossTensor);

            DTYPE lossLocal = prob / wc;
            bool doUpdate = (!IsNAN(lossLocal) && !IsINF(lossLocal) && lossLocal < 1e3F);
          
            //XTensor &g = labelSmoothingP > 0 ? goldSmoothed : gold;  

            if (doUpdate) {
                
                /* recale the output for normalized loss */
                //RescaleOutput(&output, &labelOnehot, &paddingDec);
                
                /* back-propagation */
                net.Backward(lossTensor);
                //net.Backward(output, labelOnehot, paddingDec, CROSSENTROPY);
                //net.Backward(output, label, labelSmoothingP, CROSSENTROPY);
                
                gradStep += 1;
                loss += prob;
                wordCount += wc;
                wordCountTotal += wc;
                
                //totalW = wc + ws;
                wordCountBatch += ws;
                /* update the parameters */
                if(gradStep == updateStep){
                    
                    /* learning rate */
                    lr = lrate * (1.0F / (float)sqrt((float)d)) * (float)MIN(pow((float)validStep + 1, -0.5F - lrbias), ((float)validStep + 1) * pow((float)nwarmup, -1.5F - lrbias));
                    
                    /* model update */
                    Update(model, lr);
                    
                    gradStep = 0;
                    validStep++;
                }
            }
            else
                nSkipped++;
            
            if(++step >= nstep){
                isEnd = true;
                break;
            }
            
            if (step % 100 == 0) {
                double elapsed = GetClockSec() - startT;
                XPRINT8(0, stderr, "[INFO] elapsed=%.1fs, step=%d, epoch=%d, tword=%d, sword=%d, loss=%.3f, ppl=%.3f, sppl=%.3f",
                        elapsed, step, epoch, wordCountTotal, wordCountBatch, loss/wordCount, exp(loss/wordCount), exp(prob/wc));
                if (!doUpdate)
                    XPRINT(0, stderr, " (no update)");
                XPRINT(0, stderr, "\n");
            }

            if(nStepCheckpoint > 0 && ++nStepCheck >= nStepCheckpoint){
                MakeCheckpoint(model, validFN, modelFN, "step", step);
                nStepCheck = 0;
                nCheckpoint++;
            }
        }
        
        fclose(file);
        
        if (isEnd)
            break;

        if(useEpochCheckpoint)
            MakeCheckpoint(model, validFN, modelFN, "epoch", epoch);
    }

    double elapsed = GetClockSec() - startT;
    
    epoch = MIN(epoch, nepoch);
    
    XPRINT7(0, stderr, "[INFO] lr=%.2e, elapsed=%.1fs, step=%d, epoch=%d, word=%d, loss=%.3f, ppl=%.3f\n",
            lr, elapsed, step, epoch, wordCountTotal, loss/wordCount, exp(loss/wordCount));
    XPRINT4(0, stderr, "[INFO] training finished (took %.1fs, step=%d, skipped=%d and epoch=%d)\n",
            elapsed, step, nSkipped, epoch);

    delete[] trainFN;
}

/* 
test the model
>> fn - test data file
>> ofn - output data file
>> model - model that is trained
*/
void T2TTrainer::Test(const char * fn, const char * ofn, T2TModel * model)
{
    int wc = 0;
    int ws = 0;
    int wordCount = 0;
    int wordCountTotal = 0;
    int sentCount = 0;
    float loss = 0;

    /* data files */
    FILE * file = fopen(fn, "rb");
    CheckNTErrors(file, "Cannot read the test file");
    FILE * ofile = fopen(ofn, "wb");
    CheckNTErrors(ofile, "Cannot open the output file");

    int devID = model->devID;

    XNet net;
    
    double startT = GetClockSec();
        
    wordCount = 0;
        
    /* batch of input sequences */
    XTensor batchEnc;
    XTensor batchDec;

    /* label */
    XTensor label;

    /* padding */
    XTensor paddingEnc;
    XTensor paddingDec;

    /* gold standard */
    XTensor gold;

    /* an array that keeps the sequences */
    int * seqs = new int[MILLION];
    
    batchLoader.ClearBuf();

    while(batchLoader.LoadBatch(file, model->isLM, 
                                &batchEnc, &paddingEnc, &batchDec, &paddingDec, &gold, &label,
                                seqs, vSize, vSizeTgt,
                                1, 1, false, ws, wc, devID, false))
    {
        CheckNTErrors(batchEnc.order == 2, "wrong tensor order of the sequence batch");
            
        /* output probabilities */
        XTensor output;
            
        /* make the network */
        if(model->isLM)
            model->MakeLM(batchEnc, output, paddingEnc, false);
        else if(model->isMT)
            model->MakeMT(batchEnc, batchDec, output, paddingEnc, paddingDec, false);
        else{
            ShowNTErrors("Illegal model type!");
        }

        int bSize = output.GetDim(0);
        int length = output.GetDim(1);

        /* prediction probabilities */
        XTensor probs;
        InitTensor1D(&probs, bSize * length);

        XTensor labelOnehot;

        labelOnehot = IndexToOnehot(label, vSizeTgt, 0);

        /* get probabilities */
        float prob = GetProb(&output, &labelOnehot, &probs);

        /* dump the test result */
        for(int s = 0; s < bSize; s++){
            DTYPE sum = 0;
            int * seq = seqs + s * length;
            for(int i = 0; i < length; i++){
                if(seq[i] >= 0){
                    fprintf(ofile, "%d ", seq[i]);
                }
                else
                    break;
            }
            fprintf(ofile, "||| ");
            for(int i = 0; i < length; i++){
                if(seq[i] >= 0){
                    DTYPE p = probs.Get1D(s * length + i);
                    fprintf(ofile, "%.3e ", p);
                    sum += p;
                }
                else
                    break;
            }
            fprintf(ofile, "||| %e\n", sum);
        }
            
        loss += -prob;
        wordCount += wc;
        wordCountTotal += wc;
        sentCount += 1;
    }
        
    fclose(file);
    fclose(ofile);

    delete[] seqs;
    
    double elapsed = GetClockSec() - startT;

    XPRINT3(0, stderr, "[INFO] test finished (took %.1fs, word=%d, and ppl=%.3f)\n",
            elapsed,wordCountTotal, exp(loss / wordCount));
}

/* 
make a checkpoint 
>> model - the model
>> validFN - validation data file
>> modelFN - model data file
>> label - label of the model
>> id - id of the checkpoint
*/
void T2TTrainer::MakeCheckpoint(T2TModel * model, const char * validFN, const char * modelFN, const char * label, int id)
{
    char * fn = new char[MAX_LINE_LENGTH];
    char * fn2 = new char[MAX_LINE_LENGTH];
    sprintf(fn, "%s.%s.%03d", modelFN, label, id);
    sprintf(fn2, "%s.%s.%03d.output", modelFN, label, id);

    model->Dump(fn);
    //if(validFN != NULL){
        //T2TTrainer trainer;
        //trainer.Init(argNum, argArray);
        //trainer.Test(validFN, fn2, model);
    //}

    delete[] fn;
    delete[] fn2;
}

/*
get word probabilities for a batch of sequences
>> output - word distribution for each position
>> gold - gold standard
>> wordProbs - word probability for gold prediction
*/
float T2TTrainer::GetProb(XTensor * output, XTensor * gold, XTensor * wordProbs)
{
    XTensor probs;
    InitTensorV2(&probs, output);
    
    _Multiply(output, gold, &probs);
    
    /* probability of each word */
    XTensor wprobs;
    InitTensor1D(&wprobs, output->unitNum/output->GetDim(-1), X_FLOAT, output->devID);
    
    int dims[2] = {output->unitNum/output->GetDim(-1), output->GetDim(-1)};
    probs.Reshape(2, dims);
    _ReduceSum(&probs, &wprobs, 1);
    
    if(wordProbs != NULL)
        _CopyValues(&wprobs, wordProbs);
    
    /* reshape the tensor to fit it into the reduce procedure
       TODO: XTensor supports scalars */
    dims[0] = 1;
    dims[1] = probs.unitNum;
    probs.Reshape(2, dims);
    
    /* probability for the batch */
    XTensor result;
    InitTensor1D(&result, 1, X_FLOAT, output->devID);
    _ReduceSum(&probs, &result, 1);
    
    return result.Get1D(0);
}

/* 
update the model by delta rule
\theta_new = \theta - \lrate * grad
where
\lrate = d^-0.5 * min(stepNum^-0.5, stepNum * warmupStepNum^-1.5)
>> model - the t2t model
>> lr - learning rate
*/
void T2TTrainer::Update(T2TModel * model, const float lr)
{
    TensorList ws(100);

    model->GetParams(ws);

    for(int i = 0; i < ws.count; i++){
        XTensor * para = (XTensor*)ws.Get(i);
        XTensor * paraGrad = para->grad;

        if (paraGrad == NULL)
            continue;

        CheckNTErrors(para != NULL, "NULL parameter tensor!");
        CheckNTErrors(paraGrad != NULL, "NULL gradient tensor!");

        if(useAdam){
            adamBeta1T *= adamBeta1;
            adamBeta2T *= adamBeta2;
            DTYPE e = lr * (DTYPE)sqrt(1 - adamBeta2T) / (1 - adamBeta1T);
            DTYPE d = adamDelta * (DTYPE)sqrt(1 - adamBeta2T);

            /* m = beta_1 * m + (1-beta_1) * grad */
            XTensor * m = (XTensor*)moments.Get(i);
            _ScaleAndShiftMe(m, adamBeta1, 0);
            _Sum(m, paraGrad, m, (1.0F - adamBeta1));
            
            /* v = beta_2 * v + (1-beta_2) * grad * grad*/
            XTensor * v = (XTensor*)moments2nd.Get(i);
            _Multiply(paraGrad, paraGrad, v, adamBeta2/(1.0F - adamBeta2));
            _ScaleAndShiftMe(v, (1.0F - adamBeta2), 0);

            /* v2 = m / (sqrt(v) + delta) */
            XTensor * v2 = NewTensorBuf(v, v->devID);
            _Power(v, v2, 0.5F);
            _ScaleAndShiftMe(v2, 1.0F, d);
            _Div(m, v2, v2);

            /* the delta rule */
            _Sum(para, v2, para, -e);

            DelTensorBuf(v2);

        }
        else{
            /* the delta rule */
            _Sum(para, paraGrad, para, -lr);
        }

        /* clear gradient */
        paraGrad->SetZeroAll();
    }
}

/* 
prepare model for training 
>> model - the model for training
*/
void T2TTrainer::PrepareModel(T2TModel * model)
{
    moments.Clear();
    moments2nd.Clear();

    TensorList ws(100);

    model->GetParams(ws);

    for(int i = 0; i < ws.count; i++){
        XTensor * para = (XTensor*)ws.Get(i);
        XNoder::MakeGrad(para);

        if(useAdam){
            XTensor * m = new XTensor(para);
            XTensor * m2 = new XTensor(para);
            m->SetZeroAll();
            m2->SetZeroAll();
            moments.Add(m);
            moments2nd.Add(m2);
        }
    }

    adamBeta1T = 1.0F;
    adamBeta2T = 1.0F;
}

/* 
do padding on the output 
>> output - output tensor of the network
>> gold - gold standard
>> padding - padding of a batch of sentences
>> lsP - smoothing factor
*/
void T2TTrainer::PadOutput(XTensor * output, XTensor * gold, XTensor * padding)
{
    if(output == NULL || padding == NULL)
        return;
    
    int on = output->order;
    int * dimso = new int[on];

    memcpy(dimso, output->dimSize, sizeof(int) * on);

    output->Reshape(output->unitNum/dimso[output->order - 1], dimso[output->order - 1]);

    XTensor * padding2 = NewTensorBuf(1, &padding->unitNum, X_FLOAT, padding->devID);

    _CopyValues(padding, padding2);
    _MultiplyDim(output, padding2, output, 0);
    _ScaleAndShiftMe(padding2, 1e9F, -1e9F);
    _SumDim(output, padding2, output, 0);
    
    output->Reshape(on, dimso);
    
    if(gold != NULL){
        gold->Reshape(gold->unitNum/dimso[gold->order - 1], dimso[gold->order - 1]);
        _CopyValues(padding, padding2);
        _MultiplyDim(gold, padding2, gold, 0);
        gold->Reshape(on, dimso);
    }

    delete[] dimso;
    DelTensorBuf(padding2);
}

/*
recale the output and gold tensors for normalized loss
>> output - output tensor of the network
>> gold - gold standard
>> padding - padding of a batch of sentences
*/
void T2TTrainer::RescaleOutput(XTensor * output, XTensor * gold, XTensor * padding)
{
    CheckNTErrors(output->order == 3, "Wrong dimension number!");
    CheckNTErrors(gold->order == 3, "Wrong dimension number!");

    DTYPE count = _ReduceSumAll(padding);
    
    _ExpMe(output);
    _ScaleAndShiftMe(output, 1/count);
    _LogMe(output);

    _ScaleAndShiftMe(gold, 1/count);
}
    
/*
perform label smoothing
>> gold - gold standard
>> smoothed - result of label smoothing
>> p - smoothing factor
*/
void T2TTrainer::LabelSmooth(XTensor * gold, XTensor * smoothed, DTYPE p)
{
    CheckNTErrors(p >= 0 && p <= 1.0F, "Smoothing factor must be in range [0,1]");
    
    int n = gold->GetDim(-1);
    DTYPE q = 1.0F - p;
    DTYPE gift = p / n;
    
    InitTensor(smoothed, gold);
    _CopyValues(gold, smoothed);
    
    if(p == 0)
        return;

    _ScaleAndShiftMe(smoothed, q, gift);
}

}
