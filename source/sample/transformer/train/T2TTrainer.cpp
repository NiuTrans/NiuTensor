/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2020, Natural Language Processing Lab, Northeastern University.
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

#include <cmath>
#include "T2TTrainer.h"
#include "../module/T2TUtility.h"
#include "../../../tensor/XUtility.h"
#include "../../../tensor/core/CHeader.h"
#include "../../../tensor/loss/LHeader.h"
#include "../../../network/XNoder.h"

#ifndef WIN32
#include <sys/time.h>
#include <unistd.h>
#endif

namespace transformer
{

/* constructor */
T2TTrainer::T2TTrainer()
{
    cfg = NULL;
}

/* de-constructor */
T2TTrainer::~T2TTrainer()
{
    for (int i = 0; i < moments.count; i++) {
        XTensor* m = (XTensor*)moments.Get(i);
        delete m;
    }

    for (int i = 0; i < moments2nd.count; i++) {
        XTensor* m = (XTensor*)moments2nd.Get(i);
        delete m;
    }
}

/*
initialization
>> config - configurations of the training process
*/
void T2TTrainer::Init(T2TConfig& config)
{
    cfg = &config;
    lrate = config.lrate;
    lrbias = config.lrbias;
    sBatchSize = config.sBatchSize;
    wBatchSize = config.wBatchSize;
    nepoch = config.nepoch;
    nstep = config.nstep;
    d = config.modelSize;
    nwarmup = config.nwarmup;
    vSize = config.srcVocabSize;
    vSizeTgt = config.tgtVocabSize;
    useAdam = config.useAdam;
    adamBeta1 = config.adamBeta1;
    adamBeta2 = config.adamBeta2;
    adamDelta = config.adamDelta;
    isShuffled = config.isShuffled;
    labelSmoothingP = config.labelSmoothingP;
    nStepCheckpoint = config.nStepCheckpoint;
    useEpochCheckpoint = config.useEpochCheckpoint;
    updateStep = config.updateStep;
    isDebugged = config.isDebugged;
    isLenSorted = config.isLenSorted;

    adamBeta1T = 1.0F;
    adamBeta2T = 1.0F;

    batchLoader.Init(config);
}

int tc = 0;

/*
train the model
>> fn - training data file
>> validFN - validation data file
>> modelFN - where we keep the model
>> model - model to train
*/
void T2TTrainer::Train(const char* fn, const char* validFN, const char* modelFN, T2TModel* model)
{
    int step = 0;
    int wc = 0;
    int ws = 0;
    int wordCount = 0;
    int wordCountTotal = 0;
    int batchCountTotal = 0;
    bool isEnd = false;
    float loss = 0;
    float lr = 0;
    int nStepCheck = 0;
    int nCheckpoint = 0;
    int nSkipped = 0;
    int gradStep = 0;
    int validStep = 0;
    int epoch = 0;

    char* trainFN = new char[(int)strlen(fn) + 10];
    strcpy(trainFN, fn);

#ifndef WIN32
    if (isShuffled)
        sprintf(trainFN, "%s.random", fn);
#endif

    int devID = model->devID;
    XNet net;

    PrepareModel(model);

    double startT = GetClockSec();

    for (epoch = 1; epoch <= nepoch; epoch++) {
#ifndef WIN32
        if (isShuffled) {
            fprintf(stderr, "shuffle the file\n");
            batchLoader.Shuffle(fn, trainFN);
        }
#endif

        FILE* file = fopen(trainFN, "r");
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

        while (batchLoader.LoadBatch(file, model->isLM,
            &batchEnc, &paddingEnc, &batchDec, &paddingDec, &gold, &label,
            NULL, vSize, vSizeTgt,
            sBatchSize, wBatchSize, isLenSorted, ws, wc, devID, true))
        {
            CheckNTErrors(batchEnc.order == 2, "wrong tensor order of the sequence batch");

            /* output probabilities */
            XTensor output;

            /* make the network */
            if (model->isLM)
                model->MakeLM(batchEnc, output, paddingEnc, true);
            else if (model->isMT)
                model->MakeMT(batchEnc, batchDec, output, paddingEnc, paddingDec, true);
            else {
                ShowNTErrors("Illegal model type!");
            }

            /* get loss and probabilities */
            XTensor labelOnehot;
            XTensor lossTensor;

            labelOnehot = IndexToOnehot(label, vSizeTgt, labelSmoothingP);

            lossTensor = CrossEntropy(output, labelOnehot, paddingDec);

            float lossBatch = ReduceSumAllValue(lossTensor);

            DTYPE lossLocal = lossBatch / wc;
            bool doUpdate = (!IsNAN(lossLocal) && !IsINF(lossLocal) && lossLocal < 1e3F);

            if (doUpdate) {
                /* back-propagation */
                net.Backward(lossTensor);

                gradStep += 1;
                loss += lossBatch;
                wordCount += wc;
                wordCountTotal += wc;
                batchCountTotal += ws;

                /* update the parameters */
                if (gradStep == updateStep) {
                    /* learning rate */
                    lr = lrate * (1.0F / (float)sqrt((float)d)) *
                        (float)MIN(pow((float)validStep + 1, -0.5F - lrbias),
                        ((float)validStep + 1) * pow((float)nwarmup, -1.5F - lrbias));

                    /* model update */
                    Update(model, lr);

                    gradStep = 0;
                    validStep++;
                }
            }
            else
                nSkipped++;

            if (++step >= nstep) {
                isEnd = true;
                break;
            }

            if (step % 100 == 0) {
                double elapsed = GetClockSec() - startT;
                XPRINT8(0, stderr, "[INFO] elapsed=%.1fs, step=%d, epoch=%d, total word=%d, total batch=%d, loss=%.3f, ppl=%.3f, sppl=%.3f",
                    elapsed, step, epoch,
                    wordCountTotal, batchCountTotal,
                    loss / wordCount, exp(loss / wordCount), exp(lossBatch / wc));
                if (!doUpdate)
                    XPRINT(0, stderr, " (no update)");
                XPRINT(0, stderr, "\n");
            }

            if (nStepCheckpoint > 0 && ++nStepCheck >= nStepCheckpoint) {
                MakeCheckpoint(model, validFN, modelFN, "step", step);
                nStepCheck = 0;
                nCheckpoint++;
            }
        }

        fclose(file);

        if (isEnd)
            break;

        if (useEpochCheckpoint)
            MakeCheckpoint(model, validFN, modelFN, "epoch", epoch);
    }

    double elapsed = GetClockSec() - startT;

    epoch = MIN(epoch, nepoch);

    XPRINT7(0, stderr, "[INFO] lr=%.2e, elapsed=%.1fs, step=%d, epoch=%d, word=%d, loss=%.3f, ppl=%.3f\n",
        lr, elapsed, step, epoch, wordCountTotal, loss / wordCount, exp(loss / wordCount));
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
void T2TTrainer::Validate(const char* fn, const char* ofn, T2TModel* model)
{
    int wc = 0;
    int ws = 0;
    int wordCount = 0;
    int sentCount = 0;
    float loss = 0;

    /* data files */
    FILE* file = fopen(fn, "rb");
    CheckNTErrors(file, "Cannot read the test file");
    FILE* ofile = fopen(ofn, "wb");
    CheckNTErrors(ofile, "Cannot open the output file");

    double startT = GetClockSec();

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
    int* seqs = new int[MILLION];

    batchLoader.ClearBuf();

    while (batchLoader.LoadBatch(file, model->isLM,
        &batchEnc, &paddingEnc, &batchDec, &paddingDec, &gold, &label,
        seqs, vSize, vSizeTgt,
        1, 1, false, ws, wc, model->devID, false))
    {
        CheckNTErrors(batchEnc.order == 2, "wrong tensor order of the sequence batch");

        /* output probabilities */
        XTensor output;

        /* make the network */
        if (model->isLM)
            model->MakeLM(batchEnc, output, paddingEnc, false);
        else if (model->isMT)
            model->MakeMT(batchEnc, batchDec, output, paddingEnc, paddingDec, false);
        else {
            ShowNTErrors("Illegal model type!");
        }

        int bSize = output.GetDim(0);
        int length = output.GetDim(1);

        /* prediction probabilities */
        XTensor labelOnehot;
        XTensor lossTensor;
        labelOnehot = IndexToOnehot(label, vSizeTgt, 0);
        lossTensor = CrossEntropy(output, labelOnehot, paddingDec);
        float lossBatch = ReduceSumAllValue(lossTensor);

        /* dump the test result */
        for (int s = 0; s < bSize; s++) {
            DTYPE sum = 0;
            int* seq = seqs + s * length;
            for (int i = 0; i < length; i++) {
                if (seq[i] >= 0) {
                    fprintf(ofile, "%d ", seq[i]);
                }
                else
                    break;
            }
            fprintf(ofile, "||| ");
            for (int i = 0; i < length; i++) {
                if (seq[i] >= 0) {
                    DTYPE p = lossTensor.Get2D(s, i);
                    fprintf(ofile, "%.3e ", p);
                    sum += p;
                }
                else
                    break;
            }
            fprintf(ofile, "||| %e\n", sum);
        }

        loss += lossBatch;

        wordCount += wc;
        sentCount += bSize;
    }

    fclose(file);
    fclose(ofile);

    delete[] seqs;

    double elapsed = GetClockSec() - startT;

    XPRINT5(0, stderr, "[INFO] test finished (took %.1fs, sentence=%d, word=%d, loss=%.3f and ppl=%.3f)\n",
        elapsed, sentCount, wordCount, loss / wordCount, exp(loss / wordCount));
}

/*
make a checkpoint
>> model - the model
>> validFN - validation data file
>> modelFN - model data file
>> label - label of the model
>> id - id of the checkpoint
*/
void T2TTrainer::MakeCheckpoint(T2TModel* model, const char* validFN, const char* modelFN, const char* label, int id)
{
    fprintf(stderr, "make a checkpoint\n");
    char* fn = new char[MAX_LINE_LENGTH];
    sprintf(fn, "%s.%s.%03d", modelFN, label, id);
    model->Dump(fn);
    delete[] fn;

    char* fn2 = new char[MAX_LINE_LENGTH];
    sprintf(fn2, "%s.%s.%03d.output", modelFN, label, id);
    if (validFN != NULL) {
        T2TTrainer trainer;
        trainer.Init(*cfg);
        trainer.Validate(validFN, fn2, model);
    }
    delete[] fn2;
}

/*
update the model by delta rule
\theta_{new} = \theta - \lrate * grad
where
\lrate = d^-0.5 * min(stepNum^{-0.5}, stepNum * warmupStepNum^{-1.5})
>> model - the t2t model
>> lr - learning rate
*/
void T2TTrainer::Update(T2TModel* model, const float lr)
{
    TensorList ws(100);

    model->GetParams(ws);

    for (int i = 0; i < ws.Size(); i++) {
        XTensor* para = ws[i];
        XTensor* paraGrad = para->grad;

        if (paraGrad == NULL)
            continue;

        CheckNTErrors(para != NULL, "NULL parameter tensor!");
        CheckNTErrors(paraGrad != NULL, "NULL gradient tensor!");

        if (useAdam) {
            adamBeta1T *= adamBeta1;
            adamBeta2T *= adamBeta2;
            DTYPE e = lr * (DTYPE)sqrt(1 - adamBeta2T) / (1 - adamBeta1T);
            DTYPE d = adamDelta * (DTYPE)sqrt(1 - adamBeta2T);

            /* m = beta_1 * m + (1-beta_1) * grad */
            XTensor* m = (XTensor*)moments.Get(i);
            _ScaleAndShiftMe(m, adamBeta1, 0);
            _Sum(m, paraGrad, m, (1.0F - adamBeta1));

            /* v = beta_2 * v + (1-beta_2) * grad * grad*/
            XTensor* v = (XTensor*)moments2nd.Get(i);
            _Multiply(paraGrad, paraGrad, v, adamBeta2 / (1.0F - adamBeta2));
            _ScaleAndShiftMe(v, (1.0F - adamBeta2), 0);

            /* v2 = m / (sqrt(v) + delta) */
            XTensor* v2 = NewTensorBuf(v, v->devID);
            _Power(v, v2, 0.5F);
            _ScaleAndShiftMe(v2, 1.0F, d);
            _Div(m, v2, v2);

            /* the delta rule */
            _Sum(para, v2, para, -e);

            DelTensorBuf(v2);
        }
        else {
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
void T2TTrainer::PrepareModel(T2TModel* model)
{
    moments.Clear();
    moments2nd.Clear();

    TensorList ws(100);

    model->GetParams(ws);

    for (int i = 0; i < ws.Size(); i++) {
        XTensor* para = ws[i];
        XNoder::MakeGrad(para);

        if (useAdam) {
            XTensor* m = new XTensor(para);
            XTensor* m2 = new XTensor(para);
            m->SetZeroAll();
            m2->SetZeroAll();
            moments.Add(m);
            moments2nd.Add(m2);
        }
    }

    adamBeta1T = 1.0F;
    adamBeta2T = 1.0F;
}

}