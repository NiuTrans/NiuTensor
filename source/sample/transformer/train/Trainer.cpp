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

#include "Trainer.h"
#include "../Config.h"
#include "../../../network/XNoder.h"
#include "../../../tensor/XUtility.h"
#include "../../../tensor/core/CHeader.h"
#include "../../../tensor/loss/LHeader.h"

/* the nmt namespace */
namespace nmt
{

/* constructor */
Trainer::Trainer()
{
    model = NULL;
    config = NULL;
    adamBeta1T = 0.0F;
    adamBeta2T = 0.0F;
}

/* de-constructor */
Trainer::~Trainer()
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
initialize the trainer
>> myModel - the model for training
>> myConfig - configurations of the NMT system
*/
void Trainer::Init(NMTConfig& myConfig, NMTModel& myModel)
{
    model = &myModel;
    config = &myConfig;
}

/*
train the model
>> fn - training data file
>> validFN - validation data file
>> modelFN - where we keep the model
*/
void Trainer::Run()
{
    double startT = GetClockSec();

    int step = 0;
    int epoch = 0;
    int nSkipped = 0;
    int gradStep = 0;
    int validStep = 0;
    int wordCount = 0;
    int nStepCheck = 0;
    int wordCountTotal = 0;
    int batchCountTotal = 0;
    int devID = model->devID;
    float lr = 0.0F;
    float loss = 0.0F;
    bool isEnd = false;

    PrepareModel();
    
    model->SetTrainingFlag(true);
    trainBatchLoader.Init(*config, true);
    validBatchLoader.Init(*config, false);

    for (epoch = 1; epoch <= config->training.nepoch; epoch++) {

        loss = 0.0F;
        wordCount = 0;
        int sentCount = 0;

        while (sentCount < trainBatchLoader.sampleNum) {

            XNet net;
            net.Clear();

            /* batch of sequences */
            XTensor batchEnc;
            XTensor batchDec;

            /* labels */
            XTensor label;

            /* padding */
            XTensor paddingEnc;
            XTensor paddingDec;

            /* the inputs and golden labels */
            TensorList inputs;
            TensorList golds;

            inputs.Add(&batchEnc);
            inputs.Add(&paddingEnc);
            golds.Add(&batchDec);
            golds.Add(&paddingDec);
            golds.Add(&label);

            /* load a mini-batch */
            trainBatchLoader.GetBatchSimple((XList*)(&inputs), (XList*)(&golds));

            /* flush the batch to the target device */
            batchEnc.SetDevice(model->devID);
            paddingEnc.SetDevice(model->devID);
            batchDec.SetDevice(model->devID);
            paddingDec.SetDevice(model->devID);
            label.SetDevice(model->devID);

            CheckNTErrors(batchEnc.order == 2, "Wrong tensor order of the sequence batch");

            /* output probabilities */
            XTensor output;

            /* make the network */
            output = model->MakeMT(batchEnc, batchDec, paddingEnc, paddingDec);

            /* get loss and probabilities */
            XTensor labelOnehot;
            XTensor lossTensor;

            labelOnehot = IndexToOnehot(label, config->model.tgtVocabSize, config->training.labelSmoothingP);

            lossTensor = CrossEntropy(output, labelOnehot, paddingDec);

            float lossBatch = ReduceSumAllValue(lossTensor);

            DTYPE lossLocal = lossBatch / trainBatchLoader.wc;
            bool doUpdate = (!IsNAN(lossLocal) && !IsINF(lossLocal) && lossLocal < 1e3F);

            sentCount += trainBatchLoader.sc;
            wordCount += trainBatchLoader.wc;
            wordCountTotal += trainBatchLoader.wc;
            batchCountTotal += trainBatchLoader.sc;

            if (doUpdate) {

                /* back-propagation */
                net.Backward(lossTensor);

                if (model->encoder->useHistory)
                    model->encoder->history->ClearHistory();
                if (model->decoder->useHistory)
                    model->decoder->history->ClearHistory();

                gradStep += 1;
                loss += lossBatch;

                /* update the parameters */
                if (gradStep == config->training.updateFreq) {

                    lr = LRScheduler.MakeLRTransformer(config->training.lrate, step, 
                         config->training.nwarmup, config->training.warmupInitLR);

                    if (lr <= config->training.minLR) {
                        isEnd = true;
                        break;
                    }

                    /* model update */
                    Update(lr);

                    gradStep = 0;
                    validStep++;
                }
            }
            else
                nSkipped++;

            /* logging */
            if (step > 0 && step % config->common.logInterval == 0) {
                double elapsed = GetClockSec() - startT;
                LOG("elapsed=%.1fs, step=%d, epoch=%d, "
                    "total word=%d, total sent=%d, loss=%.3f, ppl=%.3f, lr=%.6e", 
                    elapsed, step, epoch, wordCountTotal, batchCountTotal,
                    loss / wordCount / log(2.0), exp(loss / wordCount), lr);
                
                if (!doUpdate)
                    XPRINT(0, stderr, " (no update)");
            }

            /* save the internal checkpoint */
            if (config->training.saveFreq > 0 && ++nStepCheck >= config->training.saveFreq) {
                MakeCheckpoint("step", step);
                nStepCheck = 0;
            }

            /* reach the maximum training step */
            if (++step >= config->training.nstep) {
                isEnd = true;
                break;
            }
        }

        LOG("end of epoch %d", epoch);

        /* end of training */
        if (isEnd)
            break;

        /* save the checkpoint every epoch */
        int checkpointID = epoch % config->training.ncheckpoint;
        MakeCheckpoint("epoch", checkpointID);
    }

    /* final logging */
    double elapsed = GetClockSec() - startT;
    epoch = MIN(epoch, config->training.nepoch);
    LOG("training finished (took %.1fs, step=%d, skipped=%d and epoch=%d)", elapsed, step, nSkipped, epoch);

    /* save the final model */
    LOG("saving the final model");
    model->DumpToFile(config->common.modelFN);
}

/*
test the model
>> fn - test data file
>> ofn - output data file
>> model - model that is trained
*/
void Trainer::Validate()
{
    double startT = GetClockSec();

    int wordCount = 0;
    int sentCount = 0;
    float loss = 0;

    while (sentCount < validBatchLoader.sampleNum) {
        /* batch of sequences */
        XTensor batchEnc;
        XTensor batchDec;

        /* labels */
        XTensor label;

        /* padding */
        XTensor paddingEnc;
        XTensor paddingDec;

        /* the inputs and golden labels */
        TensorList inputs;
        TensorList golds;

        inputs.Add(&batchEnc);
        inputs.Add(&paddingEnc);
        golds.Add(&batchDec);
        golds.Add(&paddingDec);
        golds.Add(&label);

        /* load a mini-batch */
        validBatchLoader.GetBatchSimple((XList*)(&inputs), (XList*)(&golds));

        /* flush the batch to the target device */
        batchEnc.FlushToDevice(model->devID);
        paddingEnc.FlushToDevice(model->devID);
        batchDec.FlushToDevice(model->devID);
        paddingDec.FlushToDevice(model->devID);
        label.FlushToDevice(model->devID);

        /* output probabilities */
        XTensor output;

        /* make the network */
        output = model->MakeMT(batchEnc, batchDec, paddingEnc, paddingDec);

        /* get loss and probabilities */
        XTensor labelOnehot;
        XTensor lossTensor;

        labelOnehot = IndexToOnehot(label, config->model.tgtVocabSize, 0.0F);

        lossTensor = CrossEntropy(output, labelOnehot, paddingDec);

        float lossBatch = ReduceSumAllValue(lossTensor);

        loss += lossBatch;

        wordCount += validBatchLoader.wc;
        sentCount += validBatchLoader.sc;

        if (model->encoder->useHistory)
            model->encoder->history->ClearHistory();
        if (model->decoder->useHistory)
            model->decoder->history->ClearHistory();
    }

    double elapsed = GetClockSec() - startT;

    LOG("validating finished (took %.1fs, sentence=%d, word=%d, loss=%.3f and ppl=%.3f)",
        elapsed, sentCount, wordCount, loss / wordCount / log(2.0), exp(loss / wordCount));
}

/*
make a checkpoint
>> label - label of the model
>> id - id of the checkpoint
*/
void Trainer::MakeCheckpoint(const char* label, int id)
{
    /* disable gradient flow during validating */
    DISABLE_GRAD;
    model->SetTrainingFlag(false);
    Validate();

    LOG("make a checkpoint");
    char* fn = new char[MAX_LINE_LENGTH];
    sprintf(fn, "%s.%s.%03d", config->common.modelFN, label, id);
    model->DumpToFile(fn);

    /* enable gradient flow after validating */
    ENABLE_GRAD;
    model->SetTrainingFlag(true);
    delete[] fn;
}

/*
update the model by delta rule
\theta_{new} = \theta - \lrate * grad
where
\lrate = d^-0.5 * min(stepNum^{-0.5}, stepNum * warmupStepNum^{-1.5})
>> lr - learning rate
*/
void Trainer::Update(const float lr)
{
    TensorList ws;
    model->GetParams(ws);

    for (int i = 0; i < ws.Size(); i++) {
        XTensor* para = ws[i];
        XTensor* paraGrad = para->grad;

        if (paraGrad == NULL)
            continue;

        CheckNTErrors(para != NULL, "NULL parameter tensor!");
        CheckNTErrors(paraGrad != NULL, "NULL gradient tensor!");

        if (config->training.useAdam) {
            adamBeta1T *= config->training.adamBeta1;
            adamBeta2T *= config->training.adamBeta2;
            float e = lr * sqrtf(1.0F - adamBeta2T) / (1.0F - adamBeta1T);
            float d = config->training.adamDelta * sqrtf(1.0F - adamBeta2T);

            /* m = beta_1 * m + (1-beta_1) * grad */
            XTensor* m = (XTensor*)moments.Get(i);
            _ScaleAndShiftMe(m, config->training.adamBeta1, 0);
            _Sum(m, paraGrad, m, (1.0F - config->training.adamBeta1));

            /* v = beta_2 * v + (1-beta_2) * grad * grad*/
            XTensor* v = (XTensor*)moments2nd.Get(i);
            _Multiply(paraGrad, paraGrad, v, config->training.adamBeta2 / (1.0F - config->training.adamBeta2));
            _ScaleAndShiftMe(v, (1.0F - config->training.adamBeta2), 0);

            /* v2 = m / (sqrt(v) + delta) */
            GMems.GetMem(v->devID)->LockBuf();
            XTensor* v2 = NewTensorBufV2(v, v->devID, v->mem);
            _Power(v, v2, 0.5F);
            _ScaleAndShiftMe(v2, 1.0F, d);
            _Div(m, v2, v2);

            /* the delta rule */
            _Sum(para, v2, para, -e);

            DelTensorBuf(v2);
            GMems.GetMem(v->devID)->UnlockBuf();
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
*/
void Trainer::PrepareModel()
{
    moments.Clear();
    moments2nd.Clear();

    TensorList ws;

    model->GetParams(ws);

    for (int i = 0; i < ws.Size(); i++) {
        XTensor* para = ws[i];
        XNoder::MakeGrad(para);

        if (config->training.useAdam) {
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

} /* end of the nmt namespace */