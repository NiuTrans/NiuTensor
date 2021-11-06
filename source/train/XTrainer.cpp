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
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-02-23
*
*/

#include "XTrainer.h"
#include "XLearningRate.h"

/* the nts (NiuTrans.Tensor) namespace */
namespace nts {

/* constructor */
XTrainer::XTrainer()
{
}

/* de-constructor */
XTrainer::~XTrainer()
{
}

/* 
get the device ids of the jobs 
>> config - configuration
>> ids - the array of device ids
>> num - number of the jobs
>> maxDevNum - the maximum number of devices
*/
void XTrainer::GetDevIDs(XConfig * config, int * ids, int & num, int maxDevNum)
{
    CheckNTErrors(maxDevNum > 0, "No data array for input!");

    num = 0;
    for (int i = 0; i < maxDevNum; i++) {
        char dev[16];
        sprintf(dev, "jobdev%d", i);
        int id = config->GetInt(dev, -128);
        if (id != -128) {
            ids[num++] = id;
        }
        else
            break;
    }

    if (num == 0) {
        char dev[16];
        sprintf(dev, "jobdev");
        int id = config->GetInt(dev, -128);
        if (id != -128)
            ids[num++] = id;
    }

    if (num == 0) {
        char dev[16];
        sprintf(dev, "dev");
        int id = config->GetInt(dev, -128);
        if (id != -128)
            ids[num++] = id;
    }
}

/*
run the trainer (this is the core process)
>> config - configuration
>> dataDistributor - the data distributor that generates an input for the net each time
>> model - the neural network
>> optimizer - the optimizer
*/
void XTrainer::Run(XConfig * config, DataDistributeBase * dataDistributor,
    XModel * model, XOptimizer * optimizer)
{
    CheckNTErrors(config != NULL, "No input config!");
    CheckNTErrors(dataDistributor != NULL, "No input data distributor!");
    CheckNTErrors(model != NULL, "No input neural network!");

    int epoch = 0;
    int step = 0;
    int stepAll = 0;
    int jobNum = 0;
    int accumulation = config->GetInt("accumulation", 1);
    int nwarmup = config->GetInt("nwarmup", 0);
    float lrate = optimizer->GetLearningRate();

    CheckNTErrors(accumulation >= 1, "accumulation must be larger than 0!");

    int * ids = new int[MAX_DEVICE_NUM_TRAINING];
    GetDevIDs(config, ids, jobNum, MAX_DEVICE_NUM_TRAINING);

    optimizer->ShowSettings();
    this->ShowSettings(config);

    /* create the server and workers */
    XLeaderPS leader;
    leader.MakeAll(config, model, ids, jobNum);
    //leader.SetInstantRun();
    leader.Start();

    /* learning rate scheduler */
    XLearningRate LRScheduler;

    double startT = GetClockSec();

    XPRINT(1, stderr, "[INFO] Initializing the model ... [DONE]\n");

    /* train the model */
    for (epoch = 0; epoch < optimizer->nepoch; epoch++) {

        bool ok = true;
        dataDistributor->Start();

        while (ok) {
            if (++stepAll % accumulation == 0) {

                /* learning rate scheduling */
                if (nwarmup > 0)
                    optimizer->SetLearningRate(LRScheduler.MakeLRTransformer(lrate, step + 1, nwarmup, 1e-7F));

                /* one step of udpate */
                ok = leader.Run(config, dataDistributor, optimizer);

                float loss = leader.GetLoss() / leader.GetSampleNum();

                if ((step + 1) % 100 == 0)
                    XPRINT5(1, stderr, "[INFO] elapsed=%.1fs epoch:%d step:%d sample:%d loss:%f\n",
                        GetClockSec() - startT, epoch + 1, step + 1, leader.GetSampleNum(), loss);

                leader.ResetParamGrad();

                if (++step >= optimizer->nstep)
                    break;
            }
            else {
                /* one step with no udpate */
                ok = leader.Run(config, dataDistributor, NULL);
            }
        }

        dataDistributor->End();

        if (step >= optimizer->nstep)
            break;
    }

    delete[] ids;

    XPRINT(1, stderr, "[INFO] Training Finished[DONE]\n");
}

/* show settings of training */
void XTrainer::ShowSettings(XConfig* config)
{
    int workerNum = 0;
    int* ids = new int[MAX_DEVICE_NUM_TRAINING];

    GetDevIDs(config, ids, workerNum, MAX_DEVICE_NUM_TRAINING);

    XPRINT(1, stderr, "[INFO] Training Setup:\n");
    XPRINT2(1, stderr, "%25s = %d\n", "nworker", workerNum);

    if (workerNum > 0) {
        if (ids[0] < 0) {
            XPRINT2(1, stderr, "%25s = CPU[%d]\n", "worker0(server)", ids[0]);
        }
        else{
            XPRINT2(1, stderr, "%25s = GPU[%d]\n", "worker0(server)", ids[0]);
        }

        for (int i = 1; i < workerNum; i++) {
            char name[32];
            sprintf(name, "worker%d", i);
            if (ids[i] < 0) {
                XPRINT2(1, stderr, "%25s = CPU[%d]\n", name, ids[i]);
            }
            else {
                XPRINT2(1, stderr, "%25s = GPU[%d]\n", name, ids[i]);
            }
        }
    }

    XPRINT2(1, stderr, "%25s = %d\n", "accumulation", config->GetInt("accumulation", 1));

    delete[] ids;
}

} /* end of the nts (NiuTrans.Tensor) namespace */
