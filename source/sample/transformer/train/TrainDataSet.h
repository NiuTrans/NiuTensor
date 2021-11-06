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
 * Here we define the data manager for NMT training.
 * Loading the training data requires 4 steps:
 * 1. initialize the dataset class (Init)
 * 2. load samples from the training file (LoadBatchToBuf)
 * 3. build and shuffle bucktes of batches (ShuffleBuckets)
 * 4. load a mini-batch from the buckets (GetBatchSimple)
 * 
 * $Created by: HU Chi (huchinlp@gmail.com) 2021-06
 */

#ifndef __TRAINDATASET_H__
#define __TRAINDATASET_H__

#include "../Config.h"
#include "../DataSet.h"

using namespace std;
using namespace nts;

/* the nmt namespace */
namespace nmt { 

/* The base class of datasets used in the NMT system. */
struct TrainDataSet : public DataSetBase
{
private:
    /* indicates whether it is used for training or validation */
    bool isTraining;

    /* the pointer to file stream */
    FILE* fp;

    /* a list of random keys */
    IntList randomKeys;

private:

    /* sort buckets by their keys */
    void ShuffleBuckets();

    /* group data into buckets with similar length */
    void BuildBucket();

    /* calculate the batch size according to the number of tokens */
    int GetBucket();

    /* load a pair of sequences from the file  */
    Sample* LoadSample() override;

    /* load the samples into the buffer (a list) */
    bool LoadBatchToBuf() override;

public:

    /* number of samples in the dataset */
    int sampleNum;

    /* start the process */
    bool Start();

    /* end the process */
    bool End();

    /* initialization function */
    void Init(NMTConfig& config, bool isTrainDataset) override;

    /* load the samples into tensors from the buffer */
    bool GetBatchSimple(XList* inputs, XList* golds) override;

    /* de-constructor */
    ~TrainDataSet();
};

} /* end of the nmt namespace */

#endif /* __DATASET_H__ */