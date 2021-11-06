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
 * $Created by: HU Chi (huchinlp@gmail.com) 2021-06
 */

#include <cstdlib>
#include <algorithm>

#include "TrainDataSet.h"


/* the nmt namespace */
namespace nmt {

/* shuffle buckets by their keys */
void TrainDataSet::ShuffleBuckets() {

    /* assign random keys for buckets */
    int keyIdx = 0;
    std::random_shuffle(randomKeys.items, randomKeys.items + randomKeys.Size());

    while (bufIdx < buf->Size()) {
        int bucketKey = ((Sample*)(buf->Get(bufIdx)))->bucketKey;
        while ((bufIdx < buf->Size()) &&
            (((Sample*)(buf->Get(bufIdx)))->bucketKey == bucketKey)) {
            ((Sample*)(buf->Get(bufIdx)))->bucketKey = randomKeys[keyIdx];
            bufIdx++;
        }
        keyIdx++;
    }

    /* sort buckets by the previous keys */
    sort(buf->items, buf->items + buf->count,
        [](void* a, void* b) {
            return ((Sample*)(a))->bucketKey <
                ((Sample*)(b))->bucketKey;
        });

    /* reset the buffer index */
    bufIdx = 0;
}

/*
load samples from a file into the buffer
*/
bool TrainDataSet::LoadBatchToBuf()
{
    int n = 0;

    while (n < config->common.bufSize) {
        Sample* sample = LoadSample();
        buf->Add(sample);
        n++;
    }
    LOG("loaded %d samples", n);

    /* group samples into buckets */
    SortByTgtLengthAscending();
    SortBySrcLengthAscending();

    /* build buckets for training */
    if (isTraining)
        BuildBucket();

    return true;
}

/*
load a mini-batch to a device
>> inputs - the list to store input tensors
>> golds - the list to store gold tensors
*/
bool TrainDataSet::GetBatchSimple(XList* inputs, XList* golds)
{
    if (bufIdx == buf->Size()) {
        if (isTraining)
            ShuffleBuckets();
        else
            bufIdx = 0;
    }

    wc = 0;
    sc = isTraining ? GetBucket() : config->common.sBatchSize;
    sc = MIN(sc, buf->Size() - bufIdx);

    /* get the maximum sentence length in a mini-batch */
    int maxSrcLen = MaxSrcLen(bufIdx, bufIdx + sc);
    int maxTgtLen = MaxTgtLen(bufIdx, bufIdx + sc);

    CheckNTErrors(maxSrcLen > 0, "Invalid source length for batching");
    CheckNTErrors(maxTgtLen > 0, "Invalid target length for batching");

    int* batchEncValues = new int[sc * maxSrcLen];
    float* paddingEncValues = new float[sc * maxSrcLen];

    int* labelVaues = new int[sc * maxTgtLen];
    int* batchDecValues = new int[sc * maxTgtLen];
    float* paddingDecValues = new float[sc * maxTgtLen];

    for (int i = 0; i < sc * maxSrcLen; i++) {
        batchEncValues[i] = config->model.pad;
        paddingEncValues[i] = 1.0F;
    }
    for (int i = 0; i < sc * maxTgtLen; i++) {
        batchDecValues[i] = config->model.pad;
        labelVaues[i] = config->model.pad;
        paddingDecValues[i] = 1.0F;
    }

    int curSrc = 0;
    int curTgt = 0;

    /*
    batchEnc: end with EOS (left padding)
    batchDec: begin with SOS (right padding)
    label:    end with EOS (right padding)
    */
    for (int i = 0; i < sc; ++i) {

        Sample* sample = (Sample*)(buf->Get(bufIdx + i));
        wc += int(sample->tgtSeq->Size());

        curSrc = maxSrcLen * i;
        for (int j = 0; j < int(sample->srcSeq->Size()); j++)
            batchEncValues[curSrc++] = sample->srcSeq->Get(j);

        curTgt = maxTgtLen * i;
        for (int j = 0; j < int(sample->tgtSeq->Size()); j++) {
            if (j > 0)
                labelVaues[curTgt - 1] = sample->tgtSeq->Get(j);
            batchDecValues[curTgt++] = sample->tgtSeq->Get(j);
        }
        labelVaues[curTgt - 1] = config->model.eos;
        while (curSrc < maxSrcLen * (i + 1))
            paddingEncValues[curSrc++] = 0;
        while (curTgt < maxTgtLen * (i + 1))
            paddingDecValues[curTgt++] = 0;
    }

    XTensor * batchEnc = ((TensorList*)(inputs))->Get(0);
    XTensor * paddingEnc = ((TensorList*)(inputs))->Get(1);
    XTensor * batchDec = ((TensorList*)(golds))->Get(0);
    XTensor * paddingDec = ((TensorList*)(golds))->Get(1);
    XTensor * label = ((TensorList*)(golds))->Get(2);

    InitTensor2D(batchEnc, sc, maxSrcLen, X_INT);
    InitTensor2D(paddingEnc, sc, maxSrcLen, X_FLOAT);
    InitTensor2D(batchDec, sc, maxTgtLen, X_INT);
    InitTensor2D(paddingDec, sc, maxTgtLen, X_FLOAT);
    InitTensor2D(label, sc, maxTgtLen, X_INT);

    bufIdx += sc;

    batchEnc->SetData(batchEncValues, batchEnc->unitNum);
    paddingEnc->SetData(paddingEncValues, paddingEnc->unitNum);
    batchDec->SetData(batchDecValues, batchDec->unitNum);
    paddingDec->SetData(paddingDecValues, paddingDec->unitNum);
    label->SetData(labelVaues, label->unitNum);

    delete[] batchEncValues;
    delete[] paddingEncValues;
    delete[] batchDecValues;
    delete[] paddingDecValues;
    delete[] labelVaues;

    return true;
}

/* group samples with similar length into buckets */
void TrainDataSet::BuildBucket()
{
    int idx = 0;

    /* build buckets by the length of source and target sentences */
    while (idx < int(buf->Size())) {

        /* sentence number in a bucket */
        int sentNum = 1;

        /* get the maximum source sentence length in a bucket */
        int maxSrcLen = MaxSrcLen(idx, idx + sentNum);
        int maxTgtLen = MaxTgtLen(idx, idx + sentNum);
        int maxLen = MAX(maxSrcLen, maxTgtLen);

        /* the maximum sentence number in a bucket */
        const int MAX_SENT_NUM = 5120;

        while ((sentNum < (buf->count - idx))
            && (sentNum < MAX_SENT_NUM)
            && (sentNum * maxLen <= config->common.bucketSize)) {
            sentNum++;
            maxSrcLen = MaxSrcLen(idx, idx + sentNum);
            maxTgtLen = MaxTgtLen(idx, idx + sentNum);
            maxLen = MAX(maxSrcLen, maxTgtLen);
        }

        /* make sure the number is valid */
        if ((sentNum) * maxLen > config->common.bucketSize || sentNum >= MAX_SENT_NUM) {
            sentNum--;
            sentNum = max(8 * (sentNum / 8), sentNum % 8);
        }
        if ((int(buf->Size()) - idx) < sentNum)
            sentNum = int(buf->Size()) - idx;

        /* assign the same key for items in a bucket */
        for (int i = 0; i < sentNum; i++)
            ((Sample*)(buf->Get(idx + i)))->bucketKey = idx;

        randomKeys.Add(idx);
        idx += sentNum;
    }

    /* shuffle buckets */
    ShuffleBuckets();
}

/* find the sentences in a bucket */
inline int TrainDataSet::GetBucket()
{
    int sent = 0;
    int bucketKey = ((Sample*)(buf->Get(bufIdx)))->bucketKey;

    while ((sent < (int(buf->Size()) - bufIdx)) &&
          (((Sample*)(buf->Get(bufIdx + sent)))->bucketKey == bucketKey)) {
        sent++;
    }

    sent = MIN(sent, (int(buf->Size()) - bufIdx));
    CheckNTErrors(sent > 0, "Invalid batch size");

    return sent;
}

/* start the process */
bool TrainDataSet::Start()
{
    return false;
}

/* end the process */
bool TrainDataSet::End()
{
    return true;
}

/* load a sample from the training file */
Sample* TrainDataSet::LoadSample()
{
    int srcLen;
    int tgtLen;

    fread(&srcLen, sizeof(srcLen), 1, fp);
    fread(&tgtLen, sizeof(tgtLen), 1, fp);

    CheckNTErrors(srcLen > 0, "Invalid source sentence length");
    CheckNTErrors(tgtLen > 0, "Invalid target sentence length");

    IntList* srcSent = new IntList(srcLen);
    IntList* tgtSent = new IntList(tgtLen);
    srcSent->ReadFromFile(fp, srcLen);
    tgtSent->ReadFromFile(fp, tgtLen);

    Sample* sample = new Sample(srcSent, tgtSent, 0);
    
    return sample;
}

/*
the constructor of TrainDataSet
>> cfg - the configuration of NMT system
>> isTrainDataset - indicates whether it is used for training
*/
void TrainDataSet::Init(NMTConfig& cfg, bool isTrainDataset)
{
    bufIdx = 0;
    config = &cfg;
    isTraining = isTrainDataset;

    if (isTraining)
        fp = fopen(config->training.trainFN, "rb");
    else
        fp = fopen(config->training.validFN, "rb");
    CheckNTErrors(fp, "Failed to open the training/validation file");

    /* skip the meta information */
    int meta_info[6];
    fread(meta_info, sizeof(*meta_info), 6, fp);

    /* load the number of training samples */
    fread(&sampleNum, sizeof(sampleNum), 1, fp);
    CheckNTErrors(sampleNum > 0, "There is no training/validation data");

    /* reset the buffer size */
    config->common.bufSize = sampleNum;

    LoadBatchToBuf();
}

/* de-constructor */
TrainDataSet::~TrainDataSet()
{
    fclose(fp);
}

} /* end of the nmt namespace */