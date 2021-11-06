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
 * We define the base class of datasets for NMT here.
 * It will be overrided for training and translation.
 * 
 * $Created by: HU Chi (huchinlp@gmail.com) 2021-06
 */

#ifndef __DATASET_H__
#define __DATASET_H__

#include "Config.h"
#include "../../train/XBaseTemplate.h"

using namespace std;

/* the nmt namespace */
namespace nmt {

/* the training or test sample (a pair of sequences) */
struct Sample {

    /* index of the sentence pair */
    int index;

    /* the key used to shuffle buckets */
    int bucketKey;

    /* source sequence (a list of tokens) */
    IntList * srcSeq;

    /* target sequence (a list of tokens) */
    IntList * tgtSeq;

    /* constructor */
    Sample(IntList* s, IntList* t = NULL, int myKey = -1);

    /* de-constructor */
    ~Sample();
};

/* The base class of datasets used in NiuTrans.NMT. */
class DataSetBase : public DataDistributeBase
{
public:
    /* word-counter */
    int wc;

    /* sentence-counter */
    int sc;

    /* current index of the buffer */
    int bufIdx;

    /* the buffer (a list) of sequences */
    XList * buf;

    /* the configuration of NMT system */
    NMTConfig* config;

public:
    /* get the maximum source sentence length in a range of buffer */
    int MaxSrcLen(int begin, int end);

    /* get the maximum target sentence length in a range of buffer */
    int MaxTgtLen(int begin, int end);

    /* sort the input by source sentence length (in ascending order) */
    void SortBySrcLengthAscending();

    /* sort the input by target sentence length (in ascending order) */
    void SortByTgtLengthAscending();

    /* sort the input by source sentence length (in descending order) */
    void SortBySrcLengthDescending();

    /* sort the input by target sentence length (in descending order) */
    void SortByTgtLengthDescending();

    /* release the samples in a buffer */
    void ClearBuf();

public:
    /* constructor */
    DataSetBase();

    /* load the samples into the buffer (a list) */
    virtual
    bool LoadBatchToBuf() = 0;

    /* initialization function */
    virtual
    void Init(NMTConfig& myConfig, bool isTraining) = 0;

    /* load a sample from the file stream  */
    virtual
    Sample* LoadSample() = 0;

    /* load a mini-batch from the buffer */
    virtual
    bool GetBatchSimple(XList* inputs, XList* golds = NULL) = 0;

    /* de-constructor */
    ~DataSetBase();
};

} /* end of the nmt namespace */

#endif /* __DATASET_H__ */