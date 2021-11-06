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
 * Here we define the batch manager for translation.
 * 
 * $Created by: HU Chi (huchinlp@gmail.com) 2021-06
 */

#ifndef __TRANSLATEDATASET_H__
#define __TRANSLATEDATASET_H__

#include <string>
#include <fstream>
#include "Vocab.h"
#include "../DataSet.h"

using namespace std;

/* the nmt namespace */
namespace nmt {

/* The translation batch manager for NMT. */
class TranslateDataset : public DataSetBase {
public:

    /* whether append an empty line to the buffer */
    bool appendEmptyLine;

    /* the indices of empty lines */
    IntList emptyLines;

    /* the source vocabulary */
    Vocab srcVocab;

    /* the target vocabulary */
    Vocab tgtVocab;

    /* the input file stream */
    istream* ifp;

public:
    /* check if the buffer is empty */
    bool IsEmpty();

    /* initialization function */
    void Init(NMTConfig& myConfig, bool notUsed) override;

    /* load a sample from the buffer */
    Sample* LoadSample() override;

    /* transfrom a line to a sequence */
    Sample* LoadSample(string line);

    /* load the samples into tensors from the buffer */
    bool GetBatchSimple(XList* inputs, XList* info) override;

    /* load the samples into the buffer (a list) */
    bool LoadBatchToBuf() override;

    /* constructor */
    TranslateDataset();

    /* de-constructor */
    ~TranslateDataset();
};

} /* end of the nmt namespace */

#endif /* __TRANSLATEDATASET_H__ */