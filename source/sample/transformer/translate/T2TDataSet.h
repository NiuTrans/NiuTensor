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
 * $Created by: HU Chi (huchinlp@foxmail.com) 2019-04-03
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-06
 */

#ifndef __DATASET_H__
#define __DATASET_H__

#include <cstdio>
#include <vector>
#include <fstream>
#include "T2TVocab.h"

#include "../../../tensor/XList.h"
#include "../../../tensor/XTensor.h"
#include "../../../tensor/XGlobal.h"

#define MAX_WORD_NUM 120

using namespace std;

namespace nts {
/* the struct of tokenized input */
struct Example {
    int id;
    IntList values;
};

/* the struct of tokenized output */
struct Result {
    int id;
    IntList res;
};

/* A `DataSet` is associated with a file which contains variable length data.*/
struct DataSet {
public:
    /* the data buffer */
    InputBufferType inputBuffer;

    /* a list of empty line number */
    IntList emptyLines;

    /* the result buffer */
    OutputBufferType outputBuffer;

    /* the pointer to file stream */
    ifstream* fp;

    /* size of used data in buffer */
    size_t bufferUsed;

    /* the source vocabulary */
    Vocab srcVocab;

    /* the target vocabulary */
    Vocab tgtVocab;

public:

    /* sort the input by length */
    void SortInput();

    /* reorder the output by ids */
    void SortOutput();

    /* load data from a file to the buffer */
    void LoadDataToBuffer();

    /* generate a mini-batch */
    UInt64List LoadBatch(XTensor* batchEnc, XTensor* paddingEnc,
        size_t sBatch, size_t wBatch, int devID);

    /* initialization function */
    void Init(const char* dataFile, const char* srcVocabFN, const char* tgtVocabFN);

    /* check if the buffer is empty */
    bool IsEmpty();

    /* dump the translations to a file */
    void DumpRes(const char* ofn);

    /* de-constructor */
    ~DataSet();
};
}

#endif // __DATASET_H__