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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2019-03-27
 * A week with no trips :)
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-06
 */

#ifndef __T2TTESTER_H__
#define __T2TTESTER_H__

#include "T2TSearch.h"
#include "T2TDataSet.h"

namespace transformer
{

/* This class translates test sentences with a trained model. */
class T2TTranslator
{
public:
    /* vocabulary size of the source side */
    int vSize;

    /* vocabulary size of the target side */
    int vSizeTgt;

    /* batch size for sentences */
    int sentBatch;

    /* batch size for words */
    int wordBatch;

    /* beam size */
    int beamSize;

    /* for batching */
    DataSet batchLoader;

    /* decoder for inference */
    void* seacher;

public:
    /* constructor */
    T2TTranslator();

    /* de-constructor */
    ~T2TTranslator();

    /* initialize the model */
    void Init(T2TConfig& config);

    /* test the model */
    void Translate(const char* ifn, const char* vfn, const char* ofn, 
                   const char* tfn, T2TModel* model);

    /* dump the result into the file */
    void Dump(FILE* file, XTensor* output);
};

}

#endif