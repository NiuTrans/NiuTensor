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
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04, 2020-06
 */

#include <cmath>

#include "T2TTranslator.h"
#include "T2TSearch.h"
#include "../module/T2TUtility.h"
#include "../../../tensor/XTensor.h"
#include "../../../tensor/XUtility.h"
#include "../../../tensor/core/CHeader.h"

using namespace nts;

namespace transformer
{

/* constructor */
T2TTranslator::T2TTranslator()
{
}

/* de-constructor */
T2TTranslator::~T2TTranslator()
{
    if (beamSize > 1)
        delete (BeamSearch*)seacher;
    else
        delete (GreedySearch*)seacher;
}

/* initialize the model */
void T2TTranslator::Init(T2TConfig& config)
{
    beamSize = config.beamSize;
    vSize = config.srcVocabSize;
    vSizeTgt = config.tgtVocabSize;
    sentBatch = config.sBatchSize;
    wordBatch = config.wBatchSize;

    if (beamSize > 1) {
        XPRINT1(0, stderr, "Translating with beam search (%d)\n", beamSize);
        seacher = new BeamSearch();
        ((BeamSearch*)seacher)->Init(config);
    }
    else if (beamSize == 1) {
        XPRINT1(0, stderr, "Translating with greedy search\n", beamSize);
        seacher = new GreedySearch();
        ((GreedySearch*)seacher)->Init(config);
    }
    else {
        CheckNTErrors(false, "invalid beam size\n");
    }
}

/*
test the model
>> ifn - input data file
>> sfn - source vocab file
>> tfn - target vocab file
>> ofn - output data file
>> model - pretrained model
*/
void T2TTranslator::Translate(const char* ifn, const char* sfn, const char* tfn, 
                              const char* ofn, T2TModel* model)
{
    int wc = 0;
    int wordCountTotal = 0;
    int sentCount = 0;
    int batchCount = 0;

    int devID = model->devID;

    double startT = GetClockSec();

    /* batch of input sequences */
    XTensor batchEnc;

    /* padding */
    XTensor paddingEnc;

    batchLoader.Init(ifn, sfn, tfn);
    XPRINT1(0, stderr, "[INFO] loaded the input file, elapsed=%.1fs \n", 
            GetClockSec() - startT);

    int count = 0;
    double batchStart = GetClockSec();
    while (!batchLoader.IsEmpty())
    {
        count++;

        for (int i = 0; i < model->decoder->nlayer; ++i) {
            model->decoder->selfAttCache[i].miss = true;
            model->decoder->enDeAttCache[i].miss = true;
        }

        auto indices = batchLoader.LoadBatch(&batchEnc, &paddingEnc, 
                                             sentBatch, wordBatch, devID);

        IntList* output = new IntList[indices.Size() - 1];

        /* greedy search */
        if (beamSize == 1) {
            ((GreedySearch*)seacher)->Search(model, batchEnc, paddingEnc, output);
        }
        /* beam search */
        else {
            XTensor score;
            ((BeamSearch*)seacher)->Search(model, batchEnc, paddingEnc, output, score);
        }

        for (int i = 0; i < indices.Size() - 1; ++i) {
            Result* res = new Result;
            res->id = indices[i];
            res->res = output[i];
            batchLoader.outputBuffer.Add(res);
        }
        delete[] output;

        wc += indices[-1];
        wordCountTotal += indices[-1];

        sentCount += (indices.Size() - 1);
        batchCount += 1;

        if (count % 1 == 0) {
            double elapsed = GetClockSec() - batchStart;
            batchStart = GetClockSec();
            XPRINT3(0, stderr, "[INFO] elapsed=%.1fs, sentence=%f, sword=%.1fw/s\n",
                    elapsed, float(sentCount) / float(batchLoader.inputBuffer.Size()), 
                    double(wc) / elapsed);
            wc = 0;
        }
    }

    /* append empty lines to the result */
    for (int i = 0; i < batchLoader.emptyLines.Size(); i++) {
        Result* emptyRes = new Result;
        emptyRes->id = batchLoader.emptyLines[i];
        batchLoader.outputBuffer.Add(emptyRes);
    }

    double startDump = GetClockSec();

    /* reorder the result */
    batchLoader.SortOutput();

    /* print the result to a file */
    batchLoader.DumpRes(ofn);

    double elapsed = GetClockSec() - startDump;

    XPRINT2(0, stderr, "[INFO] translation completed (word=%d, sent=%llu)\n", 
            wordCountTotal, batchLoader.inputBuffer.Size() + batchLoader.emptyLines.Size());
}

/*
dump the result into the file
>> file - data file
>> output - output tensor
*/
void T2TTranslator::Dump(FILE* file, XTensor* output)
{
    if (output != NULL && output->unitNum != 0) {
        int seqLength = output->dimSize[output->order - 1];

        for (int i = 0; i < output->unitNum; i += seqLength) {
            for (int j = 0; j < seqLength; j++) {
                int w = output->GetInt(i + j);
                if (w < 0 || w == 1 || w == 2)
                    break;
                fprintf(file, "%d ", w);
            }

            fprintf(file, "\n");
        }
    }
    else
    {
        fprintf(file, "\n");
    }
}

}