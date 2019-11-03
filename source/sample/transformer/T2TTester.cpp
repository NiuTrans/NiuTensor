/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2019, Natural Language Processing Lab, Northestern University.
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
 */

#include <math.h>
#include "T2TUtility.h"
#include "T2TTester.h"
#include "T2TSearch.h"
#include "../../tensor/XUtility.h"
#include "../../tensor/core/CHeader.h"
#include "../../network/XNoder.h"

using namespace nts;

namespace transformer
{

/* constructor */
T2TTester::T2TTester()
{
}

/* de-constructor */
T2TTester::~T2TTester()
{
}

/* initialize the model */
void T2TTester::Init(int argc, char ** argv)
{
    LoadParamInt(argc, argv, "vsize", &vSize, 1);
    LoadParamInt(argc, argv, "vsizetgt", &vSizeTgt, vSize);

    batchLoader.Init(argc, argv);
    seacher.Init(argc, argv);
}

/* 
test the model
>> fn - test data file
>> ofn - output data file
>> model - model that is trained
*/
void T2TTester::Test(const char * fn, const char * ofn, T2TModel * model)
{
    int wc = 0;
    int ws = 0;
    int wordCount = 0;
    int wordCountTotal = 0;
    int sentCount = 0;
    int batchCount = 0;
    float loss = 0;

    /* data files */
    FILE * file = fopen(fn, "rb");
    CheckNTErrors(file, "Cannot read the test file");
    FILE * ofile = fopen(ofn, "wb");
    CheckNTErrors(ofile, "Cannot open the output file");

    int devID = model->devID;

    XNet net;
    
    double startT = GetClockSec();
        
    wordCount = 0;
        
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
    int * seqs = new int[MILLION];

    batchLoader.SetRandomBatch(false);
    batchLoader.ClearBuf();

    while(batchLoader.LoadBatch(file, model->isLM, 
                                &batchEnc, &paddingEnc, &paddingDec, &paddingDec, &gold, &label,
                                seqs, vSize, vSizeTgt,
                                1, 1, false, ws, wc, devID, false))
    {
        CheckNTErrors(batchEnc.order == 2, "wrong tensor order of the sequence batch!");
        CheckNTErrors(!model->isLM, "Only MT model is supported!");
        
        XTensor output;

        seacher.Search(model, &batchEnc, &paddingEnc, &output);

        Dump(ofile, &output);

        float prob = 0;
            
        loss += -prob;
        wc = batchEnc.GetDim(-1);
        wordCount += wc;
        wordCountTotal += wc;
        sentCount += batchEnc.GetDim(-2);
        batchCount += 1;

        if (batchCount % 1 == 0) {
            double elapsed = GetClockSec() - startT;
            XPRINT3(0, stderr, 
                   "[INFO] elapsed=%.1fs, sentence=%d, sword=%d\n",
                    elapsed, sentCount, wordCount);
        }
    }
        
    fclose(file);
    fclose(ofile);

    delete[] seqs;
    
    double elapsed = GetClockSec() - startT;

    XPRINT3(0, stderr, "[INFO] test finished (took %.1fs, word=%d, and ppl=%.3f)\n",
            elapsed,wordCountTotal, exp(loss/wordCount));
}

/*
dump the result into the file
>> file - data file
>> output - output tensor
*/
void T2TTester::Dump(FILE * file, XTensor * output)
{
    int seqLength = output->GetDim(-1);

    for (int i = 0; i < output->unitNum; i += seqLength) {
        for (int j = 0; j < seqLength; j++) {
            int w = output->GetInt(i + j);
            fprintf(file, "%d ", w);
            if (w < 0)
                break;
        }

        fprintf(file, "\n");
    }
}

}
