/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2018, Natural Language Processing Lab, Northestern University. 
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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2019-04-25
 * it is cold today but i'll move to a warm place tomorrow :)
 */

#ifndef __T2TBATCHLOADER_H__
#define __T2TBATCHLOADER_H__

#include "../../network/XNet.h"

using namespace nts;

namespace transformer
{

#define MAX_SEQUENCE_LENGTH 1024 * 4

/* node to keep batch information */
struct BatchNode
{
    /* begining position */
    int beg;

    /* end position */
    int end;

    /* maximum word number on the encoder side */
    int maxEnc;

    /* maximum word number on the decoder side */
    int maxDec;

    /* a key for sorting */
    int key;
};

class T2TBatchLoader
{
public:
    /* buffer for loading words */
    int * buf;

    /* another buffer */
    int * buf2;

    /* batch buf */
    BatchNode * bufBatch;

    /* buffer size */
    int bufSize;

    /* size of batch buffer */
    int bufBatchSize;

    /* length of each sequence */
    int * seqLen;

    /* another array */
    int * seqLen2;

    /* offset of the first word for each sequence */
    int * seqOffset;

    /* number of sequences in the buffer */
    int nseqBuf;

    /* offset for next sequence in the buffer */
    int nextSeq;

    /* offset for next batch */
    int nextBatch;

    /* indicates whether we double the </s> symbol for the output of lms */
    bool isDoubledEnd;
    
    /* indicates whether we use batchsize = max * sc
       rather rather than batchsize = word-number, where max is the maximum
       length and sc is the sentence number */
    bool isSmallBatch;

    /* counterpart of "isSmallBatch" */
    bool isBigBatch;

    /* randomize batches */
    bool isRandomBatch;

    /* bucket size */
    int bucketSize;

public:
    /* constructor */
    T2TBatchLoader();

    /* de-constructor */
    ~T2TBatchLoader();

    /* initialization */
    void Init(int argc, char ** argv);

    /* load data to buffer */
    int LoadBuf(FILE * file, bool isSorted, int step);

    /* clear data buffer */
    void ClearBuf();

    /* set the random batch flag */
    void SetRandomBatch(bool flag = true);

    /* load a batch of sequences */
    int LoadBatch(FILE * file, bool isLM,
                  XTensor * batchEnc, XTensor * paddingEnc, 
                  XTensor * batchDec, XTensor * paddingDec,
                  XTensor * gold, XTensor * label,
                  int * seqs,
                  int vsEnc, int vsDec, int sBatch, int wBatch, 
                  bool isSorted, int &ws, int &wCount,
                  int devID, bool isTraining);

    /* load a batch of sequences (for language modeling) */
    int LoadBatchLM(FILE * file, 
                    XTensor * batchEnc, XTensor * paddingEnc,
                    XTensor * batchDec, XTensor * paddingDec,
                    XTensor * gold, XTensor * label,
                    int * seqs, int vs, int sBatch, int wBatch, 
                    bool isSorted, int &wCount,
                    int devID, bool isTraining);

    /* load a batch of sequences (for machine translation) */
    int LoadBatchMT(FILE * file, 
                    XTensor * batchEnc, XTensor * paddingEnc, 
                    XTensor * batchDec, XTensor * paddingDec,
                    XTensor * gold, XTensor * label,
                    int * seqs, int vsEnc, int vsDec, int sBatch, int wBatch, 
                    bool isSorted, int &ws, int &wCount,
                    int devID, bool isTraining);

    /* shuffle the data file */
    void Shuffle(const char * srcFile, const char * tgtFile);
};
}

#endif