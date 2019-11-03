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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-31
 */

#include "T2TBatchLoader.h"
#include "T2TUtility.h"
#include "../../tensor/XUtility.h"
#include "../../tensor/core/CHeader.h"
#include "../../network/XNoder.h"

namespace transformer
{

/* constructor */
T2TBatchLoader::T2TBatchLoader()
{
    seqLen = NULL;
    seqLen2 = NULL;
    nseqBuf = 0;
    nextSeq = -1;
    nextBatch = -1;
    buf = NULL;
    buf2 = NULL;
    bufBatch = NULL;
    bufSize = 0;
    bufBatchSize = 0;
    seqOffset = NULL;
}

/* de-constructor */
T2TBatchLoader::~T2TBatchLoader()
{
    delete[] buf;
    delete[] buf2;
    delete[] bufBatch;
    delete[] seqLen;
    delete[] seqLen2;
    delete[] seqOffset;
}

/* 
initialization 
>> argc - number of arguments
>> argv - list of pointers to the arguments
*/
void T2TBatchLoader::Init(int argc, char ** argv)
{
    LoadParamInt(argc, argv, "bufsize", &bufSize, 50000);
    LoadParamBool(argc, argv, "doubledend", &isDoubledEnd, false);
    LoadParamBool(argc, argv, "smallbatch", &isSmallBatch, true);
    LoadParamBool(argc, argv, "bigbatch", &isBigBatch, false);
    LoadParamBool(argc, argv, "randbatch", &isRandomBatch, false);
    LoadParamInt(argc, argv, "bucketsize", &bucketSize, 0);

    buf  = new int[bufSize];
    buf2 = new int[bufSize];
    bufBatch = new BatchNode[bufSize];
    seqLen  = new int[bufSize];
    seqLen2 = new int[bufSize];
    seqOffset = new int[bufSize];
}

char line[MAX_SEQUENCE_LENGTH];

struct SampleNode
{
    int id;
    int offset;
    int * p;
    int size;
    int value;
    int key;
};

int CompareSampleNode(const void * a, const void * b)
{
   return ((SampleNode*)b)->value - ((SampleNode*)a)->value;
}

int CompareSampleNodeV2(const void * a, const void * b)
{
    return ((SampleNode*)b)->key - ((SampleNode*)a)->key;
}

/* 
load data to buffer 
>> file - where to load data
>> isSorted - indicates whether the samples are sorted by length
>> step - the number of sequences we go over when move to the next sample
*/
int T2TBatchLoader::LoadBuf(FILE * file, bool isSorted, int step)
{
    int lineCount = 0;
    int seqCount = 0;
    int wordCount = 0;
    while(fgets(line, MAX_SEQUENCE_LENGTH - 1, file)){
        int len = (int)strlen(line);

        while(line[len - 1] == '\r' || line[len - 1] == '\n'){
            line[len - 1] = 0;
            len--;
        }

        len = (int)strlen(line);
        if(len == 0)
            continue;
        
        /* how many characters are in a word */
        int wSize = 0;
        
        /* how many words are in the sentence */
        int wNum = 0;
        int wNumLocal = 0;
        int i = 0;

        for(i = 0; i < len; i++){
            /* load word (id) seperated by space or tab */
            if((line[i] == ' ' || line[i] == '\t') && wSize > 0){
                line[i] = 0;

                if(wSize == 3 && line[i - 1] == '|' && line[i - 2] == '|' && line[i - 3] == '|'){
                    seqLen[seqCount] = wNumLocal;
                    seqOffset[seqCount] = wordCount + wNum - wNumLocal;
                    seqCount++;
                    wNumLocal = 0;
                }
                else{
                    buf[wordCount + wNum++] = atoi(line + i - wSize);
                    wNumLocal++;
                }

                wSize = 0;
            }
            else
                wSize++;
        }

        if(wSize > 0){
            buf[wordCount + wNum++] = atoi(line + i - wSize);
            wNumLocal++;
        }

        seqLen[seqCount] = wNumLocal;
        seqOffset[seqCount] = wordCount + wNum - wNumLocal;
        seqCount++;

        wordCount += wNum;
        lineCount++;

        if(wordCount >= bufSize - MAX_SEQUENCE_LENGTH)
            break;

        CheckNTErrors(seqCount % step == 0, "Wrong number of sequences!");
    }

    nseqBuf = seqCount;
    nextSeq = 0;

    /* sort the sequences by length */
    if (isSorted) {
        CheckNTErrors(seqCount % step == 0, "Wrong number of sequences!");
        SampleNode * nodes = new SampleNode[seqCount];
        int count = 0;
        int offset = 0;
        for (int i = 0; i < seqCount; i += step) {
            SampleNode &node = nodes[count];
            node.id = count;
            node.offset = i;
            node.p = buf + offset;
            node.size = 0;
            int max = 0;
            for (int j = 0; j < step; j++) {
                node.size += seqLen[i + j];
                max = MAX(max, seqLen[i + j]);
            }
            node.value = max;
            node.key = rand();
            count++;
            offset += node.size;
        }

        qsort(nodes, count, sizeof(SampleNode), CompareSampleNode);

        /* distribute samples into buckets. In each bucket, sequences have
           similar a length */
        if (bucketSize > 0) {
            int low = 0;
            int high = low + bucketSize;
            int n = count - 1;
            int m = n;
            int num = 0;
            while (num < count) {
                for (m = n; m >= 0; m--) {
                    if (nodes[m].value > high)
                        break;
                }

                qsort(nodes + m + 1, n - m, sizeof(SampleNode), CompareSampleNodeV2);
                num += (n - m);
                n = m;
                low += bucketSize;
                high = low + bucketSize;
            }
        }

        count = 0;
        offset = 0;
        for(int i = 0; i < seqCount; i += step){
            SampleNode &node = nodes[count];
            memcpy(buf2 + offset, node.p, sizeof(int) * node.size);
            for(int j = 0; j < step; j++){
                seqLen2[i + j] = seqLen[node.offset + j];
                seqOffset[i + j] = offset + (j > 0 ? seqLen[node.offset + j - 1] : 0);
            }
            count += 1;
            offset += node.size;
        }

        int * tmp = buf;
        buf = buf2;
        buf2 = tmp;
        tmp = seqLen;

        seqLen = seqLen2;
        seqLen2 = tmp;

        delete[] nodes;
    }

    return lineCount;
}

/* clear the data buffer */
void T2TBatchLoader::ClearBuf()
{
    nseqBuf = 0;
    nextSeq = -1;
}

/*
set the random batch flag
>> flag - as it is
*/
void T2TBatchLoader::SetRandomBatch(bool flag)
{
    isRandomBatch = flag;
}

/*
load a batch of sequences 
>> file - the handle to the data file
>> isLM - indicates whether the data is used for training lms
>> batchEnc - the batch of the input sequences
>> paddingEnc - padding of the input sequences
>> batchDec - the batch of the output sequences
>> paddingDec - padding of the output sequences
>> gold - gold standard
>> seqs - keep the sequences in an array
>> vsEnc - size of the encoder vocabulary
>> vsDec - size of the decoder vocabulary
>> sBatch - batch size of sequences
>> wBatch - batch size of words
>> isSorted - indicates whether the sequences are sorted by length
>> wCount - word count
>> devID - device id
>> isTraining - indicates whether we are training the model
*/
int T2TBatchLoader::LoadBatch(FILE * file, bool isLM, 
                          XTensor * batchEnc, XTensor * paddingEnc, 
                          XTensor * batchDec, XTensor * paddingDec,
                          XTensor * gold, XTensor * label,
                          int * seqs,
                          int vsEnc, int vsDec, int sBatch, int wBatch, 
                          bool isSorted, int &ws, int &wCount,
                          int devID, bool isTraining)
{
    if(isLM){
        return LoadBatchLM(file, batchEnc, paddingEnc, batchDec, paddingDec, gold, label,
                           seqs, vsEnc, sBatch, wBatch, 
                           isSorted, wCount, devID, isTraining);
    }
    else{
        return LoadBatchMT(file, batchEnc, paddingEnc, batchDec, paddingDec, gold, label,
                           seqs, vsEnc, vsDec, sBatch, wBatch, 
                           isSorted, ws, wCount, devID, isTraining);
    }
}

/* 
load a batch of sequences (for LM)
>> file - the handle to the data file
>> isLM - indicates whether the data is used for training lms
>> batchEnc - the batch of the input sequences
>> paddingEnc - padding of the input sequences
>> batchDec - the batch of the output sequences
>> paddingDec - padding of the output sequences
>> gold - gold standard (distribution of every position)
>> label - (gold standard) label index of every position
>> seqs - keep the sequences in an array
>> vSize - vocabulary size
>> sBatch - batch size of sequences
>> wBatch - batch size of words
>> isSorted - indicates whether the sequences are sorted by length
>> wCount - word count
>> devID - device id
>> isTraining - indicates whether we are training the model
*/
int T2TBatchLoader::LoadBatchLM(FILE * file, 
                            XTensor * batchEnc, XTensor * paddingEnc,
                            XTensor * batchDec, XTensor * paddingDec,
                            XTensor * gold, XTensor * label,
                            int * seqs,
                            int vSize, int sBatch, int wBatch, 
                            bool isSorted, int &wCount,
                            int devID, bool isTraining)
{
    if(nextSeq < 0 || nextSeq >= nseqBuf)
        LoadBuf(file, isSorted, 1);

    int seq = MAX(nextSeq, 0);
    int wc = 0;
    int wn = 0;
    int sc = 0;
    int max = 0;
    while(seq + sc < nseqBuf){
        int len = isDoubledEnd ? seqLen[seq + sc] : seqLen[seq + sc] - 1;
        CheckNTErrors(len > 0, "Empty sequence!");
        wn = len;
        wc += wn;
        sc += 1;

        if(max < wn)
            max = wn;

        int tc = isBigBatch ? wc : max * sc;
        if(sc >= sBatch && tc >= wBatch)
            break;
    }

    wCount = 0;
    nextSeq = seq + sc;

    if(sc <= 0)
        return 0;

    int dims[MAX_TENSOR_DIM_NUM];
    dims[0] = sc;
    dims[1] = max;
    dims[2] = vSize;

    InitTensor2D(batchEnc, sc, max, X_INT, devID);
    InitTensor2D(label, sc, max, X_INT, devID);
    InitTensor(gold, 3, dims, X_FLOAT, devID);
    InitTensor2D(paddingEnc, sc, max, X_FLOAT, devID);
    InitTensor2D(paddingDec, sc, max, X_FLOAT, devID);

    batchEnc->SetZeroAll();
    label->SetZeroAll();
    gold->SetZeroAll();
    paddingEnc->SetZeroAll();
    paddingDec->SetZeroAll();

    int seqSize = 0;
    
    int * batchEncValues = new int[batchEnc->unitNum];
    int * labelValues = new int[label->unitNum];
    MTYPE * goldOffsets = new MTYPE[gold->unitNum];
    MTYPE * paddingEncOffsets = new MTYPE[paddingEnc->unitNum];
    MTYPE * paddingDecOffsets = new MTYPE[paddingDec->unitNum];

    int wGold = 0;

    memset(batchEncValues, 0, sizeof(int) * batchEnc->unitNum);
    memset(labelValues, 0, sizeof(int) * label->unitNum);

    for(int s = seq; s < seq + sc; s++){
        int len = isDoubledEnd ? seqLen[s] : seqLen[s] - 1;
        CheckNTErrors(len <= max, "Something is wrong!");
        for(int w = 0; w < len; w++){
            int num = buf[seqOffset[s] + w];
            batchEncValues[(int)batchEnc->GetOffset2D(s - seq, w)] = num;
            paddingEncOffsets[wCount] = paddingEnc->GetOffset2D(s - seq, w);
            paddingDecOffsets[wCount] = paddingDec->GetOffset2D(s - seq, w);
            if (w > 0) {
                goldOffsets[wGold++] = gold->GetOffset3D(s - seq, w - 1, num);
                labelValues[(int)label->GetOffset2D(s - seq, w - 1)] = buf[seqOffset[s] + w];
            }
            
            if (w == len - 1) {
                if (isDoubledEnd) {
                    goldOffsets[wGold++] = gold->GetOffset3D(s - seq, w, num);
                    labelValues[(int)label->GetOffset2D(s - seq, w)] = buf[seqOffset[s] + w];
                }   
                else {
                    goldOffsets[wGold++] = gold->GetOffset3D(s - seq, w, buf[seqOffset[s] + w + 1]);
                    labelValues[(int)label->GetOffset2D(s - seq, w)] = buf[seqOffset[s] + w + 1];
                }
                    
            }

            wCount++;

            if(seqs != NULL)
                seqs[seqSize++] = buf[seqOffset[s] + w];
        }

        if(seqs != NULL){
            for(int w = len; w < max; w++)
                seqs[seqSize++] = -1;
        }
    }

    batchEnc->SetData(batchEncValues, batchEnc->unitNum);
    label->SetData(labelValues, label->unitNum);
    gold->SetDataBatched(goldOffsets, 1.0F, wGold);
    paddingEnc->SetDataBatched(paddingEncOffsets, 1.0F, wCount);
    paddingDec->SetDataBatched(paddingDecOffsets, 1.0F, wCount);

    /*XTensor * tmp = NewTensorBuf(paddingEnc, devID);
    _ConvertDataType(batchEnc, tmp);
    _NotEqual(tmp, paddingEnc, 0);
    DelTensorBuf(tmp);
        
    XTensor * tmp2 = NewTensorBuf(paddingDec, devID);
    _ConvertDataType(batchEnc, tmp2);
    _NotEqual(tmp2, paddingDec, 0);
    DelTensorBuf(tmp2);*/

    delete[] batchEncValues;
    delete[] labelValues;
    delete[] goldOffsets;
    delete[] paddingEncOffsets;
    delete[] paddingDecOffsets;

    fflush(tf);

    return sc;
}

int CompareBatchNode(const void * a, const void * b)
{
    return ((BatchNode*)b)->key - ((BatchNode*)a)->key;
}


/*
load a batch of sequences (for MT)
>> file - the handle to the data file
>> batchEnc - the batch of the input sequences
>> paddingEnc - padding of the input sequences
>> batchDec - the batch of the output sequences
>> paddingDec - padding of the output sequences
>> gold - gold standard (distribution of every position)
>> label - (gold standard) label index of every position
>> seqs - keep the sequences in an array
>> vSizeEnc - size of the encoder vocabulary
>> vSizeDec - size of the decoder vocabulary
>> sBatch - batch size of sequences
>> wBatch - batch size of words
>> isSorted - indicates whether the sequences are sorted by length
>> wCount - word count
>> devID - device id
>> isTraining - indicates whether we are training the model
*/
int T2TBatchLoader::LoadBatchMT(FILE * file, 
                            XTensor * batchEnc, XTensor * paddingEnc, 
                            XTensor * batchDec, XTensor * paddingDec,
                            XTensor * gold, XTensor * label,
                            int * seqs,
                            int vSizeEnc, int vSizeDec, int sBatch, int wBatch, 
                            bool isSorted, int &ws, int &wCount,
                            int devID, bool isTraining)
{
    if (nextBatch < 0 || nextBatch >= bufBatchSize) {
        LoadBuf(file, isSorted, 2);

        int seq = 0;

        bufBatchSize = 0;
        nextBatch = 0;

        /* we segment the buffer into batches */
        while (seq < nseqBuf) {

            int wcEnc = 0;
            int wcDec = 0;
            int wnEnc = 0;
            int wnDec = 0;
            int maxEnc = 0;
            int maxDec = 0;
            int sc = 0;

            while (seq + sc < nseqBuf) {

                /* source-side sequence */
                wnEnc = seqLen[seq + sc];

                /* target-side sequence */
                wnDec = isDoubledEnd ? seqLen[seq + sc + 1] : seqLen[seq + sc + 1] - 1;

                int tcEnc = isBigBatch ? (wcEnc + wnEnc) : MAX(maxEnc, wnEnc) * (sc + 2) / 2;
                int tcDec = isBigBatch ? (wcDec + wnDec) : MAX(maxDec, wnDec) * (sc + 2) / 2;

                if (sc != 0 && sc > sBatch * 2 && (tcEnc > wBatch || tcDec > wBatch))
                    break;

                wcEnc += wnEnc;
                sc += 1;

                if (maxEnc < wnEnc)
                    maxEnc = wnEnc;

                wcDec += wnDec;
                sc += 1;

                if (maxDec < wnDec)
                    maxDec = wnDec;
            }

            BatchNode & batch = bufBatch[bufBatchSize];
            batch.beg = seq;
            batch.end = seq + sc;
            batch.maxEnc = maxEnc;
            batch.maxDec = maxDec;
            batch.key = rand();

            bufBatchSize++;
            seq = seq + sc;
        }

        if(isRandomBatch)
            qsort(bufBatch, bufBatchSize, sizeof(BatchNode), CompareBatchNode);
    }

    if(bufBatchSize <= 0)
        return 0;

    BatchNode & batch = bufBatch[nextBatch++];
    int seq = batch.beg;
    int sc = batch.end - batch.beg;
    int maxEnc = batch.maxEnc;
    int maxDec = batch.maxDec;

    CheckNTErrors(sc % 2 == 0, "The input samples must be paired");

    int sCount = sc/2;
    int seqSize = 0;

    InitTensor2D(batchEnc, sCount, maxEnc, X_INT, devID);
    InitTensor2D(paddingEnc, sCount, maxEnc, X_FLOAT, devID);
    InitTensor2D(batchDec, sCount, maxDec, X_INT, devID);
    InitTensor2D(paddingDec, sCount, maxDec, X_FLOAT, devID);
    InitTensor2D(label, sCount, maxDec, X_INT, devID);
    //InitTensor(gold, 3, dimsDec, X_FLOAT, devID);

    batchEnc->SetZeroAll();
    paddingEnc->SetZeroAll();
    batchDec->SetZeroAll();
    paddingDec->SetZeroAll();
    label->SetZeroAll();
    //gold->SetZeroAll();

    int wCountEnc = 0;
    int wCountDec = 0;
    int wCountPad = 0;
    wCount = 0;

    int * batchEncValues = new int[batchEnc->unitNum];
    int * batchDecValues = new int[batchDec->unitNum];
    int * labelValues = new int[label->unitNum];
    MTYPE * paddingEncOffsets = new MTYPE[sc * maxEnc / 2];
    MTYPE * paddingDecOffsets = new MTYPE[sc * maxDec / 2];
    //MTYPE * goldOffsets = new MTYPE[sc * maxDec / 2];

    memset(batchEncValues, 0, sizeof(int) * batchEnc->unitNum);
    memset(batchDecValues, 0, sizeof(int) * batchDec->unitNum);
    memset(labelValues, 0, sizeof(int) * batchDec->unitNum);

    /* batch of the source-side sequences */
    for(int s = seq; s < seq + sc; s += 2){
        int len = seqLen[s];
        int sent = (s - seq)/2;
        for(int w = 0; w < len; w++){
            int num = buf[seqOffset[s] + w];
            batchEncValues[batchEnc->GetOffset2D(sent, w)] = num;
            paddingEncOffsets[wCountEnc] = paddingEnc->GetOffset2D(sent, w);
            wCountEnc++;
        }
    }
    ws = wCountEnc;
    batchEnc->SetData(batchEncValues, batchEnc->unitNum);
    paddingEnc->SetDataBatched(paddingEncOffsets, 1.0F, wCountEnc);
    //XTensor * tmp = NewTensorBuf(paddingEnc, devID);
    //_ConvertDataType(batchEnc, tmp);
    //tmp->Dump(stderr, "tmp:");
    //_NotEqual(tmp, paddingEnc, 0);
    //DelTensorBuf(tmp);

    /* batch of the target-side sequences */
    for(int s = seq + 1; s < seq + sc; s += 2){
        int len = isDoubledEnd ? seqLen[s] : seqLen[s] - 1;
        CheckNTErrors(len <= maxDec, "Something is wrong!");
        int sent = (s - seq - 1)/2;
        for(int w = 0; w < len; w++){
            int num = buf[seqOffset[s] + w];
            batchDecValues[batchDec->GetOffset2D(sent, w)] = num;
            //paddingDecOffsets[wCountDec] = paddingDec->GetOffset2D(sent, w);
            if (w < len-1){
                paddingDecOffsets[wCountPad++] = paddingDec->GetOffset2D(sent, w);
                wCount++;
            }
            if (w > 0) {
                //goldOffsets[wGold++] = gold->GetOffset3D(sent, w - 1, buf[seqOffset[s] + w]);
                labelValues[label->GetOffset2D(sent, w - 1)] = buf[seqOffset[s] + w];
            }
            if (w == len - 1) {
                if (isDoubledEnd) {
                    //goldOffsets[wGold++] = gold->GetOffset3D(sent, w, buf[seqOffset[s] + w]);
                    labelValues[label->GetOffset2D(sent, w)] = buf[seqOffset[s] + w];
                }
                else {
                    //goldOffsets[wGold++] = gold->GetOffset3D(sent, w, buf[seqOffset[s] + w + 1]);
                    labelValues[label->GetOffset2D(sent, w)] = buf[seqOffset[s] + w + 1];
                }
            }
            //wCount++;
            wCountDec++;
            if(seqs != NULL)
                seqs[seqSize++] = buf[seqOffset[s] + w];
        }

        if(seqs != NULL){
            for(int w = len; w < maxDec; w++)
                seqs[seqSize++] = -1;
        }
    }

    batchDec->SetData(batchDecValues, batchDec->unitNum);
    label->SetData(labelValues, label->unitNum);
    paddingDec->SetDataBatched(paddingDecOffsets, 1.0F, wCountPad);

    //XTensor * tmp2 = NewTensorBuf(paddingDec, devID);
    //_ConvertDataType(batchDec, tmp2);
    //_NotEqual(tmp2, paddingDec, 0);
    //DelTensorBuf(tmp2);

    //gold->SetDataBatched(goldOffsets, 1.0F, wGold);

    delete[] batchEncValues;
    delete[] batchDecValues;
    delete[] labelValues;
    delete[] paddingEncOffsets;
    delete[] paddingDecOffsets;
    //delete[] goldOffsets;

    return sc;
}

/* 
shuffle lines of the file 
>> srcFile - the source file to shuffle
>> tgtFile - the resulting file
*/
void T2TBatchLoader::Shuffle(const char * srcFile, const char * tgtFile)
{
    char * line = new char[MAX_LINE_LENGTH];
#ifndef WIN32
    sprintf(line, "shuf %s > %s", srcFile, tgtFile);
    system(line);
#else
    ShowNTErrors("Cannot shuffle the file on WINDOWS systems!");
#endif
    delete[] line;
}

}
    