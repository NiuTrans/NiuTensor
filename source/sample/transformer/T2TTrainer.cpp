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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-08-02
 */

#include <math.h>
#include "T2TTrainer.h"
#include "T2TUtility.h"
#include "../../tensor/XUtility.h"
#include "../../tensor/core/CHeader.h"
#include "../../network/XNoder.h"

#ifndef WIN32
#include <sys/time.h>
#include <unistd.h>
#endif

namespace transformer
{

/* constructor */
T2TTrainer::T2TTrainer()
{
    seqLen = NULL;
    seqLen2 = NULL;
    nseqBuf = 0;
    nextSeq = -1;
    nextBatch = -1;

    argNum = 0;
    argArray = NULL;
    buf = NULL;
    buf2 = NULL;
    bufBatch = NULL;
    bufSize = 0;
    bufBatchSize = 0;
    seqOffset = NULL;
}

/* de-constructor */
T2TTrainer::~T2TTrainer()
{
    delete[] buf;
    delete[] buf2;
    delete[] bufBatch;
    delete[] seqLen;
    delete[] seqLen2;
    delete[] seqOffset;

    for(int i = 0; i < moments.count; i++){
        XTensor * m = (XTensor*)moments.Get(i);
        delete m;
    }

    for(int i = 0; i < moments2nd.count; i++){
        XTensor * m = (XTensor*)moments2nd.Get(i);
        delete m;
    }

    for(int i = 0; i < argNum; i++)
        delete[] argArray[i];
    delete[] argArray;

}

/* 
initialization 
>> argc - number of arguments
>> argv - list of pointers to the arguments
*/
void T2TTrainer::Init(int argc, char ** argv)
{
    argNum = argc;
    argArray = new char*[argc];
    for(int i = 0; i < argNum; i++){
        argArray[i] = new char[strlen(argv[i]) + 1];
        strcpy(argArray[i], argv[i]);
    }

    bool useMem = false;

    LoadParamBool(argc, argv, "mem", &useMem, useMem);
    LoadParamFloat(argc, argv, "lrate", &lrate, 1.0F);
    LoadParamFloat(argc, argv, "lrbias", &lrbias, 0);
    LoadParamInt(argc, argv, "sbatch", &sBatchSize, 1);
    LoadParamInt(argc, argv, "wbatch", &wBatchSize, 1);
    LoadParamInt(argc, argv, "nepoch", &nepoch, 1);
    LoadParamInt(argc, argv, "nstep", &nstep, 1);
    LoadParamInt(argc, argv, "d", &d, 512);
    LoadParamInt(argc, argv, "nwarmup", &nwarmup, 4000);
    LoadParamInt(argc, argv, "vsize", &vSize, 1);
    LoadParamInt(argc, argv, "vsizetgt", &vSizeTgt, vSize);
    LoadParamBool(argc, argv, "sorted", &isLenSorted, false);
    LoadParamInt(argc, argv, "bufsize", &bufSize, 50000);
    LoadParamBool(argc, argv, "adam", &useAdam, false);
    LoadParamFloat(argc, argv, "adambeta1", &adamBeta1, 0.9F);
    LoadParamFloat(argc, argv, "adambeta2", &adamBeta2, 0.98F);
    LoadParamFloat(argc, argv, "adamdelta", &adamDelta, 1e-9F);
    LoadParamBool(argc, argv, "shuffled", &isShuffled, false);
    LoadParamFloat(argc, argv, "labelsmoothing", &labelSmoothingP, 0);
    LoadParamInt(argc, argv, "nstepcheckpoint", &nStepCheckpoint, -1);
    LoadParamBool(argc, argv, "epochcheckpoint", &useEpochCheckpoint, false);
    LoadParamInt(argc, argv, "updatestep", &updateStep, 1);
    LoadParamBool(argc, argv, "doubledend", &isDoubledEnd, false);
    LoadParamBool(argc, argv, "smallbatch", &isSmallBatch, true);
    LoadParamBool(argc, argv, "bigbatch", &isBigBatch, false);
    LoadParamBool(argc, argv, "debug", &isDebugged, false);
    LoadParamBool(argc, argv, "randbatch", &isRandomBatch, false);
    LoadParamInt(argc, argv, "bucketsize", &bucketSize, 0);

    buf  = new int[bufSize];
    buf2 = new int[bufSize];
    bufBatch = new BatchNode[bufSize];
    seqLen  = new int[bufSize];
    seqLen2 = new int[bufSize];
    seqOffset = new int[bufSize];

    adamBeta1T = 1.0F;
    adamBeta2T = 1.0F;
}

int tc = 0;

/* 
train the model
>> fn - training data file
>> validFN - validation data file
>> modelFN - where we keep the model
>> model - model to train
*/
void T2TTrainer::Train(const char * fn, const char * validFN, const char * modelFN, T2TModel * model)
{
    int step = 0;
    int wc = 0;
    int wordCount = 0;
    int wordCountTotal = 0;
    bool isEnd = false;
    float loss = 0;
    float lr = 0;
    int nStepCheck = 0;
    int nCheckpoint = 0;
    int nSkipped = 0;
    int gradStep = 0;
    int validStep = 0;
    int epoch = 0;

    char * trainFN = new char[(int)strlen(fn) + 10];
    strcpy(trainFN, fn);

#ifndef WIN32
    if(isShuffled)
        sprintf(trainFN, "%s.random", fn);
#endif

    int devID = model->devID;
    XMem * mem = model->mem;
    XNet net;

    if(isDebugged)
        net.SetGradEfficientFlag(false);
    
    PrepareModel(model);

    double startT = GetClockSec();
    
    for(epoch = 1; epoch <= nepoch; epoch++){
#ifndef WIN32
        if(isShuffled)
            Shuffle(fn, trainFN);
#endif
        
        FILE * file = fopen(trainFN, "rb");
        CheckNTErrors(file, "cannot open training file!");
        
        wordCount = 0;
        loss = 0;
        
        /* batch of sequences (on the encoder and decoder sides) */
        XTensor batchEnc;
        XTensor batchDec;

        /* padding */
        XTensor paddingEnc;
        XTensor paddingDec;

        /* gold standard */
        XTensor gold;
        
        /* label smoothed gold standard (if needed) */
        XTensor goldSmoothed;
        
        while (LoadBatch(file, model->isLM, &batchEnc, &paddingEnc, &batchDec, &paddingDec, &gold, 
                         NULL, vSize, vSizeTgt,
                         sBatchSize, wBatchSize, isLenSorted, wc, devID, mem, true)) 
        {

            CheckNTErrors(batchEnc.order == 2, "wrong tensor order of the sequence batch");

            /* output probabilities */
            XTensor output;

            /* make the network */
            if(model->isLM)
                model->MakeLM(batchEnc, output, paddingEnc, true);
            else if(model->isMT)
                model->MakeMT(batchEnc, batchDec, output, paddingEnc, paddingDec, true);
            else{
                ShowNTErrors("Illegal model type!");
            }

            /* back-propagation for obtaining gradients */
            if (labelSmoothingP > 0)
                LabelSmooth(&gold, &goldSmoothed, labelSmoothingP);

            /* make paddings for the output */
            if (output.GetDim(0) > 1)
                PadOutput(&output, &gold, &paddingDec);

            /* get probabilities */
            float prob = GetProb(&output, &gold, NULL);
            
            DTYPE lossLocal = -prob / wc;
            bool doUpdate = (!IsNAN(lossLocal) && !IsINF(lossLocal) && lossLocal < 1e3F);

            XTensor &g = labelSmoothingP > 0 ? goldSmoothed : gold;   

            if (doUpdate) {
                
                /* recale the output for normalized loss */
                RescaleOutput(&output, &g, &paddingDec);
                
                /* back-propagation */
                net.Backward(output, g, paddingDec, CROSSENTROPY);
                
                gradStep += 1;
                loss += -prob;
                wordCount += wc;
                wordCountTotal += wc;
                
                /* update the parameters */
                if(gradStep == updateStep){
                    
                    /* learning rate */
                    lr = lrate * (1.0F / (float)sqrt((float)d)) * (float)MIN(pow((float)validStep + 1, -0.5F - lrbias), ((float)validStep + 1) * pow((float)nwarmup, -1.5F - lrbias));
                    
                    /* model update */
                    Update(model, lr);
                    
                    gradStep = 0;
                    validStep++;
                }
            }
            else
                nSkipped++;
            
            if(++step >= nstep){
                isEnd = true;
                break;
            }
            
            if (step % 100 == 0) {
                double elapsed = GetClockSec() - startT;
                XPRINT8(0, stderr, "[INFO] lr=%.2e, elapsed=%.1fs, step=%d, epoch=%d, word=%d, loss=%.3f, ppl=%.3f, sppl=%.3f",
                        lr, elapsed, step, epoch, wordCountTotal, loss/wordCount, exp(loss/wordCount), exp(-prob/wc));
                if (!doUpdate)
                    XPRINT(0, stderr, " (no update)");
                XPRINT(0, stderr, "\n");
            }

            if(nStepCheckpoint > 0 && ++nStepCheck >= nStepCheckpoint){
                MakeCheckpoint(model, validFN, modelFN, "step", step);
                nStepCheck = 0;
                nCheckpoint++;
            }
        }
        
        fclose(file);
        
        if (isEnd)
            break;

        if(useEpochCheckpoint)
            MakeCheckpoint(model, validFN, modelFN, "epoch", epoch);
    }

    double elapsed = GetClockSec() - startT;
    
    epoch = MIN(epoch, nepoch);
    
    XPRINT7(0, stderr, "[INFO] lr=%.2e, elapsed=%.1fs, step=%d, epoch=%d, word=%d, loss=%.3f, ppl=%.3f\n",
            lr, elapsed, step, epoch, wordCountTotal, loss/wordCount, exp(loss/wordCount));
    XPRINT4(0, stderr, "[INFO] training finished (took %.1fs, step=%d, skipped=%d and epoch=%d)\n",
            elapsed, step, nSkipped, epoch);

    delete[] trainFN;
}

/* 
test the model
>> fn - test data file
>> ofn - output data file
>> model - model that is trained
*/
void T2TTrainer::Test(const char * fn, const char * ofn, T2TModel * model)
{
    int wc = 0;
    int wordCount = 0;
    int wordCountTotal = 0;
    int sentCount = 0;
    float loss = 0;

    /* data files */
    FILE * file = fopen(fn, "rb");
    CheckNTErrors(file, "Cannot read the test file");
    FILE * ofile = fopen(ofn, "wb");
    CheckNTErrors(ofile, "Cannot open the output file");

    int devID = model->devID;
    XMem * mem = model->mem;

    XNet net;
    
    double startT = GetClockSec();
        
    wordCount = 0;
        
    /* batch of input sequences */
    XTensor batchEnc;
    XTensor batchDec;

    /* padding */
    XTensor paddingEnc;
    XTensor paddingDec;

    /* gold standard */
    XTensor gold;

    /* an array that keeps the sequences */
    int * seqs = new int[MILLION];
    
    ClearBuf();

    while(LoadBatch(file, model->isLM, &batchEnc, &paddingEnc, &paddingDec, &paddingDec, &gold, 
                    seqs, vSize, vSizeTgt,
                    1, 1, false, wc, devID, mem, false))
    {
        CheckNTErrors(batchEnc.order == 2, "wrong tensor order of the sequence batch");
            
        /* output probabilities */
        XTensor output;
            
        /* make the network */
        if(model->isLM)
            model->MakeLM(batchEnc, output, paddingEnc, false);
        else if(model->isMT)
            model->MakeMT(batchEnc, batchDec, output, paddingEnc, paddingDec, false);
        else{
            ShowNTErrors("Illegal model type!");
        }

        int bSize = output.GetDim(0);
        int length = output.GetDim(1);

        /* prediction probabilities */
        XTensor probs;
        InitTensor1D(&probs, bSize * length);

        /* get probabilities */
        float prob = GetProb(&output, &gold, &probs);

        /* dump the test result */
        for(int s = 0; s < bSize; s++){
            DTYPE sum = 0;
            int * seq = seqs + s * length;
            for(int i = 0; i < length; i++){
                if(seq[i] >= 0){
                    fprintf(ofile, "%d ", seq[i]);
                }
                else
                    break;
            }
            fprintf(ofile, "||| ");
            for(int i = 0; i < length; i++){
                if(seq[i] >= 0){
                    DTYPE p = probs.Get1D(s * length + i);
                    fprintf(ofile, "%.3e ", p);
                    sum += p;
                }
                else
                    break;
            }
            fprintf(ofile, "||| %e\n", sum);
        }
            
        loss += -prob;
        wordCount += wc;
        wordCountTotal += wc;
        sentCount += 1;
    }
        
    fclose(file);
    fclose(ofile);

    delete[] seqs;
    
    double elapsed = GetClockSec() - startT;

    XPRINT3(0, stderr, "[INFO] test finished (took %.1fs, word=%d, and ppl=%.3f)\n",
            elapsed,wordCountTotal, exp(loss / wordCount));
}

/* 
make a checkpoint 
>> model - the model
>> validFN - validation data file
>> modelFN - model data file
>> label - label of the model
>> id - id of the checkpoint
*/
void T2TTrainer::MakeCheckpoint(T2TModel * model, const char * validFN, const char * modelFN, const char * label, int id)
{
    char * fn = new char[MAX_LINE_LENGTH];
    char * fn2 = new char[MAX_LINE_LENGTH];
    sprintf(fn, "%s.%s.%03d", modelFN, label, id);
    sprintf(fn2, "%s.%s.%03d.output", modelFN, label, id);

    model->Dump(fn);
    if(validFN != NULL){
        T2TTrainer trainer;
        trainer.Init(argNum, argArray);
        trainer.Test(validFN, fn2, model);
    }

    delete[] fn;
    delete[] fn2;
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
int T2TTrainer::LoadBuf(FILE * file, bool isSorted, int step)
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
            int bucketCount = 0;
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
void T2TTrainer::ClearBuf()
{
    nseqBuf = 0;
    nextSeq = -1;
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
>> mem - memory pool
>> isTraining - indicates whether we are training the model
*/
int T2TTrainer::LoadBatch(FILE * file, bool isLM, 
                          XTensor * batchEnc, XTensor * paddingEnc, 
                          XTensor * batchDec, XTensor * paddingDec,
                          XTensor * gold,
                          int * seqs,
                          int vsEnc, int vsDec, int sBatch, int wBatch, 
                          bool isSorted, int &wCount,
                          int devID, XMem * mem, 
						  bool isTraining)
{
    if(isLM){
        return LoadBatchLM(file, batchEnc, paddingEnc, batchDec, paddingDec, gold, 
                           seqs, vsEnc, sBatch, wBatch, 
                           isSorted, wCount, devID, mem, isTraining);
    }
    else{
        return LoadBatchMT(file, batchEnc, paddingEnc, batchDec, paddingDec, gold, 
                           seqs, vsEnc, vsDec, sBatch, wBatch, 
                           isSorted, wCount, devID, mem, isTraining);
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
>> gold - gold standard
>> seqs - keep the sequences in an array
>> vs - vocabulary size
>> sBatch - batch size of sequences
>> wBatch - batch size of words
>> isSorted - indicates whether the sequences are sorted by length
>> wCount - word count
>> devID - device id
>> mem - memory pool
>> isTraining - indicates whether we are training the model
*/
int T2TTrainer::LoadBatchLM(FILE * file, 
                            XTensor * batchEnc, XTensor * paddingEnc,
                            XTensor * batchDec, XTensor * paddingDec,
                            XTensor * gold,
                            int * seqs,
                            int vs, int sBatch, int wBatch, 
                            bool isSorted, int &wCount,
                            int devID, XMem * mem,
							bool isTraining)
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
    dims[2] = vs;

    InitTensor2D(batchEnc, sc, max, X_INT, devID, mem);
    InitTensor(gold, 3, dims, X_FLOAT, 1.0F, devID, mem);
    InitTensor2D(paddingEnc, sc, max, X_FLOAT, devID, mem);
    InitTensor2D(paddingDec, sc, max, X_FLOAT, devID, mem);

    batchEnc->SetZeroAll();
    gold->SetZeroAll();
    paddingEnc->SetZeroAll();
    paddingDec->SetZeroAll();

    int seqSize = 0;
    int wGold = 0;
    
    int * batchEncValues = new int[batchEnc->unitNum];
    MTYPE * goldOffsets = new MTYPE[gold->unitNum];

    memset(batchEncValues, 0, sizeof(int) * batchEnc->unitNum);

    for(int s = seq; s < seq + sc; s++){
        int len = isDoubledEnd ? seqLen[s] : seqLen[s] - 1;
        CheckNTErrors(len <= max, "Something is wrong!");
        for(int w = 0; w < len; w++){
            int num = buf[seqOffset[s] + w];
            batchEncValues[(int)batchEnc->GetOffset2D(s - seq, w)] = num;

            if (w > 0)
                goldOffsets[wGold++] = gold->GetOffset3D(s - seq, w - 1, num);
            
            if (w == len - 1) {
                if (isDoubledEnd)
                    goldOffsets[wGold++] = gold->GetOffset3D(s - seq, w, num);
                else
                    goldOffsets[wGold++] = gold->GetOffset3D(s - seq, w, buf[seqOffset[s] + w + 1]);
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
    gold->SetDataBatched(goldOffsets, 1.0F, wGold);

    XTensor * tmp = NewTensorBuf(paddingEnc, devID, mem);
    _ConvertDataType(batchEnc, tmp);
    _NotEqual(tmp, paddingEnc, 0);
    DelTensorBuf(tmp);
        
    XTensor * tmp2 = NewTensorBuf(paddingDec, devID, mem);
    _ConvertDataType(batchEnc, tmp2);
    _NotEqual(tmp2, paddingDec, 0);
    DelTensorBuf(tmp2);

    delete[] batchEncValues;
    delete[] goldOffsets;

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
>> gold - gold standard
>> seqs - keep the sequences in an array
>> vsEnc - size of the encoder vocabulary
>> vsDec - size of the decoder vocabulary
>> sBatch - batch size of sequences
>> wBatch - batch size of words
>> isSorted - indicates whether the sequences are sorted by length
>> wCount - word count
>> devID - device id
>> mem - memory pool
>> isTraining - indicates whether we are training the model
*/
int T2TTrainer::LoadBatchMT(FILE * file, 
                            XTensor * batchEnc, XTensor * paddingEnc, 
                            XTensor * batchDec, XTensor * paddingDec,
                            XTensor * gold,
                            int * seqs,
                            int vsEnc, int vsDec, int sBatch, int wBatch, 
                            bool isSorted, int &wCount,
                            int devID, XMem * mem, 
							bool isTraining)
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
    int dimsDec[3] = {sCount, maxDec, vsDec};

    InitTensor2D(batchEnc, sCount, maxEnc, X_INT, devID, mem);
    InitTensor2D(paddingEnc, sCount, maxEnc, X_FLOAT, devID, mem);
    InitTensor2D(batchDec, sCount, maxDec, X_INT, devID, mem);
    InitTensor2D(paddingDec, sCount, maxDec, X_FLOAT, devID, mem);
    InitTensor(gold, 3, dimsDec, X_FLOAT, 1.0F, devID, mem);

    batchEnc->SetZeroAll();
    paddingEnc->SetZeroAll();
    batchDec->SetZeroAll();
    paddingDec->SetZeroAll();
    gold->SetZeroAll();

    int wCountEnc = 0;
    int wCountDec = 0;
    int wGold = 0;
    wCount = 0;

    int * batchEncValues = new int[batchEnc->unitNum];
    int * batchDecValues = new int[batchDec->unitNum];
    MTYPE * goldOffsets = new MTYPE[sc * maxDec / 2];

    memset(batchEncValues, 0, sizeof(int) * batchEnc->unitNum);
    memset(batchDecValues, 0, sizeof(int) * batchDec->unitNum);

    /* batch of the source-side sequences */
    for(int s = seq; s < seq + sc; s += 2){
        int len = seqLen[s];
        int sent = (s - seq)/2;
        for(int w = 0; w < len; w++){
            int num = buf[seqOffset[s] + w];
            batchEncValues[batchEnc->GetOffset2D(sent, w)] = num;
            wCountEnc++;
        }
    }

    batchEnc->SetData(batchEncValues, batchEnc->unitNum);
    XTensor * tmp = NewTensorBuf(paddingEnc, devID, mem);
    _ConvertDataType(batchEnc, tmp);
    _NotEqual(tmp, paddingEnc, 0);
    DelTensorBuf(tmp);

    /* batch of the target-side sequences */
    for(int s = seq + 1; s < seq + sc; s += 2){
        int len = isDoubledEnd ? seqLen[s] : seqLen[s] - 1;
        CheckNTErrors(len <= maxDec, "Something is wrong!");
        int sent = (s - seq - 1)/2;
        for(int w = 0; w < len; w++){
            int num = buf[seqOffset[s] + w];
            batchDecValues[batchDec->GetOffset2D(sent, w)] = num;
            
            if (w > 0)
                goldOffsets[wGold++] = gold->GetOffset3D(sent, w - 1, buf[seqOffset[s] + w]);
            
            if (w == len - 1) {
                if (isDoubledEnd)
                    goldOffsets[wGold++] = gold->GetOffset3D(sent,  w, buf[seqOffset[s] + w]);
                else
                    goldOffsets[wGold++] = gold->GetOffset3D(sent, w, buf[seqOffset[s] + w + 1]);
            }
            wCount++;
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

    XTensor * tmp2 = NewTensorBuf(paddingDec, devID, mem);
    _ConvertDataType(batchDec, tmp2);
    _NotEqual(tmp2, paddingDec, 0);
    DelTensorBuf(tmp2);

    gold->SetDataBatched(goldOffsets, 1.0F, wGold);

    delete[] batchEncValues;
    delete[] batchDecValues;
    delete[] goldOffsets;

    return sc;
}

/* 
shuffle lines of the file 
>> srcFile - the source file to shuffle
>> tgtFile - the resulting file
*/
void T2TTrainer::Shuffle(const char * srcFile, const char * tgtFile)
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
    
/*
get word probabilities for a batch of sequences
>> output - word distribution for each position
>> gold - gold standard
>> wordProbs - word probability for gold prediction
*/
float T2TTrainer::GetProb(XTensor * output, XTensor * gold, XTensor * wordProbs)
{
    XTensor probs;
    InitTensor(&probs, output);
    
    _Multiply(output, gold, &probs);
    
    /* probability of each word */
    XTensor wprobs;
    InitTensor1D(&wprobs, output->unitNum/output->GetDim(-1), X_FLOAT, output->devID, output->mem);
    
    int dims[2] = {output->unitNum/output->GetDim(-1), output->GetDim(-1)};
    probs.Reshape(2, dims);
    _ReduceSum(&probs, &wprobs, 1);
    
    if(wordProbs != NULL)
        _CopyValues(&wprobs, wordProbs);
    
    /* reshape the tensor to fit it into the reduce procedure
       TODO: XTensor supports scalars */
    dims[0] = 1;
    dims[1] = probs.unitNum;
    probs.Reshape(2, dims);
    
    /* probability for the batch */
    XTensor result;
    InitTensor1D(&result, 1, X_FLOAT, output->devID, output->mem);
    _ReduceSum(&probs, &result, 1);
    
    return result.Get1D(0);
}

/* 
update the model by delta rule
\theta_new = \theta - \lrate * grad
where
\lrate = d^-0.5 * min(stepNum^-0.5, stepNum * warmupStepNum^-1.5)
>> model - the t2t model
>> lr - learning rate
*/
void T2TTrainer::Update(T2TModel * model, const float lr)
{
    XList ws(100);

    model->GetParams(ws);

    for(int i = 0; i < ws.count; i++){
        XTensor * para = (XTensor*)ws.Get(i);
        XTensor * paraGrad = para->grad;

        if (paraGrad == NULL)
            continue;

        CheckNTErrors(para != NULL, "NULL parameter tensor!");
        CheckNTErrors(paraGrad != NULL, "NULL gradient tensor!");

        if(useAdam){
            adamBeta1T *= adamBeta1;
            adamBeta2T *= adamBeta2;
            DTYPE e = lr * (DTYPE)sqrt(1 - adamBeta2T) / (1 - adamBeta1T);
            DTYPE d = adamDelta * (DTYPE)sqrt(1 - adamBeta2T);

            /* m = beta_1 * m + (1-beta_1) * grad */
            XTensor * m = (XTensor*)moments.Get(i);
            _ScaleAndShiftMe(m, adamBeta1, 0);
            _Sum(m, paraGrad, m, (1.0F - adamBeta1));
            
            /* v = beta_2 * v + (1-beta_2) * grad * grad*/
            XTensor * v = (XTensor*)moments2nd.Get(i);
            _Multiply(paraGrad, paraGrad, v, adamBeta2/(1.0F - adamBeta2));
            _ScaleAndShiftMe(v, (1.0F - adamBeta2), 0);

            /* v2 = m / (sqrt(v) + delta) */
            XTensor * v2 = NewTensorBuf(v, v->devID, v->mem);
            _Power(v, v2, 0.5F);
            _ScaleAndShiftMe(v2, 1.0F, d);
            _Div(m, v2, v2);

            /* the delta rule */
            _Sum(para, v2, para, -e);

            DelTensorBuf(v2);

        }
        else{
            /* the delta rule */
            _Sum(para, paraGrad, para, -lr);
        }

        /* clear gradient */
        paraGrad->SetZeroAll();
    }
}

/* 
prepare model for training 
>> model - the model for training
*/
void T2TTrainer::PrepareModel(T2TModel * model)
{
    moments.Clear();
    moments2nd.Clear();

    XList ws(100);

    model->GetParams(ws);

    for(int i = 0; i < ws.count; i++){
        XTensor * para = (XTensor*)ws.Get(i);
        XNoder::MakeGrad(para);

        if(useAdam){
            XTensor * m = new XTensor(para);
            XTensor * m2 = new XTensor(para);
            m->SetZeroAll();
            m2->SetZeroAll();
            moments.Add(m);
            moments2nd.Add(m2);
        }
    }

    adamBeta1T = 1.0F;
    adamBeta2T = 1.0F;
}

/* 
do padding on the output 
>> output - output tensor of the network
>> gold - gold standard
>> padding - padding of a batch of sentences
>> lsP - smoothing factor
*/
void T2TTrainer::PadOutput(XTensor * output, XTensor * gold, XTensor * padding)
{
    if(output == NULL || padding == NULL)
        return;
    
    int on = output->order;
    int * dimso = new int[on];

    memcpy(dimso, output->dimSize, sizeof(int) * on);

    output->Reshape(output->unitNum/dimso[output->order - 1], dimso[output->order - 1]);

    XTensor * padding2 = NewTensorBuf(1, &padding->unitNum, X_FLOAT, 1.0F, padding->devID, padding->mem);

    _CopyValues(padding, padding2);
    _MultiplyDim(output, padding2, output, 0);
    _ScaleAndShiftMe(padding2, 1e9F, -1e9F);
    _SumDim(output, padding2, output, 0);
    
    output->Reshape(on, dimso);
    
    if(gold != NULL){
        gold->Reshape(gold->unitNum/dimso[gold->order - 1], dimso[gold->order - 1]);
        _CopyValues(padding, padding2);
        _MultiplyDim(gold, padding2, gold, 0);
        gold->Reshape(on, dimso);
    }

    delete[] dimso;
    DelTensorBuf(padding2);
}

/*
recale the output and gold tensors for normalized loss
>> output - output tensor of the network
>> gold - gold standard
>> padding - padding of a batch of sentences
*/
void T2TTrainer::RescaleOutput(XTensor * output, XTensor * gold, XTensor * padding)
{
    CheckNTErrors(output->order == 3, "Wrong dimension number!");
    CheckNTErrors(gold->order == 3, "Wrong dimension number!");

    DTYPE count = _ReduceSumAll(padding);
    
    _ExpMe(output);
    _ScaleAndShiftMe(output, 1/count);
    _LogMe(output);

    _ScaleAndShiftMe(gold, 1/count);
}
    
/*
perform label smoothing
>> gold - gold standard
>> smoothed - result of label smoothing
>> p - smoothing factor
*/
void T2TTrainer::LabelSmooth(XTensor * gold, XTensor * smoothed, DTYPE p)
{
    CheckNTErrors(p >= 0 && p <= 1.0F, "Smoothing factor must be in range [0,1]");
    
    int n = gold->GetDim(-1);
    DTYPE q = 1.0F - p;
    DTYPE gift = p / n;
    
    InitTensor(smoothed, gold);
    _CopyValues(gold, smoothed);
    
    if(p == 0)
        return;

    _ScaleAndShiftMe(smoothed, q, gift);
}

}
