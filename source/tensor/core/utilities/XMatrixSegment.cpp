/* NiuTrans.Tensor - an open-source tensor library
* Copyright (C) 2017, Natural Language Processing Lab, Northestern University.
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
* $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-04-24
*/

#include <stdarg.h>
#include <math.h>
#include "XMatrixSegment.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
segment a 2d tensor (i.e., matrix) into blocks and run jobs in parallel
>> parallelRunner - parallel runner
>> job - the function to run
>> opNum - number of operations
>> rowNum - number of rows
>> colNum - number of columns
>> argNum - number of arguments of the jobs
>> ... - arguments of the jobs
*/
void RunParallel2D(XPRunner * parallelRunner, void * job,
                   int opNum, int rowNum, int colNum, int argNum, ...)
{
    if (rowNum == 0 || colNum == 0)
        return;

    int jobNum = 1;

    if (parallelRunner != NULL && (parallelRunner->method == PRUNNER_SINGLE || parallelRunner->method == PRUNNER_MULTIPLE)) {
        if (opNum >= parallelRunner->minimumOPNum * parallelRunner->threadNum)
            jobNum = parallelRunner->GetJobNum(rowNum * colNum);
    }

    CheckNTErrors(jobNum != 0, "TODO!");

    /* argument list of the jobs */
    TensorList * jobArgList = new TensorList(argNum);

    va_list ap;
    va_start(ap, argNum);
    for (int i = 0; i < argNum; i++) {
        XTensor* p = va_arg(ap, XTensor*);
        jobArgList->Add(p);
    }
    va_end(ap);

    /* prepare the neccesary argument list for parallel processing */
    TensorList * jobs = new TensorList(jobNum);
    TensorList * args = new TensorList(jobNum);

    int * indexList = new int[jobNum * 4 * 4];

    /* segment the matrix into blocks */
    int nblock = SegmentTensor2D(rowNum, colNum, jobNum, indexList);

    /*
    assign jobs
    argument rules:
    1. block information
    2. other arguments
    */
    for (int i = 0; i < jobNum; i++) {
        IntList* indexArgs = new IntList(4);
        TensorList * blockArgs = new TensorList(argNum);
        int * blockIndex = indexList + i * 4;

        indexArgs->Add(blockIndex[0]);
        indexArgs->Add(blockIndex[1]);
        indexArgs->Add(blockIndex[2]);
        indexArgs->Add(blockIndex[3]);

        for (int j = 0; j < argNum; j++)
            blockArgs->Add(jobArgList->GetItem(j));

        args->Add((XTensor*)indexArgs);
        args->Add((XTensor*)blockArgs);

        jobs->Add((XTensor*)job);
    }

    args->count = jobNum * 2;
    jobs->count = nblock;

    /* single job */
    if (jobNum == 1)
        ((TFunction)job)(args);
    /* multiple jobs */
    else
        parallelRunner->Run(jobs, args);

    /* free the memory */
    delete[] indexList;
    for (int i = 0; i < args->count; i++) {
        TensorList * blockArgs = (TensorList*)args->GetItem(i);
        delete blockArgs;
    }
    delete args;
    delete jobs;
    delete jobArgList;
}

/*
segment a block into sub-blocks
>> rowNum - number of rows
>> colNum - number of columns
>> blockNum - number of sub-blocks
>> blockIndex - upper-left and bottom-right corners of each sub-block
<< return - the number of resulting sub-blocks
*/
int SegmentTensor2D(int rowNum, int colNum, int blockNum, int * blockIndex)
{
    int total = rowNum * colNum;
    int rowSize = (int)ceil(sqrt((float)total / blockNum));
    int colSize = rowSize;

    /* a narrow matrix */
    if (rowSize > colNum * 0.9) {
        rowSize = colNum;
        colSize = (int)ceil((float)rowNum / blockNum);
    }

    /* a narrow matrix */
    if (colSize > rowNum * 0.9) {
        colSize = rowNum;
        rowSize = (int)ceil((float)colNum / blockNum);
    }

    if (blockNum == 1) {
        colSize = rowNum;
        rowSize = colNum;
    }

    CheckNTErrors((colSize <= rowNum && rowSize <= colNum),
        "Too large block!");

    int x1, y1, x2, y2;
    int xMax = rowNum - 1;
    int yMax = colNum - 1;
    int nblock = 0, nitem = 0;
    int * indexList = blockIndex;

    int xSegNum = int((float)rowNum / colSize);
    int ySegNum = int((float)colNum / rowSize);
    int marginBlockNum = blockNum - xSegNum * ySegNum;

    /*
    To maximize the number of resulting sub-block, we have to
    make use of the margin block
    */
    if (blockNum > 1 && marginBlockNum > 0) {
        int margin = 0;
        int step = 0;
        if (rowNum < colNum) {
            margin = int(((float)marginBlockNum / blockNum) * colNum);
            step = (int)ceil((float)rowNum / marginBlockNum);
            x1 = 0;
            y1 = yMax - margin + 1;
            x2 = step - 1;
            y2 = yMax;
            while (x2 <= xMax) {
                int * blockIndex = indexList + nblock * 4;
                blockIndex[0] = x1; blockIndex[1] = y1;
                blockIndex[2] = x2; blockIndex[3] = y2;
                nblock++;
                nitem += (y2 - y1 + 1) * (x2 - x1 + 1);

                if (x2 == xMax)
                    break;

                x1 = x2 + 1;
                x2 = x1 + step - 1;

                if (x2 > xMax)
                    x2 = xMax;
            }

            yMax -= margin;
        }
        else {
            margin = int(((float)marginBlockNum / blockNum) * rowNum);
            step = (int)ceil((float)colNum / marginBlockNum);
            x1 = xMax - margin + 1;
            y1 = 0;
            x2 = xMax;
            y2 = step - 1;
            while (y2 <= yMax) {
                int * blockIndex = indexList + nblock * 4;
                blockIndex[0] = x1; blockIndex[1] = y1;
                blockIndex[2] = x2; blockIndex[3] = y2;
                nblock++;
                nitem += (y2 - y1 + 1) * (x2 - x1 + 1);

                if (y2 == yMax)
                    break;

                y1 = y2 + 1;
                y2 = y1 + step - 1;

                if (y2 > yMax)
                    y2 = yMax;
            }

            xMax -= margin;
        }

        colSize = (int)ceil((float)(xMax + 1) / xSegNum);
        rowSize = (int)ceil((float)(yMax + 1) / ySegNum);

    }

    x1 = 0;
    y1 = 0;            // upper-left corner
    x2 = colSize - 1;
    y2 = rowSize - 1;  // bottom-right corner

    /* the main body of the matrix (after removing the margin block) */
    while (x1 <= xMax) {
        y1 = 0;
        x2 = x1 + colSize - 1;
        y2 = y1 + rowSize - 1;

        if (x2 > xMax) {
            x2 = xMax;
        }

        while (y2 <= yMax) {
            int * blockIndex = indexList + nblock * 4;
            blockIndex[0] = x1; blockIndex[1] = y1;
            blockIndex[2] = x2; blockIndex[3] = y2;
            nblock++;
            nitem += (y2 - y1 + 1) * (x2 - x1 + 1);

            if (y2 == yMax)
                break;

            y1 = y2 + 1;
            y2 = y1 + rowSize - 1;

            if (y2 > yMax)
                y2 = yMax;

            CheckNTErrors((nblock <= blockNum),
                "Fail to segment the matrix!");
        }

        x1 = x2 + 1;
    }

    CheckNTErrors(nitem == rowNum * colNum,
        "Fail to segment the matrix!");

    return nblock;
}

/*
segment a block into sub-blocks (each block consists of a number of rows)
>> rowNum - number of rows
>> colNum - number of columns
>> blockNum - number of sub-blocks
>> blockIndex - upper-left and bottom-right corners of each sub-block
<< return - the number of resulting sub-blocks
*/
int SegmentTensor2DInRows(int rowNum, int colNum, int blockNum, int * blockIndex)
{
    if (rowNum < blockNum) {
        blockIndex[0] = 0;
        blockIndex[1] = 0;
        blockIndex[2] = rowNum - 1;
        blockIndex[3] = colNum - 1;
        return 1;
    }

    int segSize = (int)ceil((float)rowNum / blockNum);
    int x1 = 0;
    int x2 = x1 + segSize - 1;
    int y1 = 0;
    int y2 = colNum - 1;
    int last = rowNum - 1;
    int nblock = 0;

    while (x1 <= last) {
        x2 = x1 + segSize - 1;

        if (x2 > last) {
            x2 = last;
        }

        int * blockInfo = blockIndex + 4 * nblock;
        blockInfo[0] = x1;
        blockInfo[1] = y1;
        blockInfo[2] = x2;
        blockInfo[3] = y2;
        nblock++;

        if (x2 == last)
            break;

        x1 += segSize;
    }

    return nblock;
}
} // namespace nts(NiuTrans.Tensor)
