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
 * $Created by: HU Chi (huchinlp@gmail.com) 2021-06
 */

#include <algorithm>
#include "DataSet.h"

using namespace nts;

/* the nmt namespace */
namespace nmt {

/* get the maximum source sentence length in a range of buffer */
int DataSetBase::MaxSrcLen(int begin, int end) {
    CheckNTErrors((end > begin) && (begin >= 0)
                 && (end <= buf->count), "Invalid range");
    int maxLen = 0;
    for (int i = begin; i < end; i++) {
        IntList* srcSent = ((Sample*)buf->Get(i))->srcSeq;
        maxLen = MAX(int(srcSent->Size()), maxLen);
    }
    return maxLen;
}

/* get the maximum target sentence length in a range of buffer */
int DataSetBase::MaxTgtLen(int begin, int end) {
    CheckNTErrors((end > begin) && (begin >= 0)
                 && (end <= buf->count), "Invalid range");
    int maxLen = 0;
    for (int i = begin; i < end; i++) {
        IntList* tgtSent = ((Sample*)buf->Get(i))->tgtSeq;
        maxLen = MAX(int(tgtSent->Size()), maxLen);
    }
    return maxLen;
}

/* sort the buffer by source sentence length (in ascending order) */
void DataSetBase::SortBySrcLengthAscending() {
	stable_sort(buf->items, buf->items + buf->count,
		[](void* a, void* b) {
			return ((Sample*)(a))->srcSeq->Size() <
				   ((Sample*)(b))->srcSeq->Size();
		});
}

/* sort the buffer by target sentence length (in ascending order) */
void DataSetBase::SortByTgtLengthAscending()
{
	stable_sort(buf->items, buf->items + buf->count,
		[](void* a, void* b) {
			return ((Sample*)(a))->tgtSeq->Size() <
				   ((Sample*)(b))->tgtSeq->Size();
		});
}

/* sort the buffer by source sentence length (in descending order) */
void DataSetBase::SortBySrcLengthDescending() {
    stable_sort(buf->items, buf->items + buf->count,
        [](void* a, void* b) {
            return ((Sample*)(a))->srcSeq->Size() >
                   ((Sample*)(b))->srcSeq->Size();
        });
}

/* sort the buffer by target sentence length (in descending order) */
void DataSetBase::SortByTgtLengthDescending()
{
    stable_sort(buf->items, buf->items + buf->count,
        [](void* a, void* b) {
            return ((Sample*)(a))->tgtSeq->Size() >
                   ((Sample*)(b))->tgtSeq->Size();
        });
}

/*
clear the buffer
>> buf - the buffer (list) of samples
*/
void DataSetBase::ClearBuf()
{
    bufIdx = 0;
    for (int i = 0; i < buf->count; i++) {
        Sample* sample = (Sample*)buf->Get(i);
        delete sample;
    }
    buf->Clear();
}

/* constructor */
DataSetBase::DataSetBase()
{
    wc = 0;
    sc = 0;
    bufIdx = 0;
    config = NULL;
    buf = new XList();
}

/* de-constructor */
DataSetBase::~DataSetBase()
{
    if (buf != NULL) {
        ClearBuf();
        delete buf;
    }
}

/* constructor */
Sample::Sample(IntList* s, IntList* t, int myKey)
{
    index = -1;
    srcSeq = s;
    tgtSeq = t;
    bucketKey = myKey;
}

/* de-constructor */
Sample::~Sample()
{
    if (srcSeq != NULL)
        delete srcSeq;
    if (tgtSeq != NULL)
        delete tgtSeq;
}

} /* end of the nmt namespace */