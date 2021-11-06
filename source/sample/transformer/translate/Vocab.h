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
 * $Created by: HU Chi (huchinlp@foxmail.com) 2020-01-03
 */

#ifndef __VOCAB_H__
#define __VOCAB_H__

#include <cstdio>
#include <unordered_map>

using namespace std;

/* the nmt namespace */
namespace nmt {

/* the vocabulary class */
struct Vocab
{
    /* id of start-of-sequence token */
    int sosID;

    /* id of end-of-sequence token */
    int eosID;

    /* id of paddings */
    int padID;

    /* id of unknown tokens */
    int unkID;

    /* size of the vocabulary */
    int vocabSize;

    /* a dict that maps tokens to ids */
    unordered_map<string, int> token2id;

    /* a dict that maps ids to words */
    unordered_map<int, string> id2token;

    /* set ids for special tokens */
    void SetSpecialID(int sos, int eos, int pad, int unk);

    /* load a vocabulary from a file */
    void Load(const string& vocabFN);

    /* save a vocabulary to a file */
    void Save(const string& vocabFN);

    /* copy data from another vocab */
    void CopyFrom(const Vocab& v);

    /* constructor */
    Vocab();
};

} /* end of the nmt namespace */

#endif /* __VOCAB_H__ */