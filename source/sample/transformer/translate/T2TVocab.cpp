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

#include <fstream>

#include "T2TVocab.h"
#include "../module/T2TUtility.h"

namespace nts {

/* load a vocabulary from a file */
void Vocab::Load(const string& src)
{
    string vsz, sid;
    ifstream f(src, ios::in);
    CheckNTErrors(f.is_open(), "Unable to open the vocabulary file");

    /* get the vocab size and the start id */
    f >> vsz >> sid;
    startID = stol(sid);
    vocabSize = stol(vsz);

    string word, id;
    for (int i = 0; i < vocabSize - startID; i++) {
        f >> word >> id;
        word2id[word] = stol(id);
        id2word[stol(id)] = word;
    }

    f.close();
}

/* save a vocabulary to a file */
void Vocab::Save(const string& src)
{
    ofstream f(src, ios::out);

    /* the first line: size of the vocab and the start id */
    f << vocabSize << "\t" << startID;

    /* other lines: words and indices */
    for (const auto& p : word2id)
        f << p.first << "\t" << p.second;

    f.close();
}

/*
copy data from another vocabulary
>> v - the target vocabulary
*/
void Vocab::CopyFrom(const Vocab& v)
{
    for (const auto& w2i : v.word2id)
        word2id.insert(w2i);

    for (const auto& i2w : v.id2word)
        id2word.insert(i2w);
}

}