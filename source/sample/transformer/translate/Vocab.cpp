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
#include "Vocab.h"
#include "../Config.h"

/* the nmt namespace */
namespace nmt {

/* set ids for special tokens */
void Vocab::SetSpecialID(int sos, int eos, int pad, int unk)
{
    sosID = sos;
    eosID = eos;
    padID = pad;
    unkID = unk;
}

/* load a vocabulary from a file */
void Vocab::Load(const string& vocabFN)
{
    string vsz, sid;
    ifstream f(vocabFN, ios::in);
    CheckNTErrors(f.is_open(), "Failed to open the vocabulary file");

    /* get the vocab size and the start id */
    f >> vsz >> sid;
    sosID = (int)stol(sid);
    vocabSize = (int)stol(vsz);

    string word, id;
    for (int i = 0; i < vocabSize - sosID; i++) {
        f >> word >> id;
        token2id[word] = (int)stol(id);
        id2token[(int)stol(id)] = word;
    }
    for (int i = 0; i < sosID; i++) {
        id2token[i] = "";
    }

    f.close();
}

/* save a vocabulary to a file */
void Vocab::Save(const string& vocabFN)
{
    ofstream f(vocabFN, ios::out);

    /* the first line: size of the vocab and the start id */
    f << vocabSize << "\t" << sosID;

    /* other lines: words and indices */
    for (const auto& p : token2id)
        f << p.first << "\t" << p.second;

    f.close();
}

/*
copy data from another vocabulary
>> v - the target vocabulary
*/
void Vocab::CopyFrom(const Vocab& v)
{
    for (const auto& w2i : v.token2id)
        token2id.insert(w2i);

    for (const auto& i2w : v.id2token)
        id2token.insert(i2w);
}

/* constructor */
Vocab::Vocab()
{
    sosID = -1;
    eosID = -1;
    padID = -1;
    unkID = -1;
    vocabSize = -1;
}

} /* end of the nmt namespace */
