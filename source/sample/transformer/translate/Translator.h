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
This class translates test sentences with a trained model. 
It will dump the result to the output file if specified, else the standard output.
*/

#ifndef __TRANSLATOR_H__
#define __TRANSLATOR_H__

#include "../Model.h"
#include "Searcher.h"
#include "TranslateDataSet.h"

/* the nmt namespace */
namespace nmt
{

class Translator
{
private:
    /* translate a batch of sequences */
    void TranslateBatch(XTensor& batchEnc, XTensor& paddingEnc, IntList& indices);

private:
    /* the translation model */
    NMTModel* model;

    /* for batching */
    TranslateDataset batchLoader;

    /* the searcher for translation */
    void* seacher;

    /* configuration of the NMT system */
    NMTConfig* config;

    /* output buffer */
    XList* outputBuf;

public:
    /* constructor */
    Translator();

    /* de-constructor */
    ~Translator();

    /* initialize the translator */
    void Init(NMTConfig& myConfig, NMTModel& myModel);

    /* the translation function */
    bool Translate();

    /* sort the outputs by the indices (in ascending order) */
    void SortOutputs();

    /* dump the translations to a file */
    void DumpResToFile(const char* ofn);

    /* dump the translations to stdout */
    void DumpResToStdout();
};

} /* end of the nmt namespace */

#endif /*  */