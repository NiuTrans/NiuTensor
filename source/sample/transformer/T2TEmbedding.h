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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-08-01
 */

#ifndef __T2TEMBEDDING_H__
#define __T2TEMBEDDING_H__

#include "../../network/XNet.h"

using namespace nts;

namespace transformer
{

#define DEFAULT_EMBEDDING_SIZE 512

/* 
embedding (of word at position i):
word embedding + positional embedding
*/
class T2TEmbedder
{
public:
    /* device id */
    int devID;
    
    /* vocabulary size */
    int vSize;

    /* embedding size */
    int eSize;

    /* maximum length of the sequence */
    int maxLength;

    /* dimension size of the hidden layers in the t2t model */
    int d;

    /* word embedding matrix */
    XTensor w;

    /* predefined positional embeddings. It can speeds up 
       the embedding processing by re-loading. */
    XTensor posEmbeddingBase;

public:
    /* constructor */
    T2TEmbedder();

    /* de-constructor */
    ~T2TEmbedder();

    /* initialize the model */
    void InitModel(int argc, char ** argv, int myDevID = -1, bool isEnc = true);

    /* make positional embeddings */
    void MakePosEmbedding(int eSize, int d, int length);

    /* make the network */
    XTensor Make(XTensor &input);
};

}

#endif
