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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-08-01
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-07
 */

#ifndef __EMBEDDING_H__
#define __EMBEDDING_H__

#include "../Config.h"
#include "../../../network/XNet.h"

using namespace nts;

/* the nmt namespace */
namespace nmt
{

#define DEFAULT_EMBEDDING_SIZE 512

/*
embedding (of word at position i):
word embedding + positional embedding
*/
class Embedder
{
public:
    /* indicates whether it is run with fp16 */
    bool fp16;

    /* indicates whether it is shared with other modules */
    bool shareEncDecEmb;

    /* indicates whether train the model */
    bool isTraining;

    /* device id */
    int devID;

    /* vocabulary size */
    int vSize;

    /* embedding size */
    int eSize;

    /* maximum length of the sequence */
    int maxLength;

    /* padding index */
    int padIdx;

    /* word embedding matrix */
    XTensor* w;

    /* predefined positional embeddings. It can speeds up
       the embedding processing by re-loading. */
    XTensor posEmbeddingBase;

public:
    /* set the training flag */
    void SetTrainingFlag(bool myIsTraining);

    /* constructor */
    Embedder();

    /* de-constructor */
    ~Embedder();

    /* initialize the model */
    void InitModel(NMTConfig& config, bool isEnc = true);

    /* make positional embeddings */
    void MakePosEmbedding(int length);

    /* make the network */
    XTensor Make(XTensor& input, bool isDec, int nstep);
};

} /* end of the nmt namespace */

#endif /* __EMBEDDING_H__ */