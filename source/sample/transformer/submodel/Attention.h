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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-31
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04, 2020-06
 */

#ifndef __ATTENTION_H__
#define __ATTENTION_H__

#include "NNUtil.h"
#include "../Config.h"
#include "../../../network/XNet.h"
#include "../../../tensor/core/CHeader.h"

using namespace nts;

/* the nmt namespace */
namespace nmt
{
/* attention type */
enum { NONE, SELF_ATT, EN_DE_ATT };

/* layer cache for keys and values */
class Cache
{
public:
    /* cache for keys, (B, L, H) */
    XTensor key;

    /* cache for values, (B, L, H) */
    XTensor value;

public:
    /* indicates cache miss if 'true' */
    bool miss;

    /* indicates whether we use cache */
    bool enable;

    /* constructor */
    Cache();

    /* update the states cache */
    void Update(XTensor&& k, XTensor&& v);

    /* keep alive states */
    void KeepAlive(XTensor& aliveIdx);

    /* reorder alive states */
    void Reorder(XTensor& reorder);
};

/* multi-head attention */
class Attention
{
public:
    /* indicates whether train the model */
    bool isTraining;

    /* device id */
    int devID;

    /* head number */
    int nhead;

    /* transformation matrix for Q */
    XTensor weightQ;

    /* bias for Q */
    XTensor biasQ;

    /* transformation matrix for K */
    XTensor weightK;

    /* bias for K */
    XTensor biasK;

    /* transformation matrix for V */
    XTensor weightV;

    /* bias for V */
    XTensor biasV;

    /* RPR emb */
    XTensor RPEmbK;

    /* transformation after dot-product attention */
    XTensor weightO;

    /* bias after dot-product attention */
    XTensor biasO;

    /* size of transformed Q and K */
    int kDim;

    /* size of transformed V */
    int vDim;

    /* size of input Q, K and V */
    int embDim;

    /* indicates whether we use the RPR attention */
    bool useRPR;

    /* dropout probability */
    DTYPE dropoutP;

    /* the maximum relative window size */
    int maxRP;

public:
    /* set the training flag */
    void SetTrainingFlag(bool myIsTraining);

    /* constructor */
    Attention();

    /* de-constructor */
    ~Attention();

    /* initialize the model */
    void InitModel(NMTConfig& config, bool isEnc, bool isSelfAtt);

    /* make the network */
    XTensor Make(XTensor& k, XTensor& q, XTensor& v,
                 XTensor* mask, Cache* cache, int cacheType);

    /* make the attention network given keys, queries and values (after linear transformation) */
    XTensor MakeAttention(XTensor& k, XTensor& q, XTensor& v, XTensor* mask, bool isEnc);

    /* make the attention network given keys, queries and values (after linear transformation) */
    XTensor MakeRPRAttention(XTensor& k, XTensor& q, XTensor& v, XTensor* mask, bool isEnc);

    /* generate relative position embeddings */
    XTensor GetRPEmbedding(int lenQ, int lenKV, bool isEnc);

    /* relative position-aware dot-product attention inner calculation */
    XTensor RPDotProduct(XTensor& x, XTensor& y, XTensor& z, const bool is_key);
};

} /* end of the nmt namespace */

#endif /* __ATTENTION_H__ */
