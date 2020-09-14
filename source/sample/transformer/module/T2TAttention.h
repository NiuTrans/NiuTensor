/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2020, Natural Language Processing Lab, Northeastern University.
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

#ifndef __T2TATTENTION_H__
#define __T2TATTENTION_H__

#include "T2TNNUtil.h"
#include "T2TUtility.h"
#include "../../../network/XNet.h"
#include "../../../tensor/core/CHeader.h"

using namespace nts;

namespace transformer
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
class T2TAttention
{
public:
    /* device id */
    int devID;

    /* head number */
    int nhead;

    /* transformation matrix for Q */
    XTensor wq;

    /* bias for Q */
    XTensor bq;

    /* transformation matrix for K */
    XTensor wk;

    /* bias for K */
    XTensor bk;

    /* transformation matrix for V */
    XTensor wv;

    /* bias for V */
    XTensor bv;

    XTensor wBig;

    XTensor bBig;

    /* RPR emb */
    XTensor RPEmbK;

    /* transformation after dot-product attention */
    XTensor wo;

    /* bias after dot-product attention */
    XTensor bo;

    /* size of transformed Q and K */
    int dk;

    /* size of transformed V */
    int dv;

    /* size of input Q, K and V */
    int d;

    /* indicates whether we use the RPR attention */
    bool useRPR;

    /* dropout probability */
    DTYPE dropoutP;

    /* the maximum relative window size */
    int maxRP;

public:
    /* constructor */
    T2TAttention();

    /* de-constructor */
    ~T2TAttention();

    /* initialize the model */
    void InitModel(T2TConfig& config);

    /* make the network */
    XTensor Make(XTensor& k, XTensor& q, XTensor& v,
                 XTensor* mask, bool isTraining,
                 Cache* cache, int cacheType);

    /* make the attention network given keys, queries and values (after linear transformation) */
    XTensor MakeAttention(XTensor& k, XTensor& q, XTensor& v,
                          XTensor* mask, bool isTraining);

    /* make the attention network given keys, queries and values (after linear transformation) */
    XTensor MakeRPRAttention(XTensor& k, XTensor& q, XTensor& v,
                             XTensor* mask, bool isTraining, bool isEnc);

    XTensor GetRPEmbedding(const int lenQ, const int lenKV, const int maxRelativeLen, const bool isEnc);

    XTensor RPDotProduct(XTensor& x, XTensor& y, XTensor& z, const bool is_key);
};
}

#endif
