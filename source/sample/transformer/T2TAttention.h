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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-31
 */

#ifndef __T2TATTENTION_H__
#define __T2TATTENTION_H__

#include "../../network/XNet.h"

using namespace nts;

namespace transformer
{

/* 
multi-head attention 
y(Q, K, V) = cat(head_1, head_2, ..., head_n)
where head_i = Attention(Q * w_i^Q, K * w_i^K, V * w_i^V)
      attention(Q, K, V) = softmax(Q * K^T/d_k^0.5) V
      d_k = dimension size of K
*/
class T2TAttention
{
public:
    /* device id */
    int devID;
    
    /* head number */
    int nhead;

    /* transformation matrix for K */
    XTensor wk;

    /* transformation matrix for Q */
    XTensor wq;

    /* transformation matrix for V */
    XTensor wv;

    /* transformation after dot-product attention */
    XTensor wa;
    
    XTensor wbig;
    
    /* size of transformed Q and K */
    int dk;

    /* size of transformed V */
    int dv;

    /* size of input Q, K and V */
    int d;

    /* indicates whether the attention is masked */
    bool isMasked;

    /* some positions can be ignored in attention. this is useful in lm where the first position needs
       special design for the attention model. */
    int ignored;

    /* indicates whether the model is used for training */
    bool isTraining;
    
    /* dropout probability */
    DTYPE dropoutP;

public:
    /* constructor */
    T2TAttention();

    /* de-constructor */
    ~T2TAttention();

    /* initialize the model */
    void InitModel(int argc, char ** argv, 
                   bool myIsMasked, int myIgnored, 
                   int myDevID = -1);

    /* make the network */
    XTensor Make(XTensor &k, XTensor &q, XTensor &v, XTensor &mask, bool isTraining);
    
    /* make the network given a big tensor that keeps keys, queries and values */
    XTensor MakeBig(XTensor &kqv, XTensor &mask, bool isTraining);
    
    /* make the attention network given keys, queries and values (after linear transformation) */
    XTensor MakeAttention(XTensor &k, XTensor &q, XTensor &v, XTensor &mask, bool isTraining);
};

}

#endif
