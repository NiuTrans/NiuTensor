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

#include <math.h>
#include "T2TAttention.h"
#include "T2TUtility.h"
#include "T2TEmbedding.h"
#include "../../tensor/core/CHeader.h"

namespace transformer
{

/* constructor */
T2TAttention::T2TAttention()
{
    nhead = -1;
    dk = -1;
    dv = -1;
    d  = -1;
    isMasked = false;
    ignored = 0;
}

/* deconstructor */
T2TAttention::~T2TAttention()
{
}

/* 
initialize the model 
>> argc - number of arguments
>> argv - list of pointers to the arguments
>> myIgnored - number of position ignored in attention (from the begining)
>> myIsMasked - indicates whether the attention is with a mask
>> myDevID - device id
>> myMem - the memory pool
*/
void T2TAttention::InitModel(int argc, char ** argv, 
                             bool myIsMasked, int myIgnored, 
                             int myDevID, XMem * myMem)
{
    devID = myDevID;
    mem = myMem;
    isMasked = myIsMasked;
    ignored = myIgnored;
    
    float minmax = 0;

    LoadParamInt(argc, argv, "nhead", &nhead, 8);
    LoadParamInt(argc, argv, "d", &dk, DEFAULT_EMBEDDING_SIZE);
    LoadParamInt(argc, argv, "d", &dv, DEFAULT_EMBEDDING_SIZE);
    LoadParamInt(argc, argv, "d", &d, DEFAULT_EMBEDDING_SIZE);
    LoadParamFloat(argc, argv, "attminmax", &minmax, 0.1F);
    LoadParamFloat(argc, argv, "dropoutatt", &dropoutP, 0);

    InitTensor2D(&wk, d, dk, X_FLOAT, devID, mem);
    InitTensor2D(&wq, d, dk, X_FLOAT, devID, mem);
    InitTensor2D(&wv, d, dv, X_FLOAT, devID, mem);
    InitTensor2D(&wa, d, d, X_FLOAT, devID, mem);
    
    float scale = 1.0F;
    float finfoutk = (float)sqrt(6.0F * scale/(d + dk));
    float finfoutv = (float)sqrt(6.0F * scale/(d + dv));
    float finfouta = (float)sqrt(6.0F * scale / (d + d));

    wk.SetDataRand(-finfoutk, finfoutk);
    wq.SetDataRand(-finfoutk, finfoutk);
    wv.SetDataRand(-finfoutv, finfoutv);
    wa.SetDataRand(-finfouta, finfouta);
}

/* 
make the network 
>> k - keys. It might be of size B * L * H
       where B = batch size, L = sequence length, 
       and H = vector size of each position
>> q - queries
>> v - values
>> mask - as it is
>> isTraining - indicates whether the model is used for training
<< return - multi-attention result
*/
XTensor T2TAttention::Make(XTensor &k, XTensor &q, XTensor &v, XTensor &mask, bool isTraining)
{
    XTensor k2;
    XTensor q2;
    XTensor v2;

    /* linear transofmration before self-attention */
    k2 = MMul(k, wk);
    q2 = MMul(q, wq);
    v2 = MMul(v, wv);

    XTensor kheads;
    XTensor qheads;
    XTensor vheads;

    /* multi head */
    kheads = Split(k2, k2.order - 1, nhead);
    qheads = Split(q2, q2.order - 1, nhead);
    vheads = Split(v2, v2.order - 1, nhead);

    XTensor att;
    XTensor dot;
    XTensor scalar;

    /* scalar = softmax(Q * K^T / sqrt(dk)) * V */
    dot = BMMul(qheads, X_NOTRANS, kheads, X_TRANS);

    if(isMasked)
        dot = dot + mask;

    dot = Linear(dot, 1.0F/(float)sqrt((float)dk/nhead));

    scalar = Softmax(dot, -1);
    
    if(isTraining && dropoutP > 0)
        scalar = Dropout(scalar, dropoutP);

    att = BMMul(scalar, vheads);

    /* concatenate the heads */
    return MMul(Merge(att, att.order - 1), wa);
}

}
