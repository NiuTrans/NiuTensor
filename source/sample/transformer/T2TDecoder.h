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

#ifndef __T2TDECODER_H__
#define __T2TDECODER_H__

#include "T2TEncoder.h"

namespace transformer
{

class AttDecoder : public AttEncoder
{
public:
    /* encoder-decoder attention model of each layer */
    T2TAttention * attentionsEnde;

    /* layer normalization for encoder-decoder attention */
    T2TLN * attEndeLayerNorms;
public:
    /* constructor */
    AttDecoder();

    /* deconstructor */
    ~AttDecoder();

    /* initialize the model */
    void InitModel(int argc, char ** argv, 
                   bool myIsMasked, int myIgnored, 
                   int myDevID = -1, XMem * myMem = NULL);

    /* make the decoding network */
    XTensor Make(XTensor &inputDec, XTensor &outputEnc, XTensor &mask, XTensor &maskEncDec, bool isTraining);
};

}

#endif