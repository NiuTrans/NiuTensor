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

#ifndef __T2TLAYERNORMAL_H__
#define __T2TLAYERNORMAL_H__

#include "../../network/XNet.h"

using namespace nts;

namespace transformer
{

/* layer normalization: y = norm(x) * w + b 
   where norm(x) = (x - mean)/standardDeviation */
class T2TLN
{
public:
    /* device id */
    int devID;

    /* the transformation matrix w */
    XTensor w;

    /* the bias term b */
    XTensor b;

    /* dimension size of the model */
    int d;
    
public:
    /* constructor */
    T2TLN();
    
    /* de-constructor */
    ~T2TLN();
    
    /* initialize the model */
    void InitModel(int argc, char ** argv, int myDevID = -1);
    
    /* make the network */
    XTensor Make(XTensor &input);
};

}

#endif
