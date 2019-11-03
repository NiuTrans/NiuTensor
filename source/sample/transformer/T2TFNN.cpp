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
#include "T2TFNN.h"
#include "T2TUtility.h"
#include "T2TEmbedding.h"
#include "../../tensor/core/CHeader.h"
#include "../../tensor/function/FHeader.h"

namespace transformer
{

/* constructor */
T2TFNN::T2TFNN()
{
    inSize  = -1;
    outSize = -1;
    hSize   = -1;
}

/* deconstructor */
T2TFNN::~T2TFNN()
{
}

/* 
initialize the model 
>> argc - number of arguments
>> argv - list of pointers to the arguments
>> myDevID - device id
*/
void T2TFNN::InitModel(int argc, char ** argv, int myDevID)
{
    devID = myDevID;
    
    float minmax = 0;

    LoadParamInt(argc, argv, "d", &inSize, DEFAULT_EMBEDDING_SIZE);
    LoadParamInt(argc, argv, "d", &outSize, DEFAULT_EMBEDDING_SIZE);
    LoadParamInt(argc, argv, "fnnh", &hSize, outSize * 4);
    LoadParamFloat(argc, argv, "fnnminmax", &minmax, 0.1F);
    LoadParamFloat(argc, argv, "dropoutfnn", &dropoutP, 0);

    InitTensor2D(&w1, inSize, hSize, X_FLOAT, devID);
    InitTensor1D(&b1, hSize, X_FLOAT, devID);

    InitTensor2D(&w2, hSize, outSize, X_FLOAT, devID);
    InitTensor1D(&b2, outSize, X_FLOAT, devID);

    float scale = 1.0F;
    _SetDataFanInOut(&w1, scale);
    _SetDataFanInOut(&w2, scale);

    b1.SetZeroAll();
    b2.SetZeroAll();
}

/* 
make the network 
y = max(0, x * w1 + b1) * w2 + b2
>> input - the input tensor
>> return - the output tensor 
*/
XTensor T2TFNN::Make(XTensor &input, bool isTraining)
{
    XTensor t1;

    /* t1 = max(0, x * w1 + b1) */
    //t1 = Rectify(MMul(input, w1) + b1);
    t1 = Rectify(MulAndShift(input, w1, b1));
    
    if(isTraining && dropoutP > 0)
        t1 = Dropout(t1, dropoutP);

    /* result = t1 * w2 + b2 */
    //return MMul(t1, w2) + b2;
    return MulAndShift(t1, w2, b2);
}


}
