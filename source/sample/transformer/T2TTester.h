/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2019, Natural Language Processing Lab, Northestern University.
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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2019-03-27
 * A week with no trips :)
 */

#ifndef __T2TTESTER_H__
#define __T2TTESTER_H__

#include "T2TSearch.h"
#include "T2TBatchLoader.h"

namespace transformer
{

/* This class translates test sentences with a trained model. */
class T2TTester
{
public:
    /* vocabulary size of the source side */
    int vSize;

    /* vocabulary size of the target side */
    int vSizeTgt;
    
    /* for batching */
    T2TBatchLoader batchLoader;

    /* decoder for inference */
    T2TSearch seacher;

public:
    /* constructor */
    T2TTester();

    /* de-constructor */
    ~T2TTester();

    /* initialize the model */
    void Init(int argc, char ** argv);

    /* test the model */
    void Test(const char * fn, const char * ofn, T2TModel * model);

    /* dump the result into the file */
    void Dump(FILE * file, XTensor * output);
};

}

#endif