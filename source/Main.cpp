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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-10
 */

#include <stdio.h>
#include "./network/XNet.h"
#include "./tensor/XUtility.h"
#include "./tensor/function/FHeader.h"
#include "./tensor/core/CHeader.h"
#include "./tensor/test/Test.h"
#include "./sample/fnnlm/FNNLM.h"
#include "./sample/transformer/Transformer.h"

//#define CRTDBG_MAP_ALLOC
//#include <stdlib.h>
//#include <crtdbg.h>

using namespace nts;
using namespace fnnlm;
using namespace transformer;

int main( int argc, const char ** argv )
{
    if(argc > 1 && !strcmp(argv[1], "-test"))
        Test();
    else if(argc > 1 && !strcmp(argv[1], "-fnnlm"))
        FNNLMMain(argc - 1, argv + 1);
    else if(argc > 1 && !strcmp(argv[1], "-t2t"))
        TransformerMain(argc - 1, argv + 1);
    else{
        fprintf(stderr, "Thanks for using NiuTensor! This is a library for building\n");
        fprintf(stderr, "neural networks in an easy way. \n\n");
        fprintf(stderr, "Run this program with \"-test\" for unit test!\n");
        fprintf(stderr, "Or run this program with \"-fnnlm\" for sample FNNLM!\n");
        fprintf(stderr, "Or run this program with \"-t2t\" for sample Transformer!\n");
    }

    return 0;
}
