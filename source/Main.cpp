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

#include "./train/TTrain.h"
#include "./tensor/test/Test.h"
#include "./sample/fnnlm/FNNLM.h"
#include "./sample/transformer/NMT.h"

#include <iostream>

using namespace nts;
using namespace fnnlm;
using namespace nmt;

int main( int argc, const char ** argv )
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    XConfig config;

    if(argc > 1){
        config.Create(argc - 1, argv + 1);
        verboseLevel = config.GetInt("verbose", 1);
    }

    if (argc > 1 && !strcmp(argv[1], "-test"))
        Test();
    else if (argc > 1 && !strcmp(argv[1], "-testtrain"))
        TestTrain();
    else if(argc > 1 && !strcmp(argv[1], "-fnnlm"))
        FNNLMMain(argc - 1, argv + 1);
    else if(argc > 1 && !strcmp(argv[1], "-t2t"))
        NMTMain(argc - 1, argv + 1);
    else{
        fprintf(stderr, "Thanks for using NiuTensor! This is a library for building\n");
        fprintf(stderr, "neural networks in an easy way. \n\n");
        fprintf(stderr, "   Run this program with \"-test\" for unit test!\n");
        fprintf(stderr, "Or run this program with \"-testtrain\" for test of the trainer!\n");
        fprintf(stderr, "Or run this program with \"-fnnlm\" for sample FNNLM!\n");
        fprintf(stderr, "Or run this program with \"-t2t\" for sample Transformer!\n");
    }

    return 0;
}
