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

void BackwardTest();
void TransposeTest();
void SumDimTest();

using namespace nts;
using namespace fnnlm;
using namespace transformer;

int main( int argc, const char ** argv )
{
    //_CrtSetDbgFlag(_CrtSetDbgFlag(_CRTDBG_REPORT_FLAG) | _CRTDBG_LEAK_CHECK_DF);
    //_CrtSetBreakAlloc(2708);

    if(argc > 1 && !strcmp(argv[1], "-test"))
        Test();
    else if(argc > 1 && !strcmp(argv[1], "-fnnlm"))
        FNNLMMain(argc - 1, argv + 1);
    else if(argc > 1 && !strcmp(argv[1], "-t2t"))
        TransformerMain(argc - 1, argv + 1);
    else{
        fprintf(stderr, "Thanks for using NiuTrans.Network! This is a library for building\n");
        fprintf(stderr, "neural networks in an easy way. \n\n");
        fprintf(stderr, "Run this program with \"-test\" for unit test!\n");
        fprintf(stderr, "Or run this program with \"-fnnlm\" for sample FNNLM!\n");
        fprintf(stderr, "Or run this program with \"-t2t\" for sample Transformer!\n");
    }

    //_CrtDumpMemoryLeaks();
    
    return 0;
}

void BackwardTest()
{
    XNet net;

    XTensor a;
    XTensor b;
    XTensor c;
    a.enableGrad = true;
    b.enableGrad = false;
    c.enableGrad = false;
    XTensor mean;
    XTensor origin;
    InitTensor2DV2(&a, 2, 3);
    InitTensor1DV2(&b, 2);

    a.SetZeroAll();
    b.SetZeroAll();
    a.Set2D(1.0F, 0, 0);
    a.Set2D(2.0F, 0, 1);
    a.Set2D(3.0F, 0, 2);
    a.Set2D(4.0F, 1, 0);
    a.Set2D(5.0F, 1, 1);
    a.Set2D(6.0F, 1, 2);

    b.Set1D(2.0F, 0);
    b.Set1D(1.0F, 1);

    DivDim(a, b, c, 0);
    c.Dump(stderr, "c:");
    auto loss = CrossEntropy(c, a);

    //XLink::ShowNetwork(stderr, &c);

    net.Backward(loss);

    a.grad->Dump(stderr);

}

void TransposeTest()
{
#ifdef USE_CUDA
    XMem mem0(0, UNI_FREE, MILLION * 64, 1024, MILLION * 64);
    //XMem mem1(1, UNI_FREE, MILLION * 64, 1024, MILLION * 64);
    XTensor x;
    XTensor y;
    XTensor z;

    int loops = 2000;

    int B = 3 * 2 * 4;
    int K = 8 * 1;
    int N = 50;
    int H = 512 * 4;

    int nnn = GDevs.nGPU;

    InitTensor3DV2(&x, B, N, H, X_FLOAT, 0);
    InitTensor4DV2(&y, K, B, N, H/K, X_FLOAT, 0);
    InitTensor3DV2(&z, B, N, H, X_FLOAT, 0);

    cudaEvent_t ctime0;
    cudaEvent_t ctime1;
    cudaEvent_t ctime2;
    cudaEvent_t ctime3;
    cudaEvent_t ctime4;
    cudaEvent_t ctime5;

    float elapsedSplit = 0.0;
    float elapsedMerge = 0.0;
    float elapsedSum = 0.0;

    cudaEventCreate(&ctime0);
    cudaEventCreate(&ctime1);
    cudaEventCreate(&ctime2);
    cudaEventCreate(&ctime3);
    cudaEventCreate(&ctime4);
    cudaEventCreate(&ctime5);

    cudaEventRecord(ctime0, 0);

    double time0 = GetClock();
    for(int i = 0; i < loops; i++)
        _Split(&x, &y, 2, K);
    double time1 = GetClock();
    
    cudaEventRecord(ctime1, 0);
    cudaEventSynchronize(ctime1);
    cudaEventElapsedTime(&elapsedSplit, ctime0, ctime1);

    cudaEventRecord(ctime2, 0);

    double time2 = GetClock();
    for(int i = 0; i < loops; i++)
        _Merge(&y, &x, 3);
    double time3 = GetClock();

    cudaEventRecord(ctime3, 0);
    cudaEventSynchronize(ctime3);
    cudaEventElapsedTime(&elapsedMerge, ctime2, ctime3);

    cudaEventRecord(ctime4, 0);

    double time4 = GetClock();
    for(int i = 0; i < loops; i++)
        _Sum(&x, &z, &x);
    double time5 = GetClock();

    cudaEventRecord(ctime5, 0);
    cudaEventSynchronize(ctime5);
    cudaEventElapsedTime(&elapsedSum, ctime4, ctime5);

    fprintf(stderr, "split:%f merge:%f sum:%f\n", time1 - time0, time3 - time2, time5 - time4);
    fprintf(stderr, "split:%f merge:%f sum:%f\n", elapsedSplit, elapsedMerge, elapsedSum);
#endif
}

void SumDimTest()
{
    XTensor x;
    XTensor y;
    XTensor z;

    int a = 5;
    int b = 7;
    int c = 3;

    InitTensor3DV2(&x, a, b, c, X_FLOAT, -1);
    InitTensor1DV2(&y, c, X_FLOAT, -1);
    InitTensor3DV2(&z, a, b, c, X_FLOAT, -1);

    x.SetZeroAll();
    y.SetZeroAll();
    z.SetZeroAll();

    DTYPE * data = new DTYPE[x.unitNum];

    for(int i = 0; i < x.unitNum; i++)
        data[i] = (DTYPE)i;
    x.SetData(data, x.unitNum);

    for(int i = 0; i < y.unitNum; i++)
        data[i] = -(DTYPE)i;
    y.SetData(data, y.unitNum);

    _SumDim(&x, &y, &z, 2);

    z.Dump(stderr, "z:");

    delete[] data;
}
