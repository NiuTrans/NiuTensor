/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2017, Natural Language Processing Lab, Northestern University. 
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
 *
 * This is the entrance of the low-level tensor library : NiuTrans.Tensor
 *
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2015-12-14
 *
 */

#include <stdio.h>
#include <math.h>
#include <time.h>
#include "XTensor.h"
#include "XDevice.h"
#include "./test/Test.h"
#include "./core/CHeader.h"

//#define CRTDBG_MAP_ALLOC
//#include <stdlib.h>  
//#include <crtdbg.h> 

using namespace nts;

void SmallTest();
void TransposeTest();
void LittleTest();
void T2TTest();
void T2TTest2();
void PowerTest();

int main( int argc, const char ** argv )
{
    //PowerTest();
    //LittleTest();

    //T2TTest();
    //T2TTest2();

    //return 0;
    //_CrtSetBreakAlloc(123);

    /* a tiny test */
    //SmallTest();

    //_CrtDumpMemoryLeaks();
    //return 0;

    if(argc > 1 && !strcmp(argv[1], "-test"))
        Test();
    else{
        fprintf(stderr, "Thanks for using NiuTrans.Tensor! This is a library that eases the\n");
        fprintf(stderr, "use of tensors. All you need is to ... \n\n");
        fprintf(stderr, "Run this program with \"-test\" for unit test!\n");
    }

    //_CrtDumpMemoryLeaks();

    return 0;
}

void myRead(XTensor * tensor, const char * filename, const char * label)
{
    FILE * file = fopen(filename, "rb");
    if(file == NULL)
        printf("%s\n", filename);
    tensor->Read(file, label);
}

void myDump(XTensor * tensor, const char * filename, const char * label)
{
    FILE * file = fopen(filename, "wb");
    if(file == NULL)
        printf("%s\n", filename);
    tensor->Dump(file, label);
}

void PowerTest()
{
    XTensor input;
    XTensor output;
    InitTensor2D(&input, 256, 10000, X_FLOAT, 0);
    InitTensor2D(&output, 256, 10000, X_FLOAT, 0);
    myRead(&input, "1.txt", "");

    _Power(&input, &output, 2);
    output.Dump(stderr, "", 200);
}

void SmallTest()
{
    XTensor a;
    XTensor b;
    XTensor c;
    XTensor d;

    InitTensor2D(&a, 2, 2);
    InitTensor2D(&b, 2, 2);
    a.SetZeroAll();
    b.SetZeroAll();
    a.Set2D(1.0F, 0, 0);
    a.Set2D(2.0F, 1, 1);

    b = Sum(a, Multiply(a, a));

    /* this is prohibited !!!!!!!!!!!!! */
    //XTensor c = a * b + a;
    //XTensor d = a + b + c.Lin(0.5F);
    
    c = a * b + a;
    d = a + b + c.Lin(0.5F);

    XLink::CheckNetwork(&d);
    //XLink::ShowNetwork(stderr, &d);
        
    a.Dump(stderr, "a:");
    b.Dump(stderr, "b:");
    c.Dump(stderr, "c:");
    d.Dump(stderr, "d:");
}

void TransposeTest()
{
    XTensor a;
    XTensor b;

    int I = 2;
    int J = 3;

    InitTensor4D(&a, 2, 3, 4, 5);

    int * dims = new int[a.order];
    memcpy(dims, a.dimSize, sizeof(int) * a.order);
    dims[I] = a.dimSize[J];
    dims[J] = a.dimSize[I];

    InitTensor(&b, 4, dims);

    a.SetZeroAll();
    b.SetZeroAll();

    float * data = new float[a.unitNum];
    for(int i = 0; i < a.unitNum; i++)
        data[i] = (float)i;

    a.SetData(data, a.unitNum, 0);

    _Transpose(&a, &b, I, J);
    b.Dump(stderr, "b:");

    delete[] data;
}

void LittleTest()
{
    int a = 5000;
    int b = 100000;
    int c = a*b;
    printf("%d\n", c);

    exit(1);
}

void T2TTest()
{
    XTensor * input;
    XTensor * weight;
    XTensor * output;
    XTensor * gold;
    XTensor * dedy;
    XTensor * dedx;
    XTensor * dedxTmp;
    XTensor * dedw;
    XTensor * padding;

    DTYPE loss;

    int * dimSize = new int[2];
    dimSize[0] = 256;
    dimSize[1] = 10001;

    int * dimSize2 = new int[3];
    dimSize2[0] = 2;
    dimSize2[1] = 31;
    dimSize2[2] = 256;
   
    int * dimSize3 = new int[3];
    dimSize3[0] = 2;
    dimSize3[1] = 31;
    dimSize3[2] = 10001;

    int * dimSize4 = new int[2];
    dimSize4[0] = 2;
    dimSize4[1] = 31;

    input = NewTensor(3, dimSize2, X_FLOAT, 1.0F, 0);
    weight = NewTensor(2, dimSize, X_FLOAT, 1.0F, 0);
    dedw = NewTensor(2, dimSize, X_FLOAT, 1.0F, 0);
    gold = NewTensor(3, dimSize3, X_FLOAT, 1.0F, 0);
    output = NewTensor(3, dimSize3, X_FLOAT, 1.0F, 0);
    dedy = NewTensor(3, dimSize3, X_FLOAT, 1.0F, 0);
    dedx = NewTensor(3, dimSize3, X_FLOAT, 1.0F, 0);
    dedxTmp = NewTensor(3, dimSize3, X_FLOAT, 1.0F, 0);
    padding = NewTensor(2, dimSize4, X_FLOAT, 1.0F, 0);

    //weight = NewTensor(2, dimSize);
    //dedw = NewTensor(2, dimSize);
    //input = NewTensor(3, dimSize2);
    //gold = NewTensor(3, dimSize3);
    //output = NewTensor(3, dimSize3);
    //dedy = NewTensor(3, dimSize3);
    //dedx = NewTensor(3, dimSize3);
    //dedxTmp = NewTensor(3, dimSize3);
    //padding = NewTensor(2, dimSize4);

    myRead(input, "x.txt", "x");
    myRead(weight, "w.txt", "w");
    myRead(gold, "gold.txt", "gold");
    myRead(padding, "padding.txt", "padding");

    XTensor inter;
    inter = MMul(*input, *weight);

    _Softmax(&inter, output, 2);

    //_LogMe(output);
    loss = _CrossEntropyFast(output, gold, REDUCE_MEAN, NULL, padding);

    printf("loss: %f\n", loss);

    _CrossEntropyBackward(dedy, output, gold, NULL);
    //_CrossEntropyBackward(dedy, output, gold, NULL, padding);

    myDump(dedy, "dedy.txt", "dedy");

    _SoftmaxBackward(NULL, output, input, dedy, dedx, NULL, -1, NOLOSS);
    _Sub(output, gold, dedxTmp);

    myDump(dedx, "dedx.txt", "dedx");
    dedx->Dump(stderr, "dedx", 200);
    dedxTmp->Dump(stderr, "dedxTmp", 200);

    input->Reshape(input->unitNum/input->GetDim(-1), input->GetDim(-1));
    dedx->Reshape(dedx->unitNum/dedx->GetDim(-1), dedx->GetDim(-1));

    _MatrixMulBatched(input, X_TRANS, dedx, X_NOTRANS, dedw);

    myDump(dedw, "dedw.txt", "dedw");
}

void T2TTest2()
{
    int dimSize[3];
    dimSize[0] = 161;
    dimSize[1] = 47;
    dimSize[2] = 10001;
    XTensor * probs = NewTensor(3, dimSize, X_FLOAT, 1.0F, 0);
    //XTensor * probs = NewTensor(3, dimSize, X_FLOAT, 1.0F, -1);

    //myRead(probs, "probs.txt", " ");
    _SetDataFixedFloat(probs, 1.0F);

    probs->Reshape(1, probs->unitNum);

    DTYPE sum = _ReduceSumAll(probs);
    printf("%e\n", sum);

    //XTensor tmp;
    //tmp = IsNonZero(*probs);
    //DTYPE nonZeroNum = ReduceSumAll(tmp);
    //printf("%f\n", nonZeroNum);
    //
    //DTYPE gpu = ReduceSum(*probs, 1).Get2D(0, 0);

    //printf("%e\n", gpu);
}

