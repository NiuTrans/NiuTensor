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
#include "./XBLAS.h"
#include "./core/sort/TopK.h"
#include "./core/movement/Gather.h"
//#define CRTDBG_MAP_ALLOC
//#include <stdlib.h>  
//#include <crtdbg.h> 

using namespace nts;

void SmallTest();
void TransposeTest();
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



