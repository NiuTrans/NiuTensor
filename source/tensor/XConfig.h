/*
* NiuTrans.Tensor - an open-source tensor library
* Copyright (C) 2021
* Natural Language Processing Lab, Northeastern University
* and
* NiuTrans Research
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
* this class defines a parameter keeper.
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-02-28
* A new semester begins today.
*/

#ifndef __XCONFIG_H__
#define __XCONFIG_H__

#include "XGlobal.h"
#include "XUtility.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#define MAX_WORD_LENGTH_IN_CONFIG 256

/* the parameter keeper */
class XConfig
{
private:
    /* number of arguments */
    int n;
    
    /* argument list (in char*) */
    char ** args;

    /* number of items we rellocate for these arguments */
    int nReal;

public:
    /* constructor */
    XConfig();

    /* de-constructor */
    ~XConfig();
    
    /* clear it */
    void Clear();

    /* create a config */
    void Create(const int myN, const char ** myArgs);

    /* add an argument */
    void Add(const char * myArg, const char * myValue);

    /* add an argument (in integer) */
    void Add(const char * myArg, int myValue);

    /* add an argument (in bool) */
    void Add(const char * myArg, bool myValue);

    /* add an argument (in float) */
    void Add(const char * myArg, float myValue);

    /* load the value of an argument to a variable (in integer) */
    void LoadInt(const char * name, int * p, int defaultP);

    /* load the value of an argument to a variable (in boolean) */
    void LoadBool(const char * name, bool * p, bool defaultP);

    /* load the value of an argument to a variable (in float) */
    void LoadFloat(const char * name, float * p, float defaultP);

    /* load the value of an argument to a variable (in char string) */
    void LoadString(const char * name, char * p, const char* defaultP);

    /* get the value of an argument (in integer) */
    int GetInt(const char * name, int defaultP);

    /* get the value of an argument (in boolean) */
    bool GetBool(const char * name, bool defaultP);

    /* get the value of an argument (in float) */
    float GetFloat(const char * name, float defaultP);

    /* get item number */
    int GetItemNum();

    /* get the item with offset i */
    char * GetItem(int i);

    /* initialize with another config model */
    void CreateFromMe(XConfig &myConfig);

};

#define MAX_PARAM_NUM 100

/* load arguments */
void extern LoadParamInt(int argc, char** argv, const char* name, int* p, int defaultP);
void extern LoadParamBool(int argc, char** argv, const char* name, bool* p, bool defaultP);
void extern LoadParamFloat(int argc, char** argv, const char* name, float* p, float defaultP);
void extern LoadParamString(int argc, char** argv, const char* name, char* p, const char* defaultP);

/* show arguments */
void extern ShowParams(int argc, char** argv);

} // namespace nts(NiuTrans.Tensor)

#endif