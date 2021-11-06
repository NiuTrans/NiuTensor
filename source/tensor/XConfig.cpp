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
* this class keeps a batch of paramters.
*
* $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2021-02-28
*/

#include "XConfig.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* constructor */
XConfig::XConfig()
{
    n = 0;
    args = NULL;
    nReal = 0;
}

/* de-constructor */
XConfig::~XConfig()
{
    for (int i = 0; i < n; i++) {
        delete[] args[i];
    }
    delete[] args;
}

/* clear it */
void XConfig::Clear()
{
    for (int i = 0; i < n; i++) {
        delete[] args[i];
    }
    delete[] args;
    n = 0;
    args = NULL;
    nReal = 0;
}

/* 
create a config 
>> myN - number of the input arguments
>> myArgs - the input arguments
*/
void XConfig::Create(const int myN, const char ** myArgs)
{
    CheckNTErrors(myN > 0, "No input parameters to XConfig!");

    for (int i = 0; i < n; i++) {
        delete[] args[i];
    }
    delete[] args;
    args = NULL;
    n = myN;
    nReal = n * 2;
    
    
    args = new char*[nReal];

    for (int i = 0; i < nReal; i++) {
        args[i] = NULL;
    }

    for (int i = 0; i < n; i++) {
        CheckNTErrors(myArgs[i] != NULL, "Illegal parameter input!");
        args[i] = new char[strlen(myArgs[i]) + 1];
        strcpy(args[i], myArgs[i]);
    }
}

/* 
add an argument 
>> myArg - the argument
>> myValue - the value of the argument
*/
void XConfig::Add(const char * myArg, const char * myValue)
{
    CheckNTErrors(myArg != NULL, "No argument!");

    if (n + 2 > nReal) {
        nReal = MAX(n * 2 + 1, 128);
        char ** newArgs = new char*[nReal];
        memset(newArgs, 0, sizeof(char*) * n);
        memcpy(newArgs, args, sizeof(char*) * n);
        delete[] args;
        args = newArgs;
    }

    args[n] = new char[strlen(myArg) + 2];
    args[n][0] = '-';
    strcpy(args[n] + 1, myArg);
    n++;

    if (myValue != NULL) {
        args[n] = new char[strlen(myValue) + 1];
        strcpy(args[n], myValue);
        n++;
    }
}

/* 
add an argument (in integer) 
>> myArg - the argument
>> myValue - the value of the argument
*/
void XConfig::Add(const char * myArg, int myValue)
{
    char value[MAX_WORD_LENGTH_IN_CONFIG];

    sprintf(value, "%d", myValue);

    Add(myArg, value);
}

/* 
add an argument (in bool) 
>> myArg - the argument
>> myValue - the value of the argument
*/
void XConfig::Add(const char * myArg, bool myValue)
{
    char value[2];

    if (myValue)
        value[0] = '1';
    else
        value[0] = '0';
    value[1] = 0;

    Add(myArg, value);
}

/*
add an argument (in float)
>> myArg - the argument
>> myValue - the value of the argument
*/
void XConfig::Add(const char * myArg, float myValue)
{
    char value[MAX_WORD_LENGTH_IN_CONFIG];

    sprintf(value, "%f", myValue);

    Add(myArg, value);
}

/* 
load the value of an argument (in integer) 
>> name - the name of the argument
>> p - where we place the loaded value
>> defaultP - the default value (used only if no argument is hit in the list)
*/
void XConfig::LoadInt(const char * name, int * p, int defaultP)
{
    LoadParamInt(n, args, name, p, defaultP);
}

/*
load the value of an argument (in boolean)
>> name - the name of the argument
>> p - where we place the loaded value
>> defaultP - the default value (used only if no argument is hit in the list)
*/
void XConfig::LoadBool(const char * name, bool * p, bool defaultP)
{
    LoadParamBool(n, args, name, p, defaultP);
}

/*
load the value of an argument (in float)
>> name - the name of the argument
>> p - where we place the loaded value
>> defaultP - the default value (used only if no argument is hit in the list)
*/void XConfig::LoadFloat(const char * name, float * p, float defaultP)
{
    LoadParamFloat(n, args, name, p, defaultP);
}

/*
load the value of an argument (in char string)
>> name - the name of the argument
>> p - where we place the loaded value
>> defaultP - the default value (used only if no argument is hit in the list)
*/
void XConfig::LoadString(const char * name, char * p, const char* defaultP)
{
    LoadParamString(n, args, name, p, defaultP);
}

/* 
get the value of an argument (in integer) 
>> name - the name of the argument
>> defaultP - the default value (used only if no argument is hit in the list)
*/
int XConfig::GetInt(const char * name, int defaultP)
{
    int r;

    LoadInt(name, &r, defaultP);

    return r;
}

/* 
get the value of an argument (in bool)
>> name - the name of the argument
>> defaultP - the default value (used only if no argument is hit in the list)
*/
bool XConfig::GetBool(const char * name, bool defaultP)
{
    bool r;

    LoadBool(name, &r, defaultP);

    return r;
}

/* 
get the value of an argument (in float) 
>> name - the name of the argument
>> defaultP - the default value (used only if no argument is hit in the list)
*/
float XConfig::GetFloat(const char * name, float defaultP)
{
    float r;

    LoadFloat(name, &r, defaultP);

    return r;
}

/* get item number */
int XConfig::GetItemNum()
{
    return n;
}

/* 
get the item with offset i 
>> i - offset
*/
char * XConfig::GetItem(int i)
{
    if (i < n && i >= 0)
        return args[i];
    else
        return NULL;
}

/* 
initialize with another config model 
>> myConfig - the configure model that we want to copy
*/
void XConfig::CreateFromMe(XConfig & myConfig)
{
    Clear();

    for (int i = 0; i < myConfig.GetItemNum(); i++)
        Add(myConfig.GetItem(i), i);
}

/*
load the value of an argument (in integer)
>> argc - number of arguments
>> argv - arguments
>> name - the argument we search for
>> p - the pointer to the target variable where we want to place the value
>> defaultP - the default value we use if no argument is found
*/
void LoadParamInt(int argc, char** argv, const char* name, int* p, int defaultP)
{
    char vname[128];
    vname[0] = '-';
    strcpy(vname + 1, name);
    bool hit = false;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], vname) && i + 1 < argc) {
            *(int*)p = atoi(argv[i + 1]);
            hit = true;
            break;
        }
    }
    if (!hit)
        *p = defaultP;
}

/*
load the value of an argument (in boolean)
>> argc - number of arguments
>> argv - arguments
>> name - the argument we search for
>> p - the pointer to the target variable where we want to place the value
>> defaultP - the default value we use if no argument is found
*/
void LoadParamBool(int argc, char** argv, const char* name, bool* p, bool defaultP)
{
    char vname[128];
    vname[0] = '-';
    strcpy(vname + 1, name);
    bool hit = false;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], vname)) {
            *(bool*)p = true;
            hit = true;
            break;
        }
    }
    if (!hit)
        *p = defaultP;
}

/*
load the value of an argument (in float)
>> argc - number of arguments
>> argv - arguments
>> name - the argument we search for
>> p - the pointer to the target variable where we want to place the value
>> defaultP - the default value we use if no argument is found
*/
void LoadParamFloat(int argc, char** argv, const char* name, float* p, float defaultP)
{
    char vname[128];
    vname[0] = '-';
    strcpy(vname + 1, name);
    bool hit = false;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], vname) && i + 1 < argc) {
            *p = (float)atof(argv[i + 1]);
            hit = true;
            break;
        }
    }
    if (!hit)
        *p = defaultP;
}

/*
load the value of an argument (in char string)
>> argc - number of arguments
>> argv - arguments
>> name - the argument we search for
>> p - the pointer to the target variable where we want to place the value
>> defaultP - the default value we use if no argument is found
*/
void LoadParamString(int argc, char** argv, const char* name, char* p, const char* defaultP)
{
    char vname[128];
    vname[0] = '-';
    strcpy(vname + 1, name);
    bool hit = false;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], vname) && i + 1 < argc) {
            strcpy(p, argv[i + 1]);
            hit = true;
            break;
        }
    }
    if (!hit)
        strcpy(p, defaultP);
}

/*
show the argument list
>> argc - number of arguments
>> argv - arguments
*/
void ShowParams(int argc, char** argv)
{
    fprintf(stderr, "args:\n");
    for (int i = 0; i < argc; i++) {
        if (argv[i][1] == 0)
            continue;
        if (argv[i][0] == '-' && (argv[i][1] < '1' || argv[i][1] > '9')) {
            if (i + 1 < argc && argv[i + 1][0] != '-')
                fprintf(stderr, " %s=%s\n", argv[i], argv[i + 1]);
            else
                fprintf(stderr, " %s=yes\n", argv[i]);
        }
    }
    fprintf(stderr, "\n");
}

} // namespace nts(NiuTrans.Tensor)