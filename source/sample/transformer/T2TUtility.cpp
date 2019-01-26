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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace transformer
{

FILE * tmpFILE;
int llnum = 0;
FILE * tf = NULL;

void LoadParamString(int argc, char ** argv, const char * name, char * p, const char * defaultP)
{
    char vname[128];
    vname[0] = '-';
    strcpy(vname + 1, name);
    bool hit = false;
    for(int i = 0; i < argc; i++){
        if(!strcmp(argv[i], vname) && i + 1 < argc){
            strcpy(p, argv[i + 1]);
            //fprintf(stderr, " %s=%s\n", name, argv[i + 1]);
            hit = true;
        }
    }
    if(!hit)
        strcpy(p, defaultP);
}

void LoadParamInt(int argc, char ** argv, const char * name, int * p, int defaultP)
{
    char vname[128];
    vname[0] = '-';
    strcpy(vname + 1, name);
    bool hit = false;
    for(int i = 0; i < argc; i++){
        if(!strcmp(argv[i], vname) && i + 1 < argc){
            *(int*)p = atoi(argv[i + 1]);
            //fprintf(stderr, " %s=%s\n", name, argv[i + 1]);
            hit = true;
        }
    }
    if(!hit)
        *p = defaultP;
}

void LoadParamBool(int argc, char ** argv, const char * name, bool * p, bool defaultP)
{
    char vname[128];
    vname[0] = '-';
    strcpy(vname + 1, name);
    bool hit = false;
    for(int i = 0; i < argc; i++){
        if(!strcmp(argv[i], vname)){
            *(bool*)p = true;
            //fprintf(stderr, " %s=%s\n", name, "true");
            hit = true;
        }
    }
    if(!hit)
        *p = defaultP;
}

void LoadParamFloat(int argc, char ** argv, const char * name, float * p, float defaultP)
{
    char vname[128];
    vname[0] = '-';
    strcpy(vname + 1, name);
    bool hit = false;
    for(int i = 0; i < argc; i++){
        if(!strcmp(argv[i], vname) && i + 1 < argc){
            *p = (float)atof(argv[i + 1]);
            //fprintf(stderr, " %s=%s\n", name, argv[i + 1]);
            hit = true;
        }
    }
    if(!hit)
        *p = defaultP;
}

void ShowParams(int argc, char ** argv)
{
    fprintf(stderr, "args:\n");
    for(int i = 0; i < argc; i++){
        if(argv[i][1] == 0)
            continue;
        if(argv[i][0] == '-' && (argv[i][1] < '1' || argv[i][1] > '9')){
            if(i + 1 < argc && argv[i + 1][0] != '-')
                fprintf(stderr, " %s=%s\n", argv[i], argv[i + 1]);
            else
                fprintf(stderr, " %s=yes\n", argv[i]);
        }
    }
    fprintf(stderr, "\n");
}

}
