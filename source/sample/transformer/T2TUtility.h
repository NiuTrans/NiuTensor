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

#ifndef __T2TUTILITY_H__
#define __T2TUTILITY_H__

#include <stdio.h>

namespace transformer
{

extern FILE * tmpFILE;

/* load arguments */
void LoadParamString(int argc, char ** argv, const char * name, char * p, const char * defaultP);
void LoadParamInt(int argc, char ** argv, const char * name, int * p, int defaultP);
void LoadParamBool(int argc, char ** argv, const char * name, bool * p, bool defaultP);
void LoadParamFloat(int argc, char ** argv, const char * name, float * p, float defaultP);

/* show arguments */
void ShowParams(int argc, char ** argv);

extern int llnum;
extern FILE * tf;

}

#endif
