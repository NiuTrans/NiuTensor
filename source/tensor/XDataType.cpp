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
 *
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-05-30
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include "XDataType.h"

/* the nts (NiuTrans.Tensor) namespace */
namespace nts{

const char * GetDataTypeName(TENSOR_DATA_TYPE type)
{
    if (type == X_INT)
        return "X_INT";
    else if (type == X_INT8)
        return "X_INT8";
    else if (type == X_FLOAT)
        return "X_FLOAT";
    else if (type == X_FLOAT16)
        return "X_FLOAT16";
    else if (type == X_DOUBLE)
        return "X_DOUBLE";
    return "NULL";
}

TENSOR_DATA_TYPE GetDataType(const char * typeName)
{
    if (!strcmp(typeName, "X_INT"))
        return X_INT;
    else if (!strcmp(typeName, "X_INT8"))
        return X_INT8;
    else if (!strcmp(typeName, "X_FLOAT"))
        return X_FLOAT;
    else if (!strcmp(typeName, "X_FLOAT16"))
        return X_FLOAT16;
    else if (!strcmp(typeName, "X_DOUBLE"))
        return X_DOUBLE;
    else {
        ShowNTErrors("Unknown data type!");
    }
}

/*
Below is for calling CPU BLAS for fast matrix operations
I'm not sure how fast it is. But it seems that other
guys are crazy about this. So I decided to have a try.
*/

/* float -> float16 */
_XINLINE_ unsigned short FloatToFloat16(float f)
{
    unsigned int x = *((unsigned int*)&f);
    unsigned short h = ((x>>16)&0x8000)|((((x&0x7f800000)-0x38000000)>>13)&0x7c00)|((x>>13)&0x03ff);
    return h;
}

/* float16 -> float */
_XINLINE_ float Float16ToFloat(unsigned short h)
{
    float f = float(((h&0x8000)<<16) | (((h&0x7c00)+0x1C000)<<13) | ((h&0x03FF)<<13));
    return f;
}

} /* end of the nts (NiuTrans.Tensor) namespace */
