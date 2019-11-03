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
* $Created by: ZHANG Yuhao (email: zhangyuhao@stu.neu.edu.cn) 2019-07-23
*/

#include "VectorBuffer.h"
#include "math.h"
namespace nts {
/* data size for each buffer */
int VectorBuffer::size()
{
    return 32 / sizeof(DTYPE);
}

/* constructor */
VectorBuffer::VectorBuffer() 
{

}

/* 
constructor
initial values with val
*/
VectorBuffer::VectorBuffer(DTYPE val)
{
    for (int i = 0; i != size(); i++) {
        values[i] = val;
    }
}

/* load data */
VectorBuffer VectorBuffer::loadu(const DTYPE* ptr, bool isExp , DTYPE power , DTYPE* bias )
{
    int count = 32 / sizeof(DTYPE);
    VectorBuffer vec;
    if (isExp) {
        if (bias == NULL) {
            if (power == (DTYPE)1.0) {
                for (int i = 0; i != count; i++) {
                    vec.values[i] = (DTYPE)exp(*(ptr + i));
                }
            }
            else if (power == (DTYPE)2.0) {
                for (int i = 0; i != count; i++) {
                    vec.values[i] = (DTYPE)exp((*(ptr + i)) * (*(ptr + i)));
                }
            }
            else if (power == (DTYPE)0.5) {
                for (int i = 0; i != count; i++) {
                    vec.values[i] = (DTYPE)exp(sqrt(*(ptr + i)));
                }
            }
            else {
                for (int i = 0; i != count; i++) {
                    vec.values[i] = (DTYPE)exp(pow(*(ptr + i), power));
                }
            }
        }/*is bias == NULL*/
        else {
            if (power == (DTYPE)1.0) {
                for (int i = 0; i != count; i++) {
                    vec.values[i] = (DTYPE)exp(*(ptr + i) - bias[i]);
                }
            }
            else if (power == (DTYPE)2.0) {
                for (int i = 0; i != count; i++) {
                    DTYPE value = *(ptr + i) - bias[i];
                    vec.values[i] = (DTYPE)exp(value * value);
                }
            }
            else if (power == (DTYPE)0.5) {
                for (int i = 0; i != count; i++) {
                    vec.values[i] = (DTYPE)exp(sqrt(*(ptr + i) - bias[i]));
                }
            }
            else {
                for (int i = 0; i != count; i++) {
                    vec.values[i] = (DTYPE)exp(pow(*(ptr + i) - bias[i], power));
                }
            }
        }
    }//isExp
    else {
        if (bias == NULL) {
            if (power == (DTYPE)1.0) {
                memcpy(vec.values, ptr, count * sizeof(DTYPE));
            }
            else if (power == (DTYPE)2.0) {
                for (int i = 0; i != count; i++) {
                    vec.values[i] = (*(ptr + i)) * (*(ptr + i));
                }
            }
            else if (power == (DTYPE)0.5) {
                for (int i = 0; i != count; i++) {
                    vec.values[i] = (DTYPE)sqrt(*(ptr + i));
                }
            }
            else {
                for (int i = 0; i != count; i++) {
                    vec.values[i] = (DTYPE)pow(*(ptr + i), power);
                }
            }
        }// if bias == NULL
        else {
            if (power == (DTYPE)1.0) {
                for (int i = 0; i != count; i++) {
                    vec.values[i] = *(ptr + i) - bias[i];
                }
            }
            else if (power == (DTYPE)2.0) {
                for (int i = 0; i != count; i++) {
                    DTYPE value = *(ptr + i) - bias[i];
                    vec.values[i] = value * value;
                }
            }
            else if (power == (DTYPE)0.5) {
                for (int i = 0; i != count; i++) {
                    vec.values[i] = (DTYPE)sqrt(*(ptr + i) - bias[i]);
                }
            }
            else {
                for (int i = 0; i != count; i++) {
                    vec.values[i] = (DTYPE)pow(*(ptr + i) - bias[i], power);
                }
            }
        }
    }
    return vec;
}

/* overloading [] */
const DTYPE& VectorBuffer::operator[](int idx)const
{
    return values[idx];
}

/* overloading + */
VectorBuffer VectorBuffer::operator+(const VectorBuffer &a)
{
    for (int i = 0; i != a.size(); i++) {
        this->values[i] = a[i] + this->values[i];
    }
    return *this;
}

/* conculte the max of two buffer */
VectorBuffer VectorBuffer::maxData(const VectorBuffer &a) {
    for (int i = 0; i != a.size(); i++) {
        this->values[i] = MAX(a[i], this->values[i]);
    }
    return *this;
}

/* conculte the max of two buffer */
VectorBuffer VectorBuffer::minData(const VectorBuffer &a) {
    for (int i = 0; i != a.size(); i++) {
        this->values[i] = MIN(a[i], this->values[i]);
    }
    return *this;
}

}/* end of the nts (NiuTrans.Tensor) namespace */