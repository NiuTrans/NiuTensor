/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2020, Natural Language Processing Lab, Northeastern University.
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
 * $Creted by: Guan Huhao 2020-02-05
 * $Updated by: Xu Chen (email: hello_master1954@163.com) 2020-05-01
 */

#ifndef FLOAT16_H
#define FLOAT16_H

namespace nts { // namespace nts(NiuTrans.Tensor)

struct float16
{
private:
    /* 
    sign is the sign bit 1 means negative, 0 means positive
    exp is the exponent with 16 offset
    data is the data, similar to ieee-754, the highest is default 1 and ignored 
    */
    unsigned short data : 10;
    unsigned short exp : 5;
    unsigned short sign : 1;

    // mask for calculate the highest 1
    static unsigned int mask[32];
    static unsigned int pow2[32];
    
    //int FindHighOne(const int &num, int &l, int &r);
    int AbsCompare(const float16 & a,const float16 & b);

public:
    float16 SetOverFlow();

    // judge whether overflow
    int IsOverlFlow() const;
    
    /* constructor by (sign, exp, data)
       similar to ieee 32 floating point
       sign: 1bit 
       exp:  5bit 
       data: 10bit */
    float16(const int& s, const int& e, const int& d);

    /* default constructor
       This initializes the 16bit floating point to 0. */
    float16();

    // constructor by a 32-bit float num
    float16(const float& data);

    // constructor by other datatype
    template<class T> float16(const T& data);

    void Dump();

    // convert float16 to float and return
    float Float();
    
    /* assignment function and tempalte function
       Float assignment function is the basic function.
       Template assignment function is force change other datetype to float,
       then call the float assignment function.
       Template assignment function now support int and double. */
    float16 operator = (const float& data);
    float16 operator = (const float16& data);
    template<class T>  float16 operator = (const T& data);

    // overload operator (less than) a < b
    int operator < (const float16& data);
    template<class T>  int operator < (const T& data);

    // overload opertator <= (less or equal than) a <= b
    int operator <= (const float16& data);
    template<class T> int operator <= (const T& data);

    // overload operator (greater than) a > b
    int operator > (const float16& data);
    template<class T> int operator > (const T& data);

    // overload opertator >= (greater or equal than) a >= b
    int operator >= (const float16& data);
    template<class T> int operator >= (const T& data);

    // overload operator + (add) a + b
    float16 operator + (const float16& data);
    template<class T> float16 operator + (const T& data);

    // overload operator += (add) a += b
    float16 operator += (const float16& data);
    template<class T> float16 operator += (const T& data);

    // overload operator -(negetive) -a
    float16 operator - ();

    // overload operator - (substraction) a - b
    float16 operator - (const float16& data);
    template<class T> float16 operator - (const T& data);

    // overload operator -= (substraction) a -= b
    float16 operator -= (const float16& data);
    template<class T> float16 operator -= (const T& data);

    // overload operator * (multiple) a * b
    float16 operator * (const float16& data);
    template<class T> float16 operator * (const T& data);

    // overload operator *= (multiple) a *= b
    float16 operator *= (const float16& data);
    template<class T> float16 operator *= (const T& data);

    // overload operator / (division) a / b
    float16 GetInverse() const;
    float16 operator / (const float16& data);
    template<class T> float16 operator / (const T& data);

    // overload operator /= (division) a /= b
    float16 operator /= (const float16& data);
    template<class T> float16 operator /= (const T& data);

};

} // namespace nts(NiuTrans.Tensor)

#endif /* FLOAT16_H */
