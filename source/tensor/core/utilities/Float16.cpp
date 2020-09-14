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

#include "../../XGlobal.h"
#include "Float16.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

float16 float16::SetOverFlow()
{
    exp = 31;
    data = 0;
    return *this;
}

int float16::IsOverlFlow() const 
{
    return exp==31;
}

// mask for calculate the highest 1
unsigned int float16::mask[32] = 
{
    0xffffffff,0xfffffffe,0xfffffffc,0xfffffff8,0xfffffff0,0xffffffe0,0xffffffc0,0xffffff80,
    0xffffff00,0xfffffe00,0xfffffc00,0xfffff800,0xfffff000,0xffffe000,0xffffc000,0xffff8000,
    0xffff0000,0xfffe0000,0xfffc0000,0xfff80000,0xfff00000,0xffe00000,0xffc00000,0xff800000,
    0xff000000,0xfe000000,0xfc000000,0xf8000000,0xf0000000,0xe0000000,0xc0000000,0x80000000
};

// to calculate the power of 2
unsigned int float16::pow2[32] = 
{
    0x00000001,0x00000002,0x00000004,0x00000008,0x00000010,0x00000020,0x00000040,0x00000080,
    0x00000100,0x00000200,0x00000400,0x00000800,0x00001000,0x00002000,0x00004000,0x00008000,
    0x00010000,0x00020000,0x00040000,0x00080000,0x00100000,0x00200000,0x00400000,0x00800000,
    0x01000000,0x02000000,0x04000000,0x08000000,0x10000000,0x20000000,0x40000000,0x80000000,
};

// compare the absolute value， if a < b return 1, else return 0
int float16::AbsCompare(const float16 & a, const float16 & b)
{
    if (a.exp < b.exp)
        return 1;
    else if (a.exp > b.exp) 
        return 0;

    return a.data < b.data;
}

// get inverse that a * inverse(a) == 1
float16 float16::GetInverse() const 
{
    float16 ans;
    ans.sign = sign;
    ans.exp = 29 - exp;
    int rec = pow2[31];
    //let it div 0x80000000
    rec /= (this->data | pow2[10]);

    if (!(rec & pow2[21])) {
        rec <<= 1;
        ans.exp++;
    }
    rec >>= 10;
    ans.data = rec;
    return ans;
}

/* constructor by (sign, exp, data), similar to ieee 32 floating point
>> s - sign: 1bit
>> e - exp:  5bit
>> d - data: 10bit 
*/
float16::float16(const int& s, const int& e, const int& d) 
{
    sign = s;
    exp = e;
    data = d;
}

/* initializes the 16bit floating point to 0 
*/
float16::float16() 
{
    sign = 0;
    exp = 0;
    data = 0;
}

/* constructor by other datatype
   We convert the data to float and convert float to float16.
>> data - num
*/
template<class T>
float16::float16(const T& data) 
{
    *this = (float)data;
}
template float16::float16 (const int &);
template float16::float16 (const double &);

/* constructor by a 32-bit float num
>> data - 32-bit float num
*/
float16::float16(const float& data) 
{
    *this = data;
}

void float16::Dump()
{
    printf("sign: %d\texp: %d\tdata: %d\n", sign, exp, data);
}

/*
convert float16 to float and return
construct of 32-bit is
the 31th bit present the sign
the 30th~23th bit present the exp, with 128 offset
rest 23th～0th store the data
*/
float float16::Float() 
{
    int ret = 0;
    ret = IsOverlFlow() ? 0x7f800000 :
        (sign ? 0x80000000 : 0) | ((exp + 112) << 23) | (data << 13);
    float p = *(float*)&ret;
    return p;
}

// basic assignment function
float16 float16::operator = (const float16& a) 
{
    sign = a.sign;
    exp = a.exp;
    data = a.data;
    return *this;
}

// convert float to float16
float16 float16::operator = (const float& a) 
{
    unsigned int p = *(unsigned int*)&a;
    sign = p & pow2[31] ? 1 : 0;

    if (a > 65535 || a < -65535) 
        return SetOverFlow();
    exp = ((p >> 23)& (0xf)) | ((p >> 26 & 0x10));
    data = (p >> 13);
    return *this;
}

/* Template assignment function is force change other datetype to float,
   then call the float assignment function.
   Template assignment function now support int and double.
*/
template <class T>
float16 float16::operator = (const T& data) 
{
    *this = (float)data;
    return *this;
}
template float16 float16:: operator = <int>(const int&);
template float16 float16:: operator = <double>(const double&);

/*
template for multi-datatype overload
>> operator - the overload operator, e.g. <, =
>> return_type - the returned datetype of function, e.g, int, float
>> expression - the returned expression
*/
#define _OVERLOAD_OPRATER_TEMPLATE(operation, returnType, expression)       \
template<class T>                                                           \
returnType float16::operator operation (const T & data)                     \
{                                                                           \
    float16 rec=(float)data;                                                \
    return expression;                                                      \
}                                                                           \
template returnType float16::operator operation <int>(const int&);          \
template returnType float16::operator operation <float>(const float&);      \
template returnType float16::operator operation <double>(const double&);

// overload operator (less than) a<b
int float16::operator < (const float16& data) 
{
    if (sign < data.sign)
        return 1;
    else if (sign > data.sign) 
        return 0;

    if (exp < data.exp) 
        return 1;
    else if (exp > data.exp) 
        return 0;
    
    return this->data < data.data;
}
_OVERLOAD_OPRATER_TEMPLATE(< , int, *this < rec)

// overload opertator <= (less or equal than) a <= b
int float16::operator <= (const float16& data) 
{
    if (sign < data.sign)
        return 1;
    else if (sign > data.sign) 
        return 0;

    if (exp < data.exp) 
        return 1;
    else if (exp > data.exp) 
        return 0;

    return this->data <= data.data;
}
_OVERLOAD_OPRATER_TEMPLATE(<= , int, *this <= rec)

// overload operator (greater than) a > b
int float16::operator > (const float16& data) 
{
    if (sign > data.sign)
        return 1;
    else if (sign < data.sign) 
        return 0;

    if (exp > data.exp) 
        return 1;
    else if (exp < data.exp) 
        return 0;

    return this->data > data.data;
}
_OVERLOAD_OPRATER_TEMPLATE(> , int, * this > rec)

// overload opertator >= (greater or equal than) a >= b
int float16::operator >= (const float16& data) 
{
    if (sign > data.sign)
        return 1;
    else if (sign < data.sign) 
        return 0;
    
    if (exp > data.exp) 
        return 1;
    else if (exp < data.exp) 
        return 0;

    return this->data >= data.data;
}
_OVERLOAD_OPRATER_TEMPLATE(>= , int, *this < rec)

// overload operator + (add) a + b
float16 float16::operator + (const float16& data)
{
    float16 ans;

    // avoid overflow inf + anything = inf
    if (this->IsOverlFlow()) 
        return *this;
    if (data.IsOverlFlow()) 
        return data;

    /* the greater number determine the sign and 
       the smaller should be >> to aligment to the greater one */
    if (AbsCompare(*this, data)) {
        ans.sign = data.sign;
        // rec the exp
        int recp = data.exp;          
        //to calculate the data
        int recd = (data.data | (pow2[10])) + 
            ((data.sign ^ sign) ? -1 : 1) * 
            (((pow2[10]) | this->data) >> (data.exp - exp));   

        //because the date may carry， if carryed >> the data, and change its exp
        if (recd) {        
            //to make the highest one is 10th bit
            while (mask[10] & recd) {      
                recd >>= 1;
                recp++;
            }
            //to make the highest one is 10th bit
            while (!(mask[10] & recd)) {    
                recd <<= 1;
                recp--;
            }
        }
        // if data==0, exp should be 0
        else 
            recp = 0;  

        ans.data = recd;
        // if overflow should set overflow
        if (recp >= 31) 
            ans.SetOverFlow(); 
        else {
            ans.exp = recp;
            ans.data = recd;
        }
    }
    // same as above. while divided into two part? reduce assignment to increase efficent
    else {             
        ans.sign = sign;
        int recp = exp;
        int recd = (this->data | (pow2[10])) + 
                   ((sign ^ data.sign) ? -1 : 1) * 
                   (((pow2[10]) | data.data) >> (exp - data.exp));
        if (recd) {
            while (mask[10] & recd) {
                recd >>= 1;
                recp++;
            }
            while (!(mask[10] & recd)) {
                recd <<= 1;
                recp--;
            }
        }
        else 
            recp = 0;

        if (recp >= 31) 
            ans.SetOverFlow();
        else {
            ans.exp = recp;
            ans.data = recd;
        }
    }
    return ans;
}
_OVERLOAD_OPRATER_TEMPLATE(+, float16, *this = *this + rec)

//overide operator +=
float16 float16::operator+=(const float16& data) {
    return *this = *this + data;
}
_OVERLOAD_OPRATER_TEMPLATE(+=, float16, *this = *this + rec)

//overide operator -（negetive） -a
float16 float16::operator - () 
{
    sign ^= 1;
    float16 rec = *this;
    sign ^= 1;
    return rec;
}

//overide operator - (substraction) a-b
float16 float16::operator - (const float16& data) 
{
    float16 ans;
    if (this->IsOverlFlow()) 
        return *this;
    if (data.IsOverlFlow()) 
        return data;

    /* same as add only diffrent is the sign judge, 
    a possitive number sub a greater number will be negtive. */
    if (AbsCompare(*this, data)) {
        ans.sign = !data.sign;
        int recp = data.exp;
        int recd = (data.data | (pow2[10])) + 
            ((data.sign ^ sign) ? 1 : -1) * 
            (((pow2[10]) | this->data) >> (data.exp - exp));
        if (recd) {
            while (mask[10] & recd) {
                recd >>= 1;
                recp++;
            }
            while (!(mask[10] & recd)) {
                recd <<= 1;
                recp--;
            }
        }
        else recp = 0;
        if (recp >= 31) 
            ans.SetOverFlow();
        else {
            ans.data = recd;
            ans.exp = recp;
        }
    }
    else {
        ans.sign = sign;
        int recp = exp;
        int recd = (this->data | (pow2[10])) + 
            ((sign ^ data.sign) ? 1 : -1) * 
            (((pow2[10]) | data.data) >> (exp - data.exp));
        if (recd) {
            while (mask[10] & recd) {
                recd >>= 1;
                recp++;
            }
            while (!(mask[10] & recd)) {
                recd <<= 1;
                recp--;
            }
        }
        else recp = 0;
        if (recp >= 31) 
            ans.SetOverFlow();
        else {
            ans.data = recd;
            ans.exp = recp;
        }
    }
    return ans;
}
_OVERLOAD_OPRATER_TEMPLATE(-, float16, *this = *this - rec)

// overide operator -=
float16 float16::operator-=(const float16& data) 
{
    return *this = *this - data;
}
_OVERLOAD_OPRATER_TEMPLATE(-=, float16, *this = *this - rec)

// overload operator * (multiple) a * b
float16 float16::operator * (const float16& data) 
{
    //if(IsOverlFlow()) 
    //    return *this;
    //if(data.IsOverlFlow()) 
    //    return data;

    float16 ans;
    // ^ to get zhe result sign different will be 1(negtive), same will be 0 positive;
    ans.sign = sign ^ data.sign;

    // mul to get answer
    int rec = (data.data | pow2[10]) * (this->data | pow2[10]); 
    
    // calculat the new exp
    int recp = exp + data.exp - 15 > 0 ? exp + data.exp - 15 : 0;       
    
    // if carryed, to fix the exp and data
    rec >>= 10;                                           
    while (rec & mask[11]) {
        ++recp;
        rec >>= 1;
    }

    if (recp >= 31) 
        ans.SetOverFlow();
    else {
        ans.exp = recp;
        ans.data = rec;
    }
    return ans;
}
_OVERLOAD_OPRATER_TEMPLATE(*, float16, (*this)* rec)

// overload operator *= (multiple) a *= b
float16 float16::operator *= (const float16& data) 
{
    return *this = *this * data;
}
_OVERLOAD_OPRATER_TEMPLATE(*=, float16, *this = *this * rec)

// overload operator / (division) a / b
float16 float16::operator / (const float16& data) 
{
    float16 ans;
    // ^ to get zhe result sign different will be 1(negtive),same will be 0 positive;
    ans.sign = sign ^ data.sign;                       
    // calculat the new exp
    int recp = exp - data.exp + 14;                        
    // defore div should move to the left to avoid precision loss
    int recd = (this->data << 21) | pow2[31];              
    recd /= (data.data | pow2[10]);
    // to make the highest one is the 21st bit
    if (recd & pow2[21]) {                              
        recd >>= 1;
        ++recp;
    }
    if (recp >= 31) 
        ans.SetOverFlow();
    else {
        recd >>= 10;
        ans.data = recd;
        ans.exp = recp;
    }
    return ans;
}
_OVERLOAD_OPRATER_TEMPLATE(/ , float16, (*this) / rec)

// overload operator /= (division) a /= b
float16 float16::operator /= (const float16& data) {
    return *this = *this / data;
}
_OVERLOAD_OPRATER_TEMPLATE(/=, float16, *this = *this / rec)

} // namespace nts(NiuTrans.Tensor)
