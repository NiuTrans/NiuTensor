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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-07-31
 */

#ifndef __UNARY_H__
#define __UNARY_H__

#include "../../XTensor.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/* set every entry to its absolute value */
void _Absolute(const XTensor * a, XTensor * b);
/* set every entry to its absolute value (do it on site)
keep the result in the input tensor a and return nothing */
void _AbsoluteMe(XTensor * a);
/* set every entry to its absolute value (do it on site)
keep the result in the input tensor a and return nothing */
void AbsoluteMe(XTensor & a);
/* set every entry to its absolute value (return an XTensor structure)
make a new tensor to keep the result and return it */
XTensor Absolute(const XTensor & a);
/* set every entry to its absolute value */
void Absolute(const XTensor & a, XTensor & b);

/* set every entry to its ceil value */
void _Ceil(const XTensor * a, XTensor * b);
/* set every entry to its ceil value (do it on site)
keep the result in the input tensor a and return nothing */
void _CeilMe(XTensor * a);
/* set every entry to its ceil value (do it on site)
keep the result in the input tensor a and return nothing */
void CeilMe(XTensor & a);
/* set every entry to its ceil value (return an XTensor structure)
make a new tensor to keep the result and return it */
XTensor Ceil(const XTensor & a);
/* set every entry to its ceil value */
void Ceil(const XTensor & a, XTensor & b);

/* set every entry to its exponent value */
void _Exp(const XTensor * a, XTensor * b);
/* set every entry to its exponent value (do it on site)
keep the result in the input tensor a and return nothing */
void _ExpMe(XTensor * a);
/* set every entry to its exponent value (do it on site)
keep the result in the input tensor a and return nothing */
void ExpMe(XTensor & a);
/* set every entry to its exponent value (return an XTensor structure)
make a new tensor to keep the result and return it */
XTensor Exp(const XTensor & a);
/* set every entry to its exponent value */
void Exp(const XTensor & a, XTensor & b);

/* set every entry to its floor value */
void _Floor(const XTensor * a, XTensor * b);
/* set every entry to its floor value (do it on site)
keep the result in the input tensor a and return nothing */
void _FloorMe(XTensor * a);
/* set every entry to its floor value (do it on site)
keep the result in the input tensor a and return nothing */
void FloorMe(XTensor & a);
/* set every entry to its floor value (return an XTensor structure)
make a new tensor to keep the result and return it */
XTensor Floor(const XTensor & a);
/* set every entry to its floor value */
void Floor(const XTensor & a, XTensor & b);

/* if source entry is non-zero, set target entry to be one, otherwise zero */
void _IsNonZero(const XTensor *a, XTensor *b);
/* if source entry is non-zero, set target entry to be one, otherwise zero (do it on site)
keep the result in the input tensor a and return nothing */
void _IsNonZeroMe(XTensor *a);
/* if source entry is non-zero, set target entry to be one, otherwise zero (do it on site)
keep the result in the input tensor a and return nothing */
void IsNonZeroMe(XTensor &a);
/* if source entry is non-zero, set target entry to be one, otherwise zero (return an XTensor structure)
make a new tensor to keep the result and return it */
XTensor IsNonZero(const XTensor &a);
/* if source entry is non-zero, set target entry to be one, otherwise zero */
void IsNonZero(const XTensor &a, XTensor & b);

/* if source entry is zero, set target entry to be one, otherwise zero */
void _IsZero(const XTensor *a, XTensor *b);
/* if source entry is zero, set target entry to be one, otherwise zero (do it on site)
keep the result in the input tensor a and return nothing */
void _IsZeroMe(XTensor *a);
/* if source entry is zero, set target entry to be one, otherwise zero (do it on site)
keep the result in the input tensor a and return nothing */
void IsZeroMe(XTensor &a);
/* if source entry is zero, set target entry to be one, otherwise zero (return an XTensor structure)
make a new tensor to keep the result and return it */
XTensor IsZero(const XTensor &a);
/* if source entry is zero, set target entry to be one, otherwise zero */
void IsZero(const XTensor &a, XTensor & b);

/* set every entry to its logarithm value */
void _Log(const XTensor * a, XTensor * b);
/* set every entry to its logarithm value (do it on site)
keep the result in the input tensor a and return nothing */
void _LogMe(XTensor * a);
/* set every entry to its logarithm value (do it on site)
keep the result in the input tensor a and return nothing */
void LogMe(XTensor & a);
/* set every entry to its logarithm value (return an XTensor structure)
make a new tensor to keep the result and return it */
XTensor Log(const XTensor & a);
/* set every entry to its logarithm value */
void Log(const XTensor & a, XTensor & b);

/* set every entry to its negative value */
void _Negate(const XTensor * a, XTensor * b);
/* set every entry to its negative value (do it on site)
keep the result in the input tensor a and return nothing */
void _NegateMe(XTensor * a);
/* set every entry to its negative value (do it on site)
keep the result in the input tensor a and return nothing */
void NegateMe(XTensor & a);
/* set every entry to its negative value (return an XTensor structure)
make a new tensor to keep the result and return it */
XTensor Negate(const XTensor & a);
/* set every entry to its negative value */
void Negate(const XTensor & a, XTensor & b);

/* set every entry to its round value */
void _Round(const XTensor * a, XTensor * b);
/* set every entry to its round value (do it on site)
keep the result in the input tensor a and return nothing */
void _RoundMe(XTensor * a);
/* set every entry to its round value (do it on site)
keep the result in the input tensor a and return nothing */
void RoundMe(XTensor & a);
/* set every entry to its round value (return an XTensor structure)
make a new tensor to keep the result and return it */
XTensor Round(const XTensor & a);
/* set every entry to its round value */
void Round(const XTensor & a, XTensor & b);

/* set every entry to its sign value */
void _Sign(const XTensor * a, XTensor * b);
/* set every entry to its sign value (do it on site)
keep the result in the input tensor a and return nothing */
void _SignMe(XTensor * a);
/* set every entry to its sign value (do it on site)
keep the result in the input tensor a and return nothing */
void SignMe(XTensor & a);
/* set every entry to its sign value (return an XTensor structure)
make a new tensor to keep the result and return it */
XTensor Sign(const XTensor & a);
/* set every entry to its sign value */
void Sign(const XTensor & a, XTensor & b);

/* set every entry to its sqrt value */
void _Sqrt(const XTensor * a, XTensor * b);
/* set every entry to its sqrt value (do it on site)
keep the result in the input tensor a and return nothing */
void _SqrtMe(XTensor * a);
/* set every entry to its sqrt value (do it on site)
keep the result in the input tensor a and return nothing */
void SqrtMe(XTensor & a);
/* set every entry to its sqrt value (return an XTensor structure)
make a new tensor to keep the result and return it */
XTensor Sqrt(const XTensor & a);
/* set every entry to its sqrt value */
void Sqrt(const XTensor & a, XTensor & b);

/* set every entry to its square value */
void _Square(const XTensor * a, XTensor * b);
/* set every entry to its square value (do it on site)
keep the result in the input tensor a and return nothing */
void _SquareMe(XTensor * a);
/* set every entry to its square value (do it on site)
keep the result in the input tensor a and return nothing */
void SquareMe(XTensor & a);
/* set every entry to its square value (return an XTensor structure)
make a new tensor to keep the result and return it */
XTensor Square(const XTensor & a);
/* set every entry to its square value */
void Square(const XTensor & a, XTensor & b);

/* set every entry to its sine value */
void _Sin(const XTensor * a, XTensor * b);
/* set every entry to its sine value (do it on site)
keep the result in the input tensor a and return nothing */
void _SinMe(XTensor * a);
/* set every entry to its sine value (do it on site)
keep the result in the input tensor a and return nothing */
void SinMe(XTensor & a);
/* set every entry to its sine value (return an XTensor structure)
make a new tensor to keep the result and return it */
XTensor Sin(const XTensor & a);
/* set every entry to its sine value */
void Sin(const XTensor & a, XTensor & b);

/* set every entry to its cosine value */
void _Cos(const XTensor * a, XTensor * b);
/* set every entry to its cosine value (do it on site)
keep the result in the input tensor a and return nothing */
void _CosMe(XTensor * a);
/* set every entry to its cosine value (do it on site)
keep the result in the input tensor a and return nothing */
void CosMe(XTensor & a);
/* set every entry to its cosine value (return an XTensor structure)
make a new tensor to keep the result and return it */
XTensor Cos(const XTensor & a);
/* set every entry to its cosine value */
void Cos(const XTensor & a, XTensor & b);

/* set every entry to its tangent value */
void _Tan(const XTensor * a, XTensor * b);
/* set every entry to its tangent value (do it on site)
keep the result in the input tensor a and return nothing */
void _TanMe(XTensor * a);
/* set every entry to its tangent value (do it on site)
keep the result in the input tensor a and return nothing */
void TanMe(XTensor & a);
/* set every entry to its tangent value (return an XTensor structure)
make a new tensor to keep the result and return it */
XTensor Tan(const XTensor & a);
/* set every entry to its tangent value */
void Tan(const XTensor & a, XTensor & b);

} // namespace nts(NiuTrans.Tensor)

#endif // end __UNARY_H__