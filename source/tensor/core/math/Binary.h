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
 * $Created by: JIANG Yufan (email: jiangyufan2018@outlook.com) 2019-04-05
 */

#ifndef __BINARY_H__
#define __BINARY_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* descale tensor entires
b = a / num */
template<class T>
void _Descale(const XTensor * a, XTensor * b, T num);
/* descale tensor entires (on site)
b = a / num */
template<class T>
void _DescaleMe(XTensor * a, T num);
/* descale tensor entires (on site)
b = a / num */
template<class T>
void DescaleMe(XTensor & a, T num); 
/* descale tensor entires
b = a / num */
template<class T>
void Descale(const XTensor & a, XTensor & b, T num);
/* descale tensor entires (return an XTensor structure)
b = a / num */
template<class T>
XTensor Descale(const XTensor & a, T num);

/* mod tensor entires
b = a % base */
template<class T>
void _Mod(const XTensor * a, XTensor * b, T base);
/* mod base entires (on site)
b = a % num */
template<class T>
void _ModMe(XTensor * a, T base);
/* mod tensor entires (on site)
b = a % base */
template<class T>
void ModMe(XTensor & a, T base);
/* mod tensor entires
b = a % base */
template<class T>
void Mod(const XTensor & a, XTensor & b, T base);
/* mod tensor entires (return an XTensor structure)
b = a % base */
template<class T>
XTensor Mod(const XTensor & a, T base);

/* get the power(x, y)
b = power(a, num) */
template<class T>
void _Power(const XTensor * a, XTensor * b, T scale);
/* get the power(x, y) (on site)
b = power(a, num) */
template<class T>
void _PowerMe(XTensor * a, T scale);
/* get the power(x, y) (on site)
b = power(a, num) */
template<class T>
void PowerMe(XTensor & a, T scale); 
/* get the power(x, y)
b = power(a, num) */
template<class T>
void Power(const XTensor & a, XTensor & b, T scale);
/* get the power(x, y) (return an XTensor structure)
b = power(a, num) */
template<class T>
XTensor Power(const XTensor & a, T scale);

/* scale up tensor entires
b = a * num */
template<class T>
void _Scale(const XTensor * a, XTensor * b, T num);
/* scale up tensor entires (on site)
b = a * num */
template<class T>
void _ScaleMe(XTensor * a, T num);
/* scale up tensor entires (on site)
b = a * num */
template<class T>
void ScaleMe(XTensor & a, T num);
/* scale up tensor entires
b = a * num */
template<class T>
void Scale(const XTensor & a, XTensor & b, T num);
/* scale up tensor entires (return an XTensor structure)
b = a * num */
template<class T>
XTensor Scale(const XTensor & a, T num);

/* shift tensor entires
b = a + num */
template<class T>
void _Shift(const XTensor * a, XTensor * b, T num);
/* shift tensor entires (on site)
b = a + num */
template<class T>
void _ShiftMe(XTensor * a, T num);
/* shift tensor entires (on site)
b = a + num */
template<class T>
void ShiftMe(XTensor & a, T num); 
/* shift tensor entires
b = a + num */
template<class T>
void Shift(const XTensor & a, XTensor & b, T num);
/* shift tensor entires (return an XTensor structure)
b = a + num */
template<class T>
XTensor Shift(const XTensor & a, T num);

} // namespace nts(NiuTrans.Tensor)

#endif // end __BINARY_H__
