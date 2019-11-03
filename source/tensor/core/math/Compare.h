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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-12-10
 */

#ifndef __COMPARE_H__
#define __COMPARE_H__

#include "../../XTensor.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/* check whether every entry is equal to the given value */
void _Equal(const XTensor * a, XTensor * b, DTYPE value);

/* check whether every entry is equal to the given value (do it on site) */
void _EqualMe(XTensor * a, DTYPE value);

/* check whether every entry is equal to the given value (do it on site) */
void EqualMe(XTensor & a, DTYPE value);

/* check whether every entry is equal to the given value (return an XTensor structure) */
XTensor Equal(const XTensor & a, DTYPE value);

/* check whether every entry is equal to the given value */
void Equal(const XTensor & a, XTensor & b, DTYPE value);

/* check whether every entry is not equal to the given value */
void _NotEqual(const XTensor * a, XTensor * b, DTYPE value);

/* check whether every entry is not equal to the given value (do it on site) */
void _NotEqualMe(XTensor * a, DTYPE value);

/* check whether every entry is not equal to the given value (do it on site) */
void NotEqualMe(XTensor & a, DTYPE value);

/* check whether every entry is not equal to the given value (return an XTensor structure) */
XTensor NotEqual(const XTensor & a, DTYPE value);

/* check whether every entry is not equal to the given value */
void NotEqual(const XTensor & a, XTensor & b, DTYPE value);

/* return maximum of two tensor for each items */
void _Max(const XTensor * a, const XTensor * b, XTensor * c);

/* return maximum of two tensor for each items (do it on site) */
void _MaxMe(XTensor * a, const XTensor * b);

/* return maximum of two tensor for each items (do it on site) */
void MaxMe(XTensor & a, const XTensor & b);

/* return maximum of two tensor for each items (return an XTensor structure) */
XTensor Max(const XTensor & a, const XTensor & b);

/* return maximum of two tensor for each items */
void Max(const XTensor & a, const XTensor & b, XTensor & c);

/* return minimum of two tensor for each items */
void _Min(const XTensor * a, const XTensor * b, XTensor * c);

/* return minimum of two tensor for each items (do it on site) */
void _MinMe(XTensor * a, const XTensor * b);

/* return minimum of two tensor for each items (do it on site) */
void MinMe(XTensor & a, const XTensor & b);

/* return minimum of two tensor for each items (return an XTensor structure) */
XTensor Min(const XTensor & a, const XTensor & b);

/* return minimum of two tensor for each items */
void Min(const XTensor & a, const XTensor & b, XTensor & c);

} // namespace nts(NiuTrans.Tensor)

#endif // end __COMPARE_H__