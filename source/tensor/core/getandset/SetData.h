/* 
* NiuTrans.Tensor - an open-source tensor library
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
* $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-05-08
*/

#ifndef __SETDATA_H__
#define __SETDATA_H__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* generate data items with a xavier initialization */
void _SetDataFanInOut(XTensor * tensor, DTYPE gain = 1.0F);

/* generate data items with a fixed value p */
void _SetDataFixed(XTensor * tensor, void * valuePointer);

/* generate data items with a fixed value p (in default type) */
void SetDataFixed(XTensor &tensor, DTYPE p);
    
/* generate data items with a fixed value p (in integer) */
void SetDataFixedInt(XTensor &tensor, int p);

/* generate data items with a fixed value p (in int) */
void _SetDataFixedInt(XTensor * tensor, int p);

/* generate data items with a fixed value p (in float) */
void _SetDataFixedFloat(XTensor * tensor, float p);

/* generate data items with a fixed value p (in double) */
void _SetDataFixedDouble(XTensor * tensor, double p);

/* generate data items with a fixed value p only if the condition entry is non-zero */
void _SetDataFixedCond(XTensor * tensor, XTensor * condition, DTYPE p);

/* generate data items with a fixed value p only if the condition entry is non-zero */
void _SetDataFixedCondInt(XTensor * tensor, XTensor * condition, int p);

/* set data items along with a given dimension (and keep the remaining items unchanged) */
void _SetDataDim(XTensor * tensor, int beg, int len, int dim, DTYPE p);

/* modify data items along with a given index and dimension (and keep the remaining items unchanged) */
void _SetDataIndexed(XTensor * source, XTensor * modify, int dim, int index);

/* generate data as lower triangular matrics for last two dimensions */
void _SetDataLowTri(XTensor * tensor, DTYPE p, int shift);

/* generate data items with a uniform distribution in [0, 1] */
void _SetDataRand(XTensor * tensor, int rNum, int cNum);

/* generate data items with a uniform distribution in [lower, upper] */
void _SetDataRand(XTensor * tensor, DTYPE lower, DTYPE upper);

/* generate data items with a range by start, end and the step */
void _SetDataRange(XTensor * tensor, DTYPE lower, DTYPE upper, DTYPE step);

/* generate data items with a uniform distribution in [lower, upper] and set 
   the item to a pre-defined value if the item >= p, set the item to 0 otherwise */
void _SetDataRandP(XTensor * tensor, DTYPE lower, DTYPE upper, DTYPE p, DTYPE value);

/* generate data items with a normal distribution with specified mean and standard deviation */
void _SetDataRandN(XTensor * tensor, DTYPE mean = 0.0F, DTYPE standardDeviation = 1.0F);

/* set the data with an array of offsets */
void _SetDataWithOffset(XTensor * tensor, MTYPE * offsets, DTYPE value, MTYPE num);

/* set the data with an array of values */
void _SetDataWithOffsetAndValue(XTensor * tensor, MTYPE * offsets, void * values, MTYPE num);

} // namespace nts(NiuTrans.Tensor)

#endif // __SETDATA_H__