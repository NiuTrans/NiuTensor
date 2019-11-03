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
* $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-07-18
* I'm surprised that I did not write this file till today.
*/

#ifndef __SETDATA_CUH__
#define __SETDATA_CUH__

#include "../../XTensor.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* generate data items with a fixed value p (in int) */
void _CudaSetDataFixedInt(XTensor * tensor, int p);

/* generate data items with a fixed value p (in float) */
void _CudaSetDataFixedFloat(XTensor * tensor, float p);

/* generate data items with a fixed value p (in double) */
void _CudaSetDataFixedDouble(XTensor * tensor, double p);

/* generate data items with a fixed value p (in float) only 
   if the condition entry is non-zero */
void _CudaSetDataFixedCondFloat(XTensor * tensor, XTensor * condition, float p);

/* generate data items with a fixed value p (in int) only 
   if the condition entry is non-zero */
void _CudaSetDataFixedCondInt(XTensor * tensor, XTensor * condition, int p);

/* set data items along with a given dimension (and keep the remaining items unchanged) */
void _CudaSetDataDim(XTensor * tensor, int beg, int len, int dim, DTYPE p);

/* modify data items along with a given index and dimension (and keep the remaining items unchanged) */
void _CudaSetDataIndexed(XTensor * source, XTensor * modify, int dim, int index);

/* generate data as lower triangular matrics for last two dimensions (cuda version) */
void _CudaSetDataLowTri(XTensor * tensor, DTYPE p, int shift);

/* generate data items with a uniform distribution in [lower, upper] */
void _CudaSetDataRand(const XTensor * tensor, DTYPE lower, DTYPE upper);

/* generate data items with a uniform distribution in [lower, upper] and set
   the item to a pre-defined value if the item >= p, set the item to 0 otherwise */
void _CudaSetDataRandP(const XTensor * tensor, DTYPE lower, DTYPE upper, DTYPE p, DTYPE value);

/* set the data with an array of offsets */
void _CudaSetDataWithOffset(XTensor * tensor, MTYPE * offsets, DTYPE value, MTYPE num);

/* set the data with an array of values */
void _CudaSetDataWithOffsetAndValue(XTensor * tensor, MTYPE * offsets, void * value, MTYPE num);

} // namespace nts(NiuTrans.Tensor)

#endif // __SETDATA_CUH__