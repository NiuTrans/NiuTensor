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

#ifndef __XDATATYPE_H__
#define __XDATATYPE_H__

#include "XGlobal.h"

/* the nts (NiuTrans.Tensor) namespace */
namespace nts{

/* data type of the tensor, e.g., int, float, and double. */
enum TENSOR_DATA_TYPE {X_INT, X_INT8, X_FLOAT, X_FLOAT16, X_DOUBLE};

/* transposed matrix type */
enum MATRIX_TRANS_TYPE{X_TRANS, X_NOTRANS};

/* default data type */
#ifdef DOUBELPRICSION
#define DEFAULT_DTYPE X_DOUBLE
#else
#define DEFAULT_DTYPE X_FLOAT
#endif

/* get data type name */
extern const char * GetDataTypeName(TENSOR_DATA_TYPE type);
extern TENSOR_DATA_TYPE GetDataType(const char * typeName);

/* data conversion (for lower precision computation) */
unsigned short FloatToFloat16(float f);
float Float16ToFloat(unsigned short h);

} /* end of the nts (NiuTrans.Tensor) namespace */

#endif