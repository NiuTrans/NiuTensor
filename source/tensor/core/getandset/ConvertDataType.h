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
 * $Created by: LI Yinqiao (li.yin.qiao.2012@hotmail.com) 2018-7-11
 */

#ifndef __CONVERTDATATYPE_H__
#define __CONVERTDATATYPE_H__

#include "../../XTensor.h"
#include "../../XDataType.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* data conversion (for lower precision computation) */
void ConvertDataType(int devID, 
                     void * s, TENSOR_DATA_TYPE typeS, 
                     void * t, TENSOR_DATA_TYPE typeT, int size);

/* convert data type */
void _ConvertDataType(const XTensor * input, XTensor * output);

/* convert data type (return an XTensor structure) */
XTensor ConvertDataType(const XTensor & input, TENSOR_DATA_TYPE dataType);

/* convert data type */
void ConvertDataType(const XTensor & input, XTensor & output, TENSOR_DATA_TYPE dataType);

} // namespace nts(NiuTrans.Tensor)

#endif // __CONVERTDATATYPE_H__
