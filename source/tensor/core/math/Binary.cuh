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

#ifndef __BINARY_CUH__
#define __BINARY_CUH__

#include "../../XTensor.h"
#include "Binary.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* descale each entry */
template<class T>
void _CudaDescale(const XTensor * a, XTensor * b, T num);

/* power each entry */
template<class T>
void _CudaPower(const XTensor * a, XTensor * b, T num);

/* mod each entry */
template<class T>
void _CudaMod(const XTensor * a, XTensor * b, T base);

/* scale each entry */
template<class T>
void _CudaScale(const XTensor * a, XTensor * b, T num);

/* shift each entry */
template<class T>
void _CudaShift(const XTensor * a, XTensor * b, T num);

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)

#endif // __BINARY_CUH__
