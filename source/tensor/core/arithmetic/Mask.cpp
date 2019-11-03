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
* $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2019-04-24
* I'll attend several conferences and workshops in the following weeks -
* busy days :(
*/

#include "../../XTensor.h"
#include "../../XName.h"
#include "../../XUtility.h"
#include "../shape/IsSameShaped.h"
#include "Mask.h"
#include "Mask.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)
/*
mask entries of a given tensor:
c(i) = a(i) if mask(i) is non-zero
c(i) = alpha if mask(i) = 0
where i is the index of the element
*/
void _Mask(const XTensor * a, const XTensor * mask, XTensor * c, DTYPE alpha)
{
    CheckNTErrors(a && mask && c, "Empty tensor input!");
    CheckNTErrors(a->unitNum == mask->unitNum && a->unitNum == c->unitNum,
        "Unmatched tensors in addition!");
    CheckNTErrors(mask->dataType == X_INT, "The mask tensor must be in X_INT!")
    //CheckNTErrors(a->dataType == mask->dataType && a->dataType == c->dataType,
    //    "Unmatched tensors in addition!");

    if (a->devID >= 0 || mask->devID >= 0 || c->devID >= 0) {
#ifdef USE_CUDA
        if (a == c) {
            int P2PAccesible = 0;
#ifdef CUDA_UVA
            cudaDeviceCanAccessPeer(&P2PAccesible, a->devID, b->devID);
#endif
            if ((a->devID < 0 && mask->devID >= 0) ||
                (a->devID >= 0 && mask->devID < 0) ||
                (a->devID >= 0 && mask->devID >= 0 && a->devID != mask->devID && !P2PAccesible))
            {
                ShowNTErrors("Cannot run this method on multiple devices simultaneously!");
            }
            else
                _CudaMask(a, mask, c, alpha);
        }
        else
            _CudaMask(a, mask, c, alpha);

#endif
    }
    else {
        if (!a->isSparse && !mask->isSparse) {
            CheckNTErrors(!c->isSparse, "Illegal use of sparse tensor in addition!");

            if (a->dataType == DEFAULT_DTYPE &&
                mask->dataType == X_INT &&
                c->dataType == DEFAULT_DTYPE)
            {
                DTYPE * ap = (DTYPE*)a->data;
                int * maskp = (int*)mask->data;
                DTYPE * cp = (DTYPE*)c->data;

                /* unrolling */
                int num = a->unitNum;
                if (num % 2 == 0) {
                    for (int i = 0; i < num; i += 2) {
                        if (maskp[i] == 0) {
                            cp[i] = alpha;
                        }
                        else {
                            cp[i] = ap[i];
                        }

                        if (maskp[i + 1] == 0) {
                            cp[i + 1] = alpha;
                        }
                        else {
                            cp[i + 1] = ap[i + 1];
                        }
                    }
                }
                else {
                    for (int i = 0; i < num; i++) {
                        if (maskp[i] == 0) {
                            cp[i] = alpha;
                        }
                        else {
                            cp[i] = ap[i];
                        }
                    }
                }
            }
            else {
                // TODO!!
                ShowNTErrors("TODO!");
            }
        }
        else {
            // TODO!!
            ShowNTErrors("TODO!");
        }
    }
}

/*
mask entries of a given tensor (on site):
a(i) = a(i) if mask(i) is non-zero
a(i) = alpha if mask(i) = 0
where i is the index of the element
*/
void _MaskMe(XTensor * a, const XTensor * mask, DTYPE alpha)
{
    _Mask(a, mask, a, alpha);
}

/*
mask entries of a given tensor (on site):
a(i) = a(i) if mask(i) is non-zero
a(i) = alpha if mask(i) = 0
where i is the index of the element
*/
void MaskMe(XTensor& a, const XTensor& mask, DTYPE alpha)
{
    _Mask(&a, &mask, &a, alpha);
}

/*
mask entries of a given tensor (return an XTensor structure):
a(i) = a(i) if mask(i) is non-zero
a(i) = alpha if mask(i) = 0
where i is the index of the element
*/
XTensor Mask(const XTensor &a, const XTensor &mask, DTYPE alpha)
{
    XTensor c(&a);
    c.SetTMPFlag();

    /* call _Mask function */
    _Mask(&a, &mask, &c, alpha);

    /* tensor connections */
    if (a.enableGrad) {
        XLink::MakeLink(&a, &mask, &c, MATH_MASK);
        XLink::AddParamToHead(&c, alpha);
    }

    return c;
}

/*
mask entries of a given tensor (return an XTensor structure):
a(i) = a(i) if mask(i) is non-zero
a(i) = alpha if mask(i) = 0
where i is the index of the element
*/
void Mask(const XTensor &a, const XTensor &mask, XTensor &c, DTYPE alpha)
{
    if (!c.isInit || !IsSameShaped(a, c)) {
        InitTensorV2(&c, &a);
    }

    /* call _Mask function */
    _Mask(&a, &mask, &c, alpha);

    if (a.enableGrad) {
        XLink::MakeLink(&a, &mask, &c, MATH_MASK);
        XLink::AddParamToHead(&c, alpha);
    }
}

}