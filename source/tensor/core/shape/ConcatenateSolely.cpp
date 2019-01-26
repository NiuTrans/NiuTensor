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
* $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-04-24
*/

#include "../../XTensor.h"
#include "../../XUtility.h"
#include "../../XName.h"
#include "ConcatenateSolely.h"
#include "MergeBlockLists.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
concatenate a list of tensors along a given dimension

>> smalls - a list of tensors for concatenation
>> big - the resulting tensor
>> dim - which dimension we perform the concatenation
*/
void _ConcatenateSolely(const XList * smalls, XTensor * big, int dim)
{
    CheckNTErrors(big->order > dim && dim >= 0, "Illegal dimension to concatenate!");

    int catDimSize = 0;
    int dimRDI = big->order - dim - 1;

    for (int i = 0; i < smalls->count; i++) {
        XTensor * tensor = (XTensor*)smalls->GetItem(i);
        CheckNTErrors((big->order == tensor->order), "Unmatched tensor orders!");
        for (int j = 0; j < big->order; j++) {
            if (j != dimRDI) {
                CheckNTErrors((big->dimSizeRDI[j] == tensor->dimSizeRDI[j]), "Unmatched tensor sizes!");
            }
            else {
                catDimSize += tensor->dimSizeRDI[j];
            }
        }
    }

    CheckNTErrors((catDimSize == big->dimSizeRDI[dimRDI]), "Unmatched tensor sizes!");

    int stride = 1;
    for (int i = 0; i < dimRDI; i++)
        stride *= big->dimSizeRDI[i];

    int blockNum = 1;
    for (int i = dimRDI + 1; i < big->order; i++)
        blockNum *= big->dimSizeRDI[i];

    int offset = 0;

    /* 
    two strategies are used - we can either resort to memcpy2d for the case of
    concatenation of a few items, or use MergeBlockLists to merge a large number
    of data blocks 
    */
    if (smalls->count <= MIN_TENSOR_CAT_NUM) {
        for (int i = 0; i < smalls->count; i++) {
            XTensor * tensor = (XTensor*)smalls->GetItem(i);
            int sPitch = stride * tensor->dimSizeRDI[dimRDI] * tensor->unitSize;
            int tPitch = stride * big->dimSizeRDI[dimRDI] * big->unitSize;
            int mSize = sPitch;
            int n = blockNum;
            XMemCopy2D((char*)big->data + offset, tPitch, big->devID,
                (char*)tensor->data, sPitch, tensor->devID,
                mSize, n);
            offset += sPitch;
        }
    }
    else {
        XList * sourceArrays = new XList(smalls->count);
        int * blockSizes = new int[smalls->count];
        for (int i = 0; i < smalls->count; i++) {
            XTensor * tensor = (XTensor*)smalls->GetItem(i);
            blockSizes[i] = stride * tensor->dimSizeRDI[dimRDI] * tensor->unitSize;
            sourceArrays->Add(tensor->data);
        }

        _MergeBlockLists(sourceArrays, blockSizes, blockNum, big->data, big->mem);

        delete[] blockSizes;
        delete sourceArrays;
    }
}
} // namespace nts(NiuTrans.Tensor)
