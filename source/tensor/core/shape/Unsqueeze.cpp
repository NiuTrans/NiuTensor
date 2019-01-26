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
#include "../../XName.h"
#include "Unsqueeze.h"
#include "MergeBlockLists.h"
#include "Unsqueeze.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
insert a dimension by copying the blocks for x times
(where x is the size of the inerted dimension)

>> a - input tensor
>> b - output tensor
>> dim - where to insert the dimension
>> dSize - size of the newly-inserted dimension
*/
void _Unsqueeze(const XTensor * a, XTensor * b, int dim, int dSize)
{
    CheckNTErrors((a && b), "Empty input tensors!");
    CheckNTErrors((a->order == b->order - 1), "Unmatched tensors!");
    CheckNTErrors((a->unitSize == b->unitSize), "Unmatched tensors!");

    int dimRDI = b->order - dim - 1;
    for (int i = 0; i < b->order; i++) {
        if (i < dimRDI) {
            CheckNTErrors((a->dimSizeRDI[i] == b->dimSizeRDI[i]), "Unmatched tensors!");
        }
        else if (i > dimRDI) {
            CheckNTErrors((a->dimSizeRDI[i - 1] == b->dimSizeRDI[i]), "Unmatched tensors!");
        }
        else {
            CheckNTErrors((dSize == b->dimSizeRDI[i]), "Unmatched tensors!");
        }
    }

    int blockSize = 1;
    int realBlockSize = 1;

    int blockNumA = 1;
    int blockNumB = 1;
    for (int i = 0; i < dimRDI; i++)
        blockSize *= a->dimSizeRDI[i];

    realBlockSize = blockSize * a->unitSize;

    blockNumA = a->unitNum / blockSize;
    blockNumB = b->unitNum / blockSize;

    CheckNTErrors((blockNumA * dSize == blockNumB), "Unmatched tensors!");

    if (a->devID >= 0 || b->devID >= 0) {
#ifdef USE_CUDA
        _CudaUnsqueeze(a, b, dim, dSize);
#else
        ShowNTErrors("Please specify USE_CUDA and recompile the code!");
#endif
    }
    else {
        XList * sourceArrays = new XList(blockNumB);
        int * blockSizes = new int[blockNumB];

        for (int i = 0; i < blockNumA; i++) {
            char * ap = (char*)a->data + i * realBlockSize;
            for (int j = 0; j < dSize; j++) {
                sourceArrays->Add(ap);
                blockSizes[i * dSize + j] = realBlockSize;
            }
        }

        _MergeBlockLists(sourceArrays, blockSizes, 1, b->data, b->mem);

        delete sourceArrays;
        delete[] blockSizes;
    }
}

/*
insert a dimension by copying the blocks for x times
(where x is the size of the inerted dimension) (returna a XTensor structure)
make a new tensor to keep the result and return it

>> a - input tensor
>> dim - where to insert the dimension
>> dSize - size of the newly-inserted dimension
<< return - a tensor by inserting a dimension by copying the blocks for x times
*/
XTensor Unsqueeze(const XTensor &a, int dim, int dSize)
{
    int order = a.order + 1;
    int * dimSize = new int[order];

    for (int i = 0; i < order; i++) {
        if (i < dim)
            dimSize[i] = a.dimSize[i];
        else if (i == dim)
            dimSize[i] = dSize;
        else
            dimSize[i] = a.dimSize[i - 1];
    }

    float dr = (!a.isSparse) ? 1.0F : a.denseRatio;
    XTensor b(order, dimSize, a.dataType, dr, a.devID, a.mem);
    b.SetTMPFlag();

    /* call _Unsqueeze function */
    _Unsqueeze(&a, &b, dim, dSize);

    /* tensor connections */
    XLink::MakeLink(&a, NULL, &b, SHAPE_UNSQUEEZE);
    XLink::AddParamToHeadInt(&b, dim);
    XLink::AddParamToHeadInt(&b, dSize);

    /* destroy variables */
    delete[] dimSize;

    return b;
}

} // namespace nts(NiuTrans.Tensor)
