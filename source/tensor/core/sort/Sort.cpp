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

#include <math.h>
#include "../../XTensor.h"
#include "../movement/CopyValues.h"
#include "../shape/IsSameShaped.h"
#include "../utilities/SetAscendingOrder.h"
#include "../../XUtility.h"
#include "../../XName.h"
#include "Sort.h"
#include "Sort.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
sort the tensor along a given dimension
>> a - input tensor
>> b - output tensor
>> index - index of the items in the resulting tensor
>> dim - the dimension along which the sorting is performed
*/
void _Sort(const XTensor * a, XTensor * b, XTensor * index, int dim)
{
    dim = MODX(dim, a->order);
    
    CheckNTErrors((_IsSameShaped(a, b)), "Input tensors should have the same type!");
    CheckNTErrors((dim >= 0 && dim < a->order), "Incorrect dimension specified!");
    CheckNTErrors((a->order == index->order), "Unmatched input tensors!");
    CheckNTErrors((index->dataType == X_INT), "Wrong data type!");

    /* make the index tensor */
    SetAscendingOrder(*index, dim);

    if (a->devID >= 0) {
#ifdef USE_CUDA
        _CudaSortBig(a, b, index, index, dim);
#else
        ShowNTErrors("Plesae specify USE_CUDA and recompile the code!");
#endif
    }
    else {
        int stride = 1;
        int blockNum = 1;
        int strideNum = a->dimSize[dim];
        for (int i = 0; i < dim; i++)
            blockNum *= a->dimSize[i];

        for (int i = dim + 1; i < a->order; i++)
            stride *= a->dimSize[i];
        int blockSize = stride * strideNum;

        _CopyValues(a, b);
        for (int k = 0; k < blockNum; k++) {
        for (int i = 0; i < stride; i++) {
                void * dataB = (char*)b->data + (k * blockSize + i) * b->unitSize;
                void * indexData = (char*)index->data + (k * blockSize + i) * sizeof(int);

                /* we sort the data array along "dim" */
                if (a->dataType == X_FLOAT)
                    XQSort(dataB, indexData, strideNum, a->unitSize, stride, CompXFloat);
                else {
                    ShowNTErrors("TODO!");
                }
            }
        }
    }
}

/*
sort the tensor along a given dimension (do it on site)
keep the result in the input tensor a and return nothing

>> a - input tensor
>> index - index of the items in the resulting tensor
>> dim - the dimension along which the sorting is performed
*/
void _SortMe(XTensor * a, XTensor * index, int dim)
{
    _Sort(a, a, index, dim);
}

/*
sort the tensor along a given dimension (do it on site)
keep the result in the input tensor a and return nothing

>> a - input tensor
>> index - index of the items in the resulting tensor
>> dim - the dimension along which the sorting is performed
*/
void SortMe(XTensor& a, XTensor& index, int dim)
{
    _Sort(&a, &a, &index, dim);
}



/*
sort the tensor along a given dimension (return an XTensor structure)
make a new tensor to keep the result and return it

>> a - input tensor
>> b - output tensor
>> index - index of the items in the resulting tensor
>> dim - the dimension along which the sorting is performed
*/
void Sort(XTensor & a, XTensor & b, XTensor & index, int dim)
{
    dim = MODX(dim, a.order);
    
    /* call _Negate function */
    _Sort(&a, &b, &index, dim);
    
    /* tensor connections */
    //TensorList list(2);
    //list.Add(&b);
    //list.Add(&index);
    // XLink::MakeLink(&a, &list, SORT_SORT);
    // XLink::AddParamToHeadInt(&b, dim);
    // XLink::AddParamToHeadInt(&index, dim);
}

} // namespace nts(NiuTrans.Tensor)
