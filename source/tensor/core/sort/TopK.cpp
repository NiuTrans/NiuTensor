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
#include "TopK.h"
#include "TopK.cuh"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
get the top-k items along a given dimension
>> a - input tensor
>> b - output tensor (top-k result)
>> index - index of the top-k items
>> dim - the dimension along which the sorting is performed
>> k - how many items returned after sorting
*/
void _TopK(const XTensor * a, XTensor * b, XTensor * index, int dim, int k)
{
    CheckNTErrors((a->unitSize == b->unitSize), "Unmatched input tensors!");
    CheckNTErrors((a->order == b->order), "Unmatched input tensors!");
    CheckNTErrors((index == NULL || a->order == index->order), "Unmatched input tensors!");
    CheckNTErrors((index->dataType == X_INT), "Wrong data type!");

    int dimRDI = a->order - dim - 1;
    for (int i = 0; i < a->order; i++) {
        if (i == dimRDI) {
            CheckNTErrors((b->dimSizeRDI[i] == k), "A too large K");
            CheckNTErrors((index == NULL || index->dimSizeRDI[i] == k), "Wrong size!");
        }
        else {
            CheckNTErrors((b->dimSizeRDI[i] == a->dimSizeRDI[i]), "Wrong size!");
            CheckNTErrors((index == NULL || index->dimSizeRDI[i] == a->dimSizeRDI[i]), "Wrong size!");
        }
    }

    if (a->devID >= 0 || b->devID >= 0) {
#ifdef USE_CUDA
        _CudaTopK(a, b, index, dim, k);
#else
        ShowNTErrors("Plesae specify USE_CUDA and recompile the code!");
#endif
    }
    else {
        CheckNTErrors((a->dataType == DEFAULT_DTYPE), "TODO!");

        int stride = 1;
        int strideNumA = a->dimSizeRDI[dimRDI];
        int strideNumB = b->dimSizeRDI[dimRDI];
        for (int i = 0; i < dimRDI; i++)
            stride *= a->dimSizeRDI[i];

        int blockNum = 1;
        for (int i = dimRDI + 1; i < a->order; i++)
            blockNum *= a->dimSizeRDI[i];
        int blockSizeA = stride * strideNumA;
        int blockSizeB = stride * strideNumB;

        XHeap<MIN_HEAP, DTYPE> heap(k);

        for (int h = 0; h < blockNum; h++) {
            for (int i = 0; i < stride; i++) {
                DTYPE * dataA = (DTYPE*)a->data + (h * blockSizeA + i);
                DTYPE * dataB = (DTYPE*)b->data + (h * blockSizeB + i);
                int * indexData = (int*)index->data + (h * blockSizeB + i);

                /* initialize the heep */
                heap.Clear(DTYPE_MIN);

                for (int j = 0; j < blockSizeA; j += stride) {
                    if (heap.count < heap.size) {
                        heap.Push(HeapNode<DTYPE>(j / stride, dataA[j]));
                    }
                    else {
                        if (dataA[j] > heap.Top().value)
                            heap.ReplaceTop(HeapNode<DTYPE>(j / stride, dataA[j]));
                    }
                }

                for (int j = strideNumA >= k ? k - 1 : strideNumA - 1; j >= 0; j--) {
                    HeapNode<DTYPE> node = heap.Pop();
                    dataB[j * stride] = node.value;
                    indexData[j * stride] = node.index;
                }
            }
        }
    }
}

/*
get the top-k items along a given dimension
>> a - input tensor
>> b - output tensor (top-k result)
>> index - index of the top-k items
>> dim - the dimension along which the sorting is performed
>> k - how many items returned after sorting
*/
void TopK(XTensor &a, XTensor &b, XTensor &index, int dim, int k)
{
    _TopK(&a, &b, &index, dim, k);

    /* tensor connection */
    XList list(2);
    list.Add(&b);
    list.Add(&index);
    XLink::MakeLink(&a, &list, SORT_TOPK);
    XLink::AddParamToHeadInt(&b, dim);
    XLink::AddParamToHeadInt(&index, k);
    XLink::AddParamToHeadInt(&b, dim);
    XLink::AddParamToHeadInt(&index, k);
}


} // namespace nts(NiuTrans.Tensor)
