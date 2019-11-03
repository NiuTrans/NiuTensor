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
 *
 * As it is, this is a heap.
 *
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2017-12-20
 *
 */

#include "XGlobal.h"
#include "XHeap.h"

/* the nts (NiuTrans.Tensor) namespace */
namespace nts{

/* constructor */
template<HeapType hType, typename T>
XHeap<hType, T>::XHeap()
{
}

/* constructor */
template<HeapType hType, typename T>
XHeap<hType, T>::XHeap(int mySize, XMem * myMem)
{
    Init(mySize, myMem);
}

/* deconstructor */
template<HeapType hType, typename T>
XHeap<hType, T>::~XHeap()
{
    delete[] items;
}

template<HeapType hType, typename T>
void XHeap<hType, T>::Init(int mySize, XMem * myMem)
{
    mem = myMem;
    size = mySize;
    count = 0;

    if (mem == NULL)
        items = new HeapNode<T>[mySize];
    else
        mem->Alloc(mem->devID, mySize * sizeof(T));
}

template<HeapType hType, typename T>
void XHeap<hType, T>::Clear(T initValue)
{
    count = 0;
    for (int i = 0; i < size; i++) {
        items[i].index = 0;
        items[i].value = initValue;
    }
}

/* compare node i and node j */
template<HeapType hType, typename T>
_XINLINE_ bool XHeap<hType, T>::Compare(int i, int j)
{
    if (hType == MIN_HEAP)
        return items[i].value < items[j].value;
    else
        return items[j].value < items[i].value;
}

/* top most item */
template<HeapType hType, typename T>
_XINLINE_ HeapNode<T> XHeap<hType, T>::Top()
{
    HeapNode<T> node = items[0];
    return node;
}

/* last item */
template<HeapType hType, typename T>
_XINLINE_ HeapNode<T> XHeap<hType, T>::End()
{
    HeapNode<T> node = items[count - 1];
    return node;
}

/* push an item into the heap */
template<HeapType hType, typename T>
_XINLINE_ void XHeap<hType, T>::Push(HeapNode<T> node)
{
    if (count < size) {
        items[count] = node;
        Up(count);
        count++;
    }
    else if(count == size){
        HeapNode<T> & item0 = items[0];
        if (hType == MIN_HEAP && item0.value >= node.value)
            return;
        else if (hType == MAX_HEAP && item0.value <= node.value)
            return;
        items[0] = node;
        Down(0);
    }
    else {
        ShowNTErrors("Overflow of the heap!");
    }
    
}

/* replace the top-most item and update the heap */
template<HeapType hType, typename T>
_XINLINE_ void XHeap<hType, T>::ReplaceTop(HeapNode<T> node)
{
    items[0] = node;
    Down(0);
}

/* pop the top most item */
template<HeapType hType, typename T>
_XINLINE_ HeapNode<T> XHeap<hType, T>::Pop()
{
    CheckNTErrors(count > 0, "Empty heap!");
    HeapNode<T> node = items[0];
    items[0] = items[count - 1];
    count--;
    items[count].index = 0;
    items[count].value = 0;
    Down(0);
    return node;
}

/* move item k down the tree */
template<HeapType hType, typename T>
_XINLINE_ void XHeap<hType, T>::Down(int k)
{
    int i = k;
    while (2 * i + 1 < count) {
        int l = 2 * i + 1, r = 2 * i + 2;
        int m = (r >= count || Compare(l, r)) ? l : r;
        if (Compare(i, m))
            break;
        HeapNode<T> tmp = items[i];
        items[i] = items[m];
        items[m] = tmp;
        i = m;
    }
}

/* move item k up the tree */
template<HeapType hType, typename T>
_XINLINE_ void XHeap<hType, T>::Up(int k)
{
    int i = k;
    int parent = (i - 1) / 2;
    while (i > 0 && !Compare(parent, i)) {
        HeapNode<T> tmp = items[i];
        items[i] = items[parent];
        items[parent] = tmp;
        i = parent;
        parent = (i - 1) / 2;
    }
}

/* explicit instantiation */
template class XHeap<MAX_HEAP, float>;
template class XHeap<MAX_HEAP, double>;
template class XHeap<MAX_HEAP, int>;
template class XHeap<MIN_HEAP, float>;
template class XHeap<MIN_HEAP, double>;
template class XHeap<MIN_HEAP, int>;

} /* end of the nts (NiuTrans.Tensor) namespace */
