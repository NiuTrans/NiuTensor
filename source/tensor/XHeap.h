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
 * Wedding anniversary !!! 
 *
*/

#ifndef __XHEAP_H__
#define __XHEAP_H__

#include "XMem.h"

/* the nts (NiuTrans.Tensor) namespace */
namespace nts{

enum HeapType{MIN_HEAP, MAX_HEAP};

/* an item in the heap */
template <typename T>
struct HeapNode
{
    /* node index */
    long long index;

    /* value of the node */
    T value;

    HeapNode()
    {
        index = -1;
        value = 0;
    };

    HeapNode(int i, T v)
    {
        index = (long long)i;
        value = v;
    };

    HeapNode(void * i, T v)
    {
        index = (long long)i;
        value = v;

    }
};

/* a heap that keeps a data array of T */
template<HeapType hType, typename T>
class XHeap
{
public:
    /* memory pool */
    XMem * mem;

    /* number of the items the heap keeps */
    int size;

    /* number of the items that are already in the heap */
    int count;

    /* items */
    HeapNode<T> * items;

public:
    /* constructor */
    XHeap();

    /* constructor */
    XHeap(int mySize, XMem * myMem = NULL);

    /* deconstructor */
    ~XHeap();

    /* initialization */
    void Init(int mySize, XMem * myMem = NULL);

    /* clear the data */
    void Clear(T initValue);

    /* compare node i and node j */
    bool Compare(int i, int j);

    /* top most item */
    HeapNode<T> Top();

    /* last item */
    HeapNode<T> End();

    /* push an item into the heap */
    void Push(HeapNode<T> node);

    /* replace the top-most item and update the heap */
    void ReplaceTop(HeapNode<T> node);

    /* pop the top most item */
    HeapNode<T> Pop();

    /* move item k down the tree */
    void Down(int k);

    /* move item k up the tree */
    void Up(int k);

    /* how many items are kept in the heap */
    inline int Count() { return count; };
};

} /* end of the nts (NiuTrans.Tensor) namespace */

#endif
