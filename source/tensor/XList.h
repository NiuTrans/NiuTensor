/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2019, Natural Language Processing Lab, Northeastern University.
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
  * Implementation of template list that keeps data items
  *
  * $Created by: HU Chi (huchinlp@foxmail.com)
  *
  */

#include "XMem.h"
#include "XGlobal.h"

#include <cstdint>
#include <string>

#ifndef __TensorList_H__
#define __TensorList_H__


/* the nts (NiuTrans.Tensor) namespace */
namespace nts {
    
/* the TensorListBase class */
template <typename T>
struct TensorListBase {
public:

    /* data items */
    T *items;

    /* number of items */
    int count;

    /* maximum number of items can be kept */
    int maxNum;

public:
    /* constructor */
    TensorListBase();

    /* constructor */
    TensorListBase(int myMaxNum);

    /* constructor */
    TensorListBase(const T* inputItems, int inputItemCount);

    /* copy-constructor */
    TensorListBase(const TensorListBase<T>& l);

    /* move-constructor */
    TensorListBase(TensorListBase<T>&& l);

    /* assignment operator for a constant reference */
    TensorListBase<T> operator=(const TensorListBase<T>& l);

    /* assignment operator for a rvalue */
    TensorListBase<T> operator=(TensorListBase<T>&& l);

    /* de-constructor */
    ~TensorListBase();

    /* reallocate */
    void Reallocate(int itemNum);

    /* add an item into the list */
    void Add(T&& item);

    /* return number of elements */
    int Size();

    /* add an item into the list */
    void Add(const T& item);

    /* add an item (as an integer) into the list */
    void AddInt(const int item);

    /* add an item (as a float) into the list */
    void AddFloat(const float item);

    /* add an item (as a long long) into the list */
    void AddLLong(const long long item);

    /* add a number of items into the list */
    void Add(const T* inputItems, int inputItemCount);

    /* append a list to the current list */
    void AddList(TensorListBase* l);

    /* insert an item to the given position of the list */
    void Insert(int pos, const T& item);

    /* insert an item to the given position of the list */
    void Insert(int pos, T&& item);

    /* get the item at position i */
    T& GetItem(int i) const;

    /* get the item at position i and force it to an integer */
    int GetItemInt(int i) const;

    /* get the item at position i and force it to a float number */
    float GetItemFloat(int i) const;

    /* get the item at position i and force it to an long long number */
    long long GetItemLLong(int i) const;

    /* set the item at position i */
    void SetItem(int i, const T& item);

    /* set the item at position i */
    void SetItem(int i, T&& item);

    /* set the item (as an integer) at position i */
    void SetItemInt(int i, const int item);

    /* set the item (as a float) at position i */
    void SetItemFloat(int i, const float item);

    /* set the item (as a long long) at position i */
    void SetItemLLong(int i, const long long item);

    /* find the position of the first matched item  */
    int FindFirst(const T& item);

    /* check if an item exists in this list */
    bool Contains(const T& item);

    /* clear the data array */
    void Clear();

    /* sort the list */
    void Sort(int itemSize);

    /* reverse the list */
    void Reverse();

    /* remove the item at position i */
    void Remove(int i);

    /* reserve space for data entry */
    void Reserve(int n);

    /* shuffle the list */
    void Shuffle(int nround = 10, int beg = -1, int len = 0);

    /* read data from a file */
    void ReadFromFile(FILE* fp, int num);

    /* short */
    T& operator[] (int i) const { return GetItem(i); };
    T& Get(int i) const { return GetItem(i); };
    int GetInt(int i) const { return GetItemInt(i); };
    float GetFloat(int i) const { return GetItemFloat(i); };
    long long GetLLong(int i) const { return GetItemLLong(i); };
    void Set(int i, T item) { SetItem(i, item); };
    void SetInt(int i, int item) { SetItemInt(i, item); };
    void SetFloat(int i, float item) { SetItemFloat(i, item); };
    void SetLLong(int i, long long item) { SetItemLLong(i, item); };
};

struct XTensor;

typedef TensorListBase<void*> XList;
typedef TensorListBase<int> IntList;
typedef TensorListBase<char> CharList;
typedef TensorListBase<char*> StrList;
typedef TensorListBase<long> LongList;
typedef TensorListBase<float> FloatList;
typedef TensorListBase<short> ShortList;
typedef TensorListBase<uint64_t> UInt64List;
typedef TensorListBase<XTensor*> TensorList;

} /* end of the nts (NiuTrans.Tensor) namespace */

#endif // __TensorList_H__
