/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2019, Natural Language Processing Lab, Northestern University.
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

#include "time.h"
#include "XMem.h"
#include "XList.h"
#include "XGlobal.h"

/* the nts (NiuTrans.Tensor) namespace */
namespace nts {

/* constructor */
template <typename T>
TensorListBase<T>::TensorListBase()
{
    mem = NULL;
    maxNum = 0;
    count = 0;
    items = NULL;
}

/* 
constructor 
>> myMaxNum - maximum number of items to keep
>> isIntListOrNot - specify if the list keeps int items
*/
template <typename T>
TensorListBase<T>::TensorListBase(int myMaxNum)
{
    mem = NULL;
    maxNum = myMaxNum;
    count = 0;
    items = new T[myMaxNum];
}

/* 
constructor 
>> myMaxNum - maximum number of items to keep
>> myMem - the memory pool used for data allocation
>> isIntListOrNot - specify if the list keeps int items
*/
template <typename T>
TensorListBase<T>::TensorListBase(int myMaxNum, XMem* myMem)
{
    mem = myMem;
    maxNum = myMaxNum;
    count = 0;
    items = (T*)mem->Alloc(mem->devID, sizeof(T) * maxNum);
}

/* de-constructor */
template <typename T>
TensorListBase<T>::~TensorListBase()
{
    if(items && mem)
        delete[] items;
}


/*
add an item into the list
>> item - a right value
*/
template <typename T>
void TensorListBase<T>::Add(T&& item)
{
    if (count == maxNum) {
        
        T* newItems;
        if (mem == NULL)
            newItems = new T[maxNum * 2 + 1];
        else
            newItems = (T*)mem->Alloc(mem->devID, sizeof(T) * (maxNum * 2 + 1));
        memcpy(newItems, items, sizeof(T) * maxNum);
        items = newItems;
        maxNum = maxNum * 2 + 1;
    }
    items[count++] = item;
}

/* return number of elements */
template<typename T>
size_t TensorListBase<T>::Size()
{
    return count;
}

/*
add an item into the list
>> item - a const reference to the item
*/
template <typename T>
void TensorListBase<T>::Add(const T& item)
{
    if (count == maxNum) {
        T* newItems;
        if (mem == NULL)
            newItems = new T[maxNum * 2 + 1];
        else
            newItems = (T*)mem->Alloc(mem->devID, sizeof(T) * (maxNum * 2 + 1));
        memcpy(newItems, items, sizeof(T) * maxNum);
        items = newItems;
        maxNum = maxNum * 2 + 1;
    }

    items[count++] = item;
}

/* 
add a number of items into the list 
>> inputItems - pointer to the array of items
>> inputItemCount - number of input items
*/
template <typename T>
void TensorListBase<T>::Add(const T* inputItems, int inputItemCount)
{
    if (count + inputItemCount >= maxNum) {
        int newMaxNum = (count + inputItemCount) * 2 + 1;
        T* newItems;
        if (mem == NULL)
            newItems = new T[newMaxNum];
        else
            newItems = (T*)mem->Alloc(mem->devID, sizeof(T) * newMaxNum);
        memcpy(newItems, items, sizeof(T) * maxNum);
        items = newItems;
        maxNum = newMaxNum;
    }
    memcpy(items + count, inputItems, sizeof(T) * inputItemCount);
    count += inputItemCount;
}

/*
append a list to the current list
>> l - the list we use to append
*/
template <typename T>
void TensorListBase<T>::AddList(TensorListBase* l)
{
    Add(l->items, l->count);
}

/*
insert an item to the given position of the list
>> pos - the position
>> item - the item for insertion
*/
template <typename T>
void TensorListBase<T>::Insert(int pos, const T& item)
{
    if (count == maxNum) {
        T* newItems;
        if (mem == NULL)
            newItems = new T[maxNum * 2 + 1];
        else
            newItems = (T*)mem->Alloc(mem->devID, sizeof(T) * (maxNum * 2 + 1));
        memcpy(newItems, items, sizeof(T) * maxNum);
        items = newItems;
        maxNum = maxNum * 2 + 1;
    }

    for (int i = count - 1; i >= pos; i--)
        items[i + 1] = items[i];
    items[pos] = item;
    count++;
}

template<typename T>
void TensorListBase<T>::Insert(int pos, T&& item)
{
    if (count == maxNum) {
        T* newItems;
        if (mem == NULL)
            newItems = new T[maxNum * 2 + 1];
        else
            newItems = (T*)mem->Alloc(mem->devID, sizeof(T) * (maxNum * 2 + 1));
        memcpy(newItems, items, sizeof(T) * maxNum);
        items = newItems;
        maxNum = maxNum * 2 + 1;
    }

    for (int i = count - 1; i >= pos; i--)
        items[i + 1] = items[i];
    items[pos] = item;
    count++;
}

/* get the item at position i */
template <typename T>
T& TensorListBase<T>::GetItem(int i) const
{
    CheckNTErrors(i >= -count && i < count, "Index of a list item is out of scope!");
    CheckNTErrors(count > 0, "Cannt index the item in an empty list!");
    if (i < 0)
        return items[count + i];
    else
        return items[i];
}

/* set the item at position i */
template <typename T>
inline void TensorListBase<T>::SetItem(int i, const T& item)
{
    if (i >= 0 && i < count)
        items[i] = item;
}

template<typename T>
inline void TensorListBase<T>::SetItem(int i, T&& item)
{
    if (i >= 0 && i < count)
        items[i] = item;
}

/* 
find the position of the first matched item 
>> item - the item for matching
<< the position where we hit the item (if any)
*/

template <typename T>
inline int TensorListBase<T>::FindFirst(const T& item)
{
    for (int i = 0; i < count; i++) {
        if (item == items[i])
            return i;
    }
    return -1;
}

template <>
inline int TensorListBase<Example>::FindFirst(const Example& item)
{
    for (int i = 0; i < count; i++) {
        if (item.id == items[i].id)
            return i;
    }
    return -1;
}

template <>
inline int TensorListBase<Result>::FindFirst(const Result& item)
{
    for (int i = 0; i < count; i++) {
        if (item.id == items[i].id)
            return i;
    }
    return -1;
}

/* clear the data array */
template <typename T>
void TensorListBase<T>::Clear()
{
    count = 0;
}

/*
compare function for two elements
*/
int Compare(const void* a, const void* b) {
    return (*(int*)(a)-*(int*)(b));
}

/* 
sort the list 
>> itemSize - size of an item
>> comp - the comparison function used in sorting
*/
template <typename T>
void TensorListBase<T>::Sort(int itemSize)
{
    qsort((void*)items, count, itemSize, Compare);
}

/* reverse the list */
template <typename T>
inline void TensorListBase<T>::Reverse()
{
    int half = count / 2;
    for (int i = 0; i < half; i++) {
        T tmp(items[i]);
        items[i] = items[count - i - 1];
        items[count - i - 1] = tmp;
    }
}

/* remove the item at position i */
template <typename T>
void TensorListBase<T>::Remove(int i)
{
    if (i >= count || i < 0)
        return;

    memcpy(items + i, items + i + 1, sizeof(T*) * (count - i - 1));

    count--;
}

template<typename T>
void TensorListBase<T>::Reserve(int n)
{
    if (items) {
        /* reserve failed */
        return;
    }

    items = new T[n];
}

/* 
copy the list 
>> myMem - memory pool used for allocating the data in the new list
<< hard copy of the list
*/
template <typename T>
TensorListBase<T>* TensorListBase<T>::Copy(XMem* myMem)
{
    TensorListBase<T>* newList = new TensorListBase<T>(maxNum, myMem);
    for (int i = 0; i < count; i++) {
        newList->Add(GetItem(i));
    }
    return newList;
}

/* 
shuffle the list
>> nround - number of rounds for shuffling
>> beg - where we start
>> len - how many items are used in shuffling
*/
template <typename T>
void TensorListBase<T>::Shuffle(int nround, int beg, int len)
{
    if (beg < 0) {
        beg = 0;
        len = count;
    }

    if (beg + len > count)
        return;

    srand((unsigned int)time(NULL));

    for (int k = 0; k < nround; k++) {
        /* Fisher CYates shuffle */
        for (int i = 0; i < len; i++) {
            float a = (float)rand() / RAND_MAX;
            size_t j = (unsigned int)(a * (i + 1));
            T t = items[beg + j];
            items[beg + j] = items[beg + i];
            items[beg + i] = t;
        }
    }
}

/* specializations and typedef of list */
template struct TensorListBase<int>;
template struct TensorListBase<char>;
template struct TensorListBase<char*>;
template struct TensorListBase<long>;
template struct TensorListBase<float>;
template struct TensorListBase<short>;
template struct TensorListBase<XTensor*>;
template struct TensorListBase<Result>;
template struct TensorListBase<Example>;
template struct TensorListBase<void*>;

} /* end of the nts (NiuTrans.Tensor) namespace */