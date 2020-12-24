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

#include "time.h"
#include "XList.h"
#include "XGlobal.h"

/* the nts (NiuTrans.Tensor) namespace */
namespace nts {

/* constructor */
template <typename T>
TensorListBase<T>::TensorListBase()
{
    maxNum = 1;
    count = 0;
    items = (T*)malloc(sizeof(T) * 1);
}

/* 
constructor 
>> myMaxNum - maximum number of items to keep
*/
template <typename T>
TensorListBase<T>::TensorListBase(int myMaxNum)
{
    CheckNTErrors(myMaxNum > 0, "check if the input number > 0");
    maxNum = myMaxNum;
    count = 0;
    items = (T*)malloc(sizeof(T) * myMaxNum);
}

/*
constructor
>> myMaxNum - maximum number of items to keep
*/
template <typename T>
TensorListBase<T>::TensorListBase(const T* inputItems, int inputItemCount)
{
    CheckNTErrors(inputItemCount > 0, "check if the input number > 0");
    maxNum = inputItemCount;
    count = inputItemCount;
    items = (T*)malloc(sizeof(T) * inputItemCount);
    memcpy(items, inputItems, inputItemCount * sizeof(T));
}

/* copy-constructor */
template<typename T>
TensorListBase<T>::TensorListBase(const TensorListBase<T>& l)
{
    CheckNTErrors(l.maxNum > 0, "check if the input number > 0");
    maxNum = l.maxNum;
    count = l.count;
    items = (T*)malloc(sizeof(T) * maxNum);
    memcpy(items, l.items, l.count * sizeof(T));
}

/* move-constructor */
template<typename T>
TensorListBase<T>::TensorListBase(TensorListBase<T>&& l)
{
    CheckNTErrors(l.maxNum > 0, "check if the input number > 0");
    maxNum = l.maxNum;
    count = l.count;
    items = l.items;
    l.items = NULL;
}

/* assignment operator for a constant reference */
template<typename T>
TensorListBase<T> TensorListBase<T>::operator=(const TensorListBase<T>& l)
{
    maxNum = l.maxNum;
    count = l.count;
    items = (T*)malloc(sizeof(T) * maxNum);
    memcpy(items, l.items, l.count * sizeof(T));
    return *this;
}

/* assignment operator for a rvalue */
template<typename T>
TensorListBase<T> TensorListBase<T>::operator=(TensorListBase<T>&& l)
{
    maxNum = l.maxNum;
    count = l.count;
    items = (T*)malloc(sizeof(T) * maxNum);
    memcpy(items, l.items, l.count * sizeof(T));
    return *this;
}

/* de-constructor */
template <typename T>
TensorListBase<T>::~TensorListBase()
{
    if(items != NULL)
        free(items);
    items = NULL;
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
        
        newItems = (T*)realloc(items, sizeof(T) * (count * 2 + 1));
        if (newItems != NULL)
            items = newItems;
        else {
            newItems = (T*)malloc(sizeof(T) * (count * 2 + 1));
            memcpy(newItems, items, count * sizeof(T));
            free(items);
            items = newItems;
        }
            

        maxNum = count * 2 + 1;
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

        newItems = (T*)realloc(items, sizeof(T) * (count * 2 + 1));
        if (newItems != NULL)
            items = newItems;
        else {
            newItems = (T*)malloc(sizeof(T) * (count * 2 + 1));
            memcpy(newItems, items, count * sizeof(T));
            free(items);
            items = newItems;
        }

        maxNum = count * 2 + 1;
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
        T* newItems;

        newItems = (T*)realloc(items, sizeof(T) * (count + inputItemCount + 1));
        if (newItems != NULL)
            items = newItems;
        else {
            newItems = (T*)malloc(sizeof(T) * (maxNum + count + inputItemCount + 1));
            memcpy(newItems, items, count * sizeof(T));
            free(items);
            items = newItems;
        }

        maxNum += (count + inputItemCount + 1);
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

        newItems = (T*)realloc(items, sizeof(T) * (count * 2 + 1));
        if (newItems != NULL)
            items = newItems;
        else {
            newItems = (T*)malloc(sizeof(T) * (count * 2 + 1));
            memcpy(newItems, items, count * sizeof(T));
            free(items);
            items = newItems;
        }

        maxNum = count * 2 + 1;
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

        newItems = (T*)realloc(items, sizeof(T) * (count * 2 + 1));
        if (newItems != NULL)
            items = newItems;
        else {
            newItems = (T*)malloc(sizeof(T) * (count * 2 + 1));
            memcpy(newItems, items, count * sizeof(T));
            free(items);
            items = newItems;
        }

        maxNum = count * 2 + 1;
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

/* check if an item exists in this list */
template<typename T>
bool TensorListBase<T>::Contains(const T& item)
{
    return FindFirst(item) >= 0;
}

/* clear the data array */
template <typename T>
void TensorListBase<T>::Clear()
{
    count = 0;
    maxNum = 0;
    if(items != NULL)
        free(items);
    items = NULL;
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
void TensorListBase<T>::Remove(int idx)
{
    CheckNTErrors(idx < count && idx > -1, "invalid index");

    for (int i = idx; i < count - 1; i++) {
        items[i] = items[i + 1];
    }

    count--;
}

template<typename T>
void TensorListBase<T>::Reserve(int n)
{
    if (items) {
        /* reserve failed */
        return;
    }

    items = (T*)malloc(sizeof(T) * n);
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

/* 
read data from a file 
>> fp - pointer to a file
>> num - number of items to be read
*/
template<typename T>
void TensorListBase<T>::ReadFromFile(FILE* fp, int num)
{
    if (maxNum < num) {
        if(!items)
            Reserve(num - maxNum);
        else {
            free(items);
            items = (T*)malloc(sizeof(T) * num);
        }
    }
    fread(items, sizeof(T), num, fp);
    maxNum = num;
    count += num;
}

/* specializations and typedef of list */
template struct TensorListBase<int>;
template struct TensorListBase<char>;
template struct TensorListBase<char*>;
template struct TensorListBase<long>;
template struct TensorListBase<float>;
template struct TensorListBase<short>;
template struct TensorListBase<XTensor*>;
template struct TensorListBase<uint64_t>;
template struct TensorListBase<void*>;
template struct TensorListBase<Example*>;
template struct TensorListBase<TrainExample*>;
template struct TensorListBase<Result*>;

} /* end of the nts (NiuTrans.Tensor) namespace */