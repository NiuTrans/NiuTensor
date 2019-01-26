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
 * Implementation of list that keeps data items
 *
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-04-17
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "XList.h"
#include "XGlobal.h"

#include "wchar.h"
#include "locale.h"
#if !defined( WIN32 ) && !defined( _WIN32 )
    #include "sys/time.h"
    #include "time.h"
    #include "iconv.h"
#else
    #include "time.h"
#endif

/* the nts (NiuTrans.Tensor) namespace */
namespace nts{

XList NULLList;

/* constructor */
XList::XList()
{
    mem    = NULL;
    maxNum = 0;
    count  = 0;
    items  = NULL;
    isIntList = false;
}

/* 
constructor 
>> myMaxNum - maximum number of items to keep
>> isIntListOrNot - specify if the list keeps int items
*/
XList::XList(int myMaxNum, bool isIntListOrNot)
{
    mem    = NULL;
    maxNum = myMaxNum;
    count  = 0;
    items  = new void*[myMaxNum];
    isIntList = isIntListOrNot;
}

/* 
constructor 
>> myMaxNum - maximum number of items to keep
>> myMem - the memory pool used for data allocation
>> isIntListOrNot - specify if the list keeps int items
*/
XList::XList(int myMaxNum, XMem * myMem, bool isIntListOrNot)
{
    mem    = myMem;
    maxNum = myMaxNum;
    count  = 0;
    items  = (void**)mem->Alloc(mem->devID, sizeof(void*) * maxNum);
    isIntList = isIntListOrNot;
}

/* de-constructor */
XList::~XList()
{
    if(isIntList){
        for(int i = 0; i < count; i++){
            int * p = (int*)items[i];
            delete[] p;
        }
    }
    if(mem == NULL)
        delete[] items;
}

/* 
allocate the data array for the list
>> myMaxNum - maximum number of items to keep
>> isIntListOrNot - specify if the list keeps int items
*/
void XList::Create(int myMaxNum, XMem * myMem)
{
    mem    = myMem;
    maxNum = myMaxNum;
    count  = 0;
    items  = (void**)mem->Alloc(mem->devID, sizeof(void*) * maxNum);
}

/*
add an item into the list
>> item - pointer to the item
*/
void XList::Add(const void * item)
{
    if( count == maxNum ){
        void ** newItems;
        if( mem == NULL )
            newItems = new void*[maxNum * 2 + 1];
        else
            newItems = (void**)mem->Alloc(mem->devID, sizeof(void*) * (maxNum * 2 + 1));
        memcpy(newItems, items, sizeof(void*) * maxNum);
        if( mem == NULL )
            delete[] items;
        items = newItems;
        maxNum = maxNum * 2 + 1;
    }
    
    MTYPE p = (MTYPE)item;
    items[count++] = (MTYPE*)p;

}

/* 
add a number of items into the list 
>> inputItems - pointer to the array of items
>> inputItemCount - number of input items
*/
void XList::Add(void ** inputItems, int inputItemCount)
{
    if( count + inputItemCount >= maxNum ){
        int newMaxNum = (count + inputItemCount) * 2 + 1;
        void ** newItems;
        if( mem == NULL )
            newItems = new void*[newMaxNum];
        else
            newItems = (void**)mem->Alloc(mem->devID, sizeof(void*) * newMaxNum);
        memcpy(newItems, items, sizeof(void*) * maxNum);
        if( mem == NULL )
            delete[] items;
        items = newItems;
        maxNum = newMaxNum;
    }
    memcpy(items + count, inputItems, sizeof(void*) * inputItemCount);
    count += inputItemCount;
}

/*
append a list to the current list
>> l - the list we use to append
*/
void XList::AddList(XList * l)
{
    Add(l->items, l->count);
}

/*
add an integer-typed item into the list
>> item - pointer to the item
*/
void XList::AddInt(int i)
{
    CheckNTErrors(isIntList, "An int list is required!");

    int * a = new int[1];
    *a = i;
    Add(a);
}

/*
insert an item to the given position of the list
>> pos - the position
>> item - the item for insertion
*/
void XList::Insert(int pos, void * item)
{
    if( count == maxNum ){
        void ** newItems;
        if( mem == NULL )
            newItems = new void*[maxNum * 2 + 1];
        else
            newItems = (void**)mem->Alloc(mem->devID, sizeof(void*) * (maxNum * 2 + 1));
        memcpy(newItems, items, sizeof(void*) * maxNum);
        if( mem == NULL )
            delete[] items;
        items = newItems;
        maxNum = maxNum * 2 + 1;
    }

    for(int i = count - 1; i >= pos; i--)
        items[i + 1] = items[i];
    items[pos] = item;
    count++;
}

/* get the item at position i */
void * XList::GetItem(int i) const
{
    CheckNTErrors(i >= 0 && i < count, "Index of a list item is out of scope!");
    return items[i];
}

/* get the integer-typed item at position i */
int XList::GetItemInt(int i)
{
    CheckNTErrors(isIntList, "An int list is required!");
    CheckNTErrors(i >= 0 && i < count, "Index of a list item is out of scope!");
    return *(int*)(items[i]);
}

/* set the item at position i */
void XList::SetItem(int i, void * item)
{
     if( i >= 0 && i < count )
        items[i] = item;
}

/* get the integer-typed item at position i */
void XList::SetItemInt(int i, int item)
{
    CheckNTErrors(isIntList, "An int list is required!");

    if( i >= 0 && i < count )
        *(int*)(items[i]) = item;
}

/* 
find the position of the first matched item 
>> item - the item for matching
<< the position where we hit the item (if any)
*/

int XList::FindFirst(void * item)
{
    for(int i = 0;i < count; i++){
        if(item == items[i])
            return i;
    }
    return -1;
}

/* clear the data array */
void XList::Clear()
{
    if(isIntList){
        for(int i = 0; i < count; i++){
            delete[] (int*)items[i];
        }
        count = 0;
    }
    else
        count = 0;
}

/* delete the data array as well as the string arrays kept in it */
void XList::ClearStringList()
{
    if(mem == NULL){
        for(int i = 0; i < count; i++){
            delete[] (char*)items[i];
        }
    }
    count = 0;
}

/* 
sort the list 
>> itemSize - size of an item
>> comp - the comparison function used in sorting
*/
void XList::Sort(int itemSize, ListCompare comp)
{
    qsort(items, count, itemSize, comp);
}

/* reverse the list */
void XList::Reverse()
{
    int half = count/2;
    for(int i = 0; i < half; i++){
        void * tmp = items[i];
        items[i] = items[count - i - 1];
        items[count - i - 1] = tmp;
    }
}

/* remove the item at position i */
void XList::Remove(int i)
{
    if(i >= count || i < 0)
        return;

    memcpy(items + i, items + i + 1, sizeof(void*) * (count - i - 1));
    
    count--;
}

/* 
copy the list 
>> myMem - memory pool used for allocating the data in the new list
<< hard copy of the list
*/
XList * XList::Copy(XMem * myMem)
{
    XList * newList = new XList(maxNum, myMem);
    for(int i = 0; i < count; i++){
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
void XList::Shuffle(int nround, int beg, int len)
{
    if(beg < 0){
        beg = 0;
        len = count;
    }

    if(beg + len > count)
        return;

    srand((unsigned int)time(NULL));

    for(int k = 0; k < nround; k++){
        /* Fisher¨CYates shuffle */
        for(int i = 0; i < len; i++){
            float a = (float)rand()/RAND_MAX;
            size_t j = (unsigned int) (a*(i+1));
            void* t = items[beg + j];
            items[beg + j] = items[beg + i];
            items[beg + i] = t;
        }
    }
}

} 
/* end of the nts (NiuTrans.Tensor) namespace */
