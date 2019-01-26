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
 * The first coding job this year!
 *
 */

#ifndef __XLIST_H__
#define __XLIST_H__

#include "XMem.h"
#include "XGlobal.h"

/* the nts (NiuTrans.Tensor) namespace */
namespace nts{

typedef int (* ListCompare)(const void * item1, const void * item2);

/* the XList class */
class XList
{
public:
    /* data items */
    void ** items;

    /* number of items */
    int count;

    /* maximum number of items can be kept */
    int maxNum;

    /* the memory pool for data array allocation */
    XMem * mem;

    /* indicates whether data items are integers */
    bool isIntList;

public:
    /* constructor */
    XList();

    /* constructor */
    XList(int myMaxNum, bool isIntListOrNot = false);

    /* constructor */
    XList(int myMaxNum, XMem * myMem, bool isIntListOrNot = false);

    /* de-constructor */
    ~XList();

    /* utilities */
    void Create(int myMaxNum, XMem * myMem);
    void Add(const void * item);
    void Add(void ** inputItems, int inputItemCount);
    void AddList(XList * l);
    void AddInt(int i);
    void Insert(int pos, void * item);
    void * GetItem(int i) const;   
    int GetItemInt(int i);
    void SetItem(int i, void * item);
    void SetItemInt(int i, int item);
    
    int FindFirst(void * item);
    void Clear();
    void ClearStringList();
    void Sort(int itemSize, ListCompare comp);
    void Reverse();
    void Remove(int i);
    XList * Copy(XMem * myMem);
    void Shuffle(int nround = 10, int beg = -1, int len = 0);

    /* short */
    _XINLINE_ void * Get(int i) {return GetItem(i);};
    _XINLINE_ int GetInt(int i) {return GetItemInt(i);};
    _XINLINE_ void Set(int i, void * item) {SetItem(i, item);};
    _XINLINE_ void SetInt(int i, int item) {SetItemInt(i, item);};

};

extern XList NULLList;

} 
/* end of the nts (NiuTrans.Tensor) namespace */

#endif
