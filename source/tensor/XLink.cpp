/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2018, Natural Language Processing Lab, Northestern University.
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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-04
 */

#include <stdio.h>
#include "XLink.h"
#include "XName.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

int XLink::paramSize = PARAM_UNTI_SIZE;

/* constuctor */
XLink::XLink()
{
    head   = NULL;
    tails  = NULL;
    params = NULL;
    tailNum  = 0;
    paramNum = 0;
    type[0] = 0;
    typeID = 0;
    caculator = NULL;
}
    
/* deconstructor */
XLink::~XLink()
{
    if(tails != NULL)
        delete[] tails;
    if(params != NULL)
        delete[] (char*)params;
}

/* reset it */
void XLink::Reset()
{
    delete[] tails;
    delete[] (char*)params;
    head   = NULL;
    tails  = NULL;
    params = NULL;
    tailNum  = 0;
    paramNum = 0;
    type[0]  = 0;
    typeID   = 0;
    caculator = NULL;
}

/* clear it */
void XLink::Clear()
{
    head   = NULL;
    tailNum  = 0;
    paramNum = 0;
    type[0]  = 0;
    typeID   = 0;
    caculator = NULL;
}

/* reset tails */
void XLink::ClearTail()
{
    tailNum = 0;
}

/*
clear the outgoing node list of tensor node
>> node - the node to be cleared
*/
void XLink::ClearOutgoing(XTensor * node)
{
    if(node == NULL)
        return;
    
    XLink &outgo = node->outgo;
    
    for(int i = 0; i < outgo.tailNum; i++){
        
        /* for each parent node */
        XTensor * parent = outgo.tails[i];
        XLink &parentIncome = parent->income;
        
        CheckNTErrors(parentIncome.tailNum > 0, "The node must have incoming edges!");
        
        /* we check for each parent node and remove the link to current node */
        for(int j = 0; j < parentIncome.tailNum; j++){
            if(parentIncome.tails[j] == node){
                memcpy(parentIncome.tails + j, parentIncome.tails + j + 1,
                       sizeof(XTensor*) * (parentIncome.tailNum - 1 - j));
                parentIncome.tailNum--;
                break;
            }
        }
    }
    
    outgo.ClearTail();
    outgo.typeID = 0;
    outgo.type[0] = 0;
    delete[] (char*)outgo.params;
    outgo.params = NULL;
}

/*
clear the incoming node list of tensor node
>> node - the node to be cleared
*/
void XLink::ClearIncoming(XTensor * node)
{
    if(node == NULL)
        return;

    XLink &income = node->income;

    for(int i = 0; i < income.tailNum; i++){

        /* for each incoming node */
        XTensor * child = income.tails[i];
        XLink &childOutgo = child->outgo;

        CheckNTErrors(childOutgo.tailNum > 0, "The node must have outgoing edges!");

        /* we check for each child node and remove the link to current node */
        for(int j = 0; j < childOutgo.tailNum; j++){
            if(childOutgo.tails[j] == node){
                memcpy(childOutgo.tails + j, childOutgo.tails + j + 1,
                       sizeof(XTensor*) * (childOutgo.tailNum - 1 - j));
                childOutgo.tailNum--;
                break;
            }
        }

        if(child->isTmp && childOutgo.tailNum == 0)
            delete child;
    }

    income.ClearTail();
    income.typeID = 0;
    income.type[0] = 0;
    delete[] (char*)income.params;
    income.params = NULL;
}

/* 
set edge type name 
>> id - id of the type
*/
void XLink::SetType(int id)
{
    type[0] = 0;
    strcpy(type, GetOPName(id));
    typeID = id;
    if(id != 0){
        CheckNTErrors(strcmp(type, "NULL"), "illegal edge type name!");
    }
}

/* 
set head 
>> h - pointer to the head tensor
*/
void XLink::SetHead(XTensor * h)
{
    head = h;
}

/* 
add a tail
>> t - pointer to the tail tensor
*/
void XLink::AddTail(XTensor * t)
{
    XTensor ** ts = tails;
    tails = new XTensor*[tailNum + 1];
    memcpy(tails, ts, sizeof(XTensor*) * tailNum);
    tails[tailNum++] = t;
    delete[] ts;
}

/* 
add two tails in one time 
>> t1 - pointer to the tail tensor
>> t2 - pointer to another tail tensor
*/
void XLink::AddTwoTails(XTensor * t1, XTensor * t2)
{
    XTensor ** ts = tails;
    tails = new XTensor*[tailNum + 2];
    memcpy(tails, ts, sizeof(XTensor*) * tailNum);
    tails[tailNum++] = t1;
    tails[tailNum++] = t2;
    delete[] ts;
}

/* 
add a parameter 
>> param - parameter in default type
*/
void XLink::AddParam(DTYPE param)
{
    void * ps = params;
    params = new char[(paramNum + 1) * paramSize];
    memcpy(params, ps, paramNum * paramSize);
    DTYPE * p = (DTYPE*)((char*)params + paramNum * paramSize);
    *p = param;
    paramNum++;
    delete[] (char*)ps;
}

/* 
add a parameter 
>> param - pointer to the parameter
>> size - size of the parameter
*/
void XLink::AddParam(void * param, int size)
{
    void * ps = params;
    params = new char[(paramNum + 1) * paramSize];
    memcpy(params, ps, paramNum * paramSize);
    char * p = (char*)params + paramNum * paramSize;
    memcpy(p, param, size);
    paramNum++;
    delete[] (char*)ps;
}

/* 
get a paramter in default type 
>> i - id of the parameter
<< return - the parameter in default type
*/
DTYPE XLink::GetParam(int i)
{
    CheckNTErrors(params != NULL, "parameter array cannot be empty!");
    char * p = (char*)params + i * paramSize;
    return *(DTYPE*)p;
}

/* 
get a paramter in integer 
>> i - id of the parameter
<< return - the parameter in integer
*/
int XLink::GetParamInt(int i)
{
    CheckNTErrors(params != NULL, "parameter array cannot be empty!");
    char * p = (char*)params + i * paramSize;
    return *(int*)p;
}

/* 
get a paramter in integer 
>> i - id of the parameter
<< return - the parameter in integer
*/
void * XLink::GetParamPointer(int i)
{
    CheckNTErrors(params != NULL, "parameter array cannot be empty!");
    char * p = (char*)params + i * paramSize;
    return *(int **)p;
}
    
/*
get a parameter in MATRIX_TRANS_TYPE
>> i - id of the parameter
<< return - the parameter in MATRIX_TRANS_TYPE
*/
MATRIX_TRANS_TYPE XLink::GetParamTrans(int i)
{
    CheckNTErrors(params != NULL, "parameter array cannot be empty!");
    char * p = (char*)params + i * paramSize;
    return *(MATRIX_TRANS_TYPE*)p;
}

/* 
create a hyperedge with two input tensors and a output tensor 
>> t1 - a tail tensor
>> t2 - another tail tensor
>> h - head tensor
>> id - id of the edge type
*/
void XLink::MakeLink(const XTensor * t1, const XTensor * t2, XTensor * h, int id)
{
    if(h == NULL)
        return;
    
    if (!t1->enableGrad)
        return;

    TensorList list(2);
    list.Add((XTensor*)t1);
    list.Add((XTensor*)t2);

    MakeLink(&list, h, id);
}

/*
create a hyperedge with two input tensors and a output tensor
>> t1 - a tail tensor
>> t2 - the second tail tensor
>> t3 - the third tail tensor
>> h - head tensor
>> id - id of the edge type
*/
void XLink::MakeLink(const XTensor * t1, const XTensor * t2, const XTensor * t3,XTensor * h, int id)
{
    if (h == NULL)
        return;

    if (!t1->enableGrad || !t2->enableGrad)
        return;
    
    TensorList list(3);
    list.Add((XTensor*)t1);
    list.Add((XTensor*)t2);
    list.Add((XTensor*)t3);

    MakeLink(&list, h, id);
}

/* 
create a hyper edge with a list of tensors and a output tensor 
>> list - a list of input tensors
>> h - head tensor
>> id - id of the edge type
*/
void XLink::MakeLink(const TensorList * list, XTensor * h, int id)
{
    /* forward */
    XLink &income = h->income;
    income.Reset();
    income.SetHead(h);
    income.SetType(id);

    for(int i = 0; i < list->count; i++){
        XTensor * t = (XTensor*)list->GetItem(i);
        if(t == NULL)
            continue;
        income.AddTail(t);
    }

    /* backward */
    for(int i = 0; i < list->count; i++){
        XTensor * t = (XTensor*)list->GetItem(i);
        if(t == NULL)
            continue;
        XLink &outgo = t->outgo;
        CheckNTErrors(outgo.head == NULL || outgo.head == t, 
                     "Wrong head of the hyperedge!");
        outgo.SetHead(t);
        outgo.AddTail(h);
    }
}

/* 
create a hyper edge with a input tensors and a list of output tensors
>> h - a input tensor
>> list - a list of output tensors
>> id - id of the edge type
*/
void XLink::MakeLink(XTensor * t, TensorList * list, int id)
{
    if (!t->enableGrad)
        return;

    /* forward */
    for(int i = 0; i < list->count; i++){
        XTensor * h = (XTensor*)list->GetItem(i);
        if(h == NULL)
            continue;
        XLink &income = h->income;
        income.Reset();
        income.SetHead(h);
        income.SetType(id);
        income.AddTail(t);
    }

    /* backward */
    XLink &outgo = t->outgo;
    outgo.SetHead(t);
    CheckNTErrors(outgo.head == NULL || outgo.head == t, "Wrong head of the hyperedge!");
    for(int i = 0; i < list->count; i++){
        XTensor * t = (XTensor*)list->GetItem(i);
        if(t == NULL)
            continue;
        outgo.AddTail(t);
    }
}

/* 
add parameters 
>> h - head
>> param - parameter we want introduce
*/
void XLink::AddParamToHead(XTensor * h, DTYPE param)
{
    CheckNTErrors(h != NULL, "head tensor cannot be empty!");
    h->income.AddParam(param);
}

/* 
add an integer parameter 
>> h - head
>> param - parameter we want introduce
*/
void XLink::AddParamToHeadInt(XTensor * h, int param)
{
    CheckNTErrors(h != NULL, "head tensor cannot be empty!");
    h->income.AddParam(&param, sizeof(int));
}

/* 
add a MATRIX_TRANS_TYPE parameter 
>> h - head
>> param - parameter we want introduce
*/
void XLink::AddParamToHeadTrans(XTensor * h, MATRIX_TRANS_TYPE param)
{
    CheckNTErrors(h != NULL, "head tensor cannot be empty!");
    h->income.AddParam(&param, sizeof(MATRIX_TRANS_TYPE));
}

/* 
add a boolean parameter 
>> h - head
>> param - parameter we want introduce
*/
void XLink::AddParamToHeadBool(XTensor * h, bool param)
{
    CheckNTErrors(h != NULL, "head tensor cannot be empty!");
    h->income.AddParam(&param, sizeof(bool));
}

/* 
add a pointer parameter 
>> h - head
>> param - parameter we want introduce
*/
void XLink::AddParamToHeadPointer(XTensor * h, void * param)
{
    CheckNTErrors(h != NULL, "head tensor cannot be empty!");
    h->income.AddParam(&param, sizeof(param));
}


/* 
replace a node with another, i.e., we redirect the links to the new node 
>> oldOne - the node to be replaced
>> newOne - the new node
*/
void XLink::Replace(const XTensor * oldOne, XTensor * newOne)
{
    if(oldOne == NULL || newOne == NULL)
        return;
    
    XLink &newIncome = newOne->income;
    XLink &newOutgo  = newOne->outgo;

    XLink::ClearOutgoing(newOne);
    XLink::ClearIncoming(newOne);

    /* incoming nodes */
    if(oldOne->income.typeID != 0){
        if(newIncome.tailNum < oldOne->income.tailNum){
            delete[] newIncome.tails;
            newIncome.tails = new XTensor*[oldOne->income.tailNum];
        }
        
        newIncome.SetType(oldOne->income.typeID);
        newIncome.head = newOne;
        newIncome.tailNum = oldOne->income.tailNum;
        memcpy(newIncome.tails, oldOne->income.tails, sizeof(XTensor*) * newIncome.tailNum);

        int paraArraySize = oldOne->income.paramNum * oldOne->income.paramSize;
        newIncome.params = new char[paraArraySize];
        memcpy(newIncome.params, oldOne->income.params, paraArraySize);
        newIncome.paramNum = oldOne->income.paramNum;

        /* update the link to each child node */
        for(int i = 0; i < newIncome.tailNum; i++){
            XTensor * child = newIncome.tails[i];
            XLink &childOutgo = child->outgo;
            bool hit = false;
            for(int j = 0; j < childOutgo.tailNum; j++){
                if(childOutgo.tails[j] == oldOne){
                    childOutgo.tails[j] = newOne;
                    hit = true;
                    break;
                }
            }

            if(childOutgo.tailNum > 0){
                CheckNTErrors(hit, "No proper node found in child.outgo edge!");
            }
        }
    }
    
    if(newOutgo.tailNum < oldOne->outgo.tailNum){
        delete[] newOutgo.tails;
        newOutgo.tails = new XTensor*[oldOne->outgo.tailNum];
    }

    /* outgoing nodes */
    newOutgo.head = newOne;
    newOutgo.tailNum = oldOne->outgo.tailNum;
    memcpy(newOutgo.tails, oldOne->outgo.tails, sizeof(XTensor*) * newOutgo.tailNum);

    /* update the link to each parent node */
    for(int i = 0; i < newOutgo.tailNum; i++){
        XTensor * parent = newOutgo.tails[i];
        XLink &parentIncome = parent->income;
        bool hit = false;
        for(int j = 0; j < parentIncome.tailNum; j++){
            if(parentIncome.tails[j] == oldOne){
                parentIncome.tails[j] = newOne;
                hit = true;
            }
        }

        if(parentIncome.tailNum > 0){
            CheckNTErrors(hit, "No proper node found in parent.income edge!");
        }
    }
}


/*
copy a node with another, i.e., we add the links to the new node
>> src - the node to be copied
>> tgt - the new node
*/
void XLink::Copy(const XTensor * reference, XTensor * target)
{
    if (reference == NULL || target == NULL)
        return;

    XLink &newIncome = target->income;
    XLink &newOutgo = target->outgo;

    XLink::ClearOutgoing(target);
    XLink::ClearIncoming(target);

    /* incoming nodes */
    if (reference->income.typeID != 0) {
        if (newIncome.tailNum < reference->income.tailNum) {
            delete[] newIncome.tails;
            newIncome.tails = new XTensor*[reference->income.tailNum];
        }

        newIncome.SetType(reference->income.typeID);
        newIncome.head = target;
        newIncome.tailNum = reference->income.tailNum;
        memcpy(newIncome.tails, reference->income.tails, sizeof(XTensor*) * newIncome.tailNum);

        int paraArraySize = reference->income.paramNum * reference->income.paramSize;
        newIncome.params = new char[paraArraySize];
        memcpy(newIncome.params, reference->income.params, paraArraySize);
        newIncome.paramNum = reference->income.paramNum;

        /* update the link to each child node */
        for (int i = 0; i < newIncome.tailNum; i++) {
            XTensor * child = newIncome.tails[i];
            XLink &childOutgo = child->outgo;
            bool hit = false;
            for (int j = 0; j < childOutgo.tailNum; j++) {
                if (childOutgo.tails[j] == reference) {
                    //childOutgo.tails[j] = target;
                    childOutgo.AddTail(target);
                    hit = true;
                    break;
                }
            }

            if (childOutgo.tailNum > 0) {
                CheckNTErrors(hit, "No proper node found in child.outgo edge!");
            }
        }
    }

    if (newOutgo.tailNum < reference->outgo.tailNum) {
        delete[] newOutgo.tails;
        newOutgo.tails = new XTensor*[reference->outgo.tailNum];
    }

    /* outgoing nodes */
    newOutgo.head = target;
    newOutgo.tailNum = reference->outgo.tailNum;
    memcpy(newOutgo.tails, reference->outgo.tails, sizeof(XTensor*) * newOutgo.tailNum);

    /* update the link to each parent node */
    for (int i = 0; i < newOutgo.tailNum; i++) {
        XTensor * parent = newOutgo.tails[i];
        XLink &parentIncome = parent->income;
        bool hit = false;
        for (int j = 0; j < parentIncome.tailNum; j++) {
            if (parentIncome.tails[j] == reference) {
                //parentIncome.tails[j] = target;
                parentIncome.AddTail(target);
                hit = true;
            }
        }

        if (parentIncome.tailNum > 0) {
            CheckNTErrors(hit, "No proper node found in parent.income edge!");
        }
    }
}
/* 
copy incoming edges of a given node
>> reference - the node we copy from
>> target - where we copy to
*/
void XLink::CopyIncoming(const XTensor * reference, XTensor * target)
{
    CheckNTErrors(reference && target, "Empty input tensors!");

    ClearIncoming(target);

    int tailNum = reference->income.tailNum;
    TensorList tails(tailNum);
    for(int i = 0; i < tailNum; i++){
        XTensor * tail = (XTensor*)reference->income.tails[i];
        tails.Add(tail);
    }

    MakeLink(&tails, target, reference->income.typeID);

    int paraNum = reference->income.paramNum;
    target->income.paramNum = paraNum;

    delete[] (char*)target->income.params;
    int size = paraNum * reference->income.paramSize;
    target->income.params = new char[size];
    memcpy(target->income.params, reference->income.params, size);
}

/* 
check the correctness of the network encoded in a root node (tensor) 
>> root - pointer to the root node
*/
void XLink::CheckNetwork(XTensor * root)
{
    XLink &income = root->income;
    if(income.head == NULL){
        CheckNTErrors(income.tailNum == 0, "Wrong number of the incoming edge tails!");
    }
    else{
        for(int i = 0; i < income.tailNum; i++){
            XTensor * child = income.tails[i];
            if(child == NULL)
                continue;
            XLink & childOutgo = child->outgo;
            bool hit = false;
            for(int j = 0; j < childOutgo.tailNum; j++){
                if(childOutgo.tails[j] == root){
                    hit = true;
                    break;
                }
            }
            CheckNTErrors(hit, "Wrong outgoing edge!");
        }
    }

    XLink &outgo = root->outgo;
    if(outgo.head == NULL){
        CheckNTErrors(outgo.tailNum == 0, "Wrong number of the incoming edge tails!");
    }
    else{
        for(int i = 0; i < outgo.tailNum; i++){
            XTensor * parent = outgo.tails[i];
            if(parent == NULL)
                continue;
            XLink & parentIncome = parent->income;
            bool hit = false;
            for(int j = 0; j < parentIncome.tailNum; j++){
                if(parentIncome.tails[j] == root){
                    hit = true;
                    break;
                }
            }
            CheckNTErrors(hit, "Wrong incoming edge!");
        }
    }

    for(int i = 0; i < income.tailNum; i++){
        XTensor * child = income.tails[i];
        CheckNetwork(child);
    }
}

/* 
show a node 
>> file - file to dump information
>> root - pointer to the node
*/
void XLink::ShowNode(FILE * file, XTensor * node)
{
    fprintf(file, "node %d - ", node->id);

    XLink &income = node->income;
    if(income.head == NULL){
        fprintf(file, "income[%d]: null ", income.tailNum);
    }
    else{
        fprintf(file, "income[%d, %s]: ", income.tailNum, GetOPName(income.typeID));
        for(int i = 0; i < income.tailNum; i++){
            XTensor * child = income.tails[i];
            if(child == NULL)
                fprintf(file, "na ");
            else
                fprintf(file, "%d ", child->id);
        }
    }
    fprintf(stderr, ", ");

    XLink &outgo = node->outgo;
    if(outgo.head == NULL || outgo.tailNum == 0){
        fprintf(file, "outgo[%d]: null ", outgo.tailNum);
    }
    else{
        fprintf(file, "outgo[%d]: ", outgo.tailNum);
        for(int i = 0; i < outgo.tailNum; i++){
            XTensor * parent = outgo.tails[i];
            if(parent == NULL)
                fprintf(file, "na ");
            else
                fprintf(file, "%d ", parent->id);
        }
    }

    fprintf(stderr, "\n");
}

/* 
search for a node in a top-down manner by its name 
>> top - the top most node
<< return - the node we found
*/
XTensor * XLink::SearchNode(XTensor * top, const char * name)
{
    if(!strcmp(top->name, name))
        return top;

    XLink &incoming = top->income;

    for(int i = 0; i < incoming.tailNum; i++){
        XTensor * child = incoming.tails[i];
        XTensor * hit = SearchNode(child, name);
        if(hit != NULL)
            return hit;
    }

    return NULL;
}

    
} // namespace nts(NiuTrans.Tensor)

