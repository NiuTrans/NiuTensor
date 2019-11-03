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
#include "XGlobal.h"
#include "XTensor.h"

#ifndef __XLINK_H__
#define __XLINK_H__

#include "XGlobal.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/* cross reference */
struct XTensor;

#define MAX_OP_NAME_LENGTH 64
#define PARAM_UNTI_SIZE    64

/*
This defines the link among tensors in networks. XLink can be
cast as a hyperedge in a graph. when we compute on tensors, we actually create a
network where nodes are tensors and edges the connections among them. Each connection is
a hyperedge whose head is the output tensor and tails are input tensors. E.g,
c = a + b
represents a network with three nodes (a, b and c) and a hyperedge that links a and b (tails) to c (head).
 
   + (=c)
  / \
 a   b
 
for c, we have a incoming edge (a, b) -> c
for a, we also have a edge c -> a in the reverse order (in a view of acyclic directed graphs)
*/
struct XLink
{
    /* head of the hyperedge */
    XTensor *  head;

    /* tails of the hyperedge */
    XTensor ** tails;

    /* number of tails */
    int tailNum;

    /* parameters used. e.g., c = a * b * \alpha 
       scalar \alpha is the parameter */
    void * params;

    /* number of parameters */
    int paramNum;

    /* size of each parameter */
    static int paramSize;

    /* name of the hyperedge type. e.g., sum, mul ... */
    char type[MAX_OP_NAME_LENGTH];
    
    /* type id */
    int typeID;

    /* caculator (pointer to the class for computation) */
    void * caculator;
    
    /* constuctor */
    XLink();
    
    /* deconstructor */
    ~XLink();

    /* reset it */
    void Reset();

    /* clear it */
    void Clear();

    /* clear tails */
    void ClearTail();

    /* clear the incoming node list of tensor node */
    static
    void ClearIncoming(XTensor * node);
    
    /* clear the outgoing node list of tensor node */
    static
    void ClearOutgoing(XTensor * node);

    /* set edge type id and name */
    void SetType(int id);

    /* set head */
    void SetHead(XTensor * h);

    /* add a tail */
    void AddTail(XTensor * t);

    /* add two tails in one time */
    void AddTwoTails(XTensor * t1, XTensor * t2);

    /* add a parameter in default type */
    void AddParam(DTYPE param);

    /* add a parameter */
    void AddParam(void * param, int size);

    /* get a paramter in default type */
    DTYPE GetParam(int i);

    /* get a paramter in integer */
    int GetParamInt(int i);

    /* get a paramter in pointer */
    void * GetParamPointer(int i);
    
    /* get a parameter in MATRIX_TRANS_TYPE */
    MATRIX_TRANS_TYPE GetParamTrans(int i);

    /* create a hyper edge with two input tensors and a output tensor */
    static
    void MakeLink(const XTensor * t1, const XTensor * t2, XTensor * h, int id);

    /* create a hyper edge with three input tensors and a output tensor */
    static
    void MakeLink(const XTensor * t1, const XTensor * t2, const XTensor * t3, XTensor * h, int id);

    /* create a hyper edge with a list of input tensors and a output tensor */
    static
    void MakeLink(const TensorList * list, XTensor * h, int id);

    /* create a hyper edge with a input tensors and a list of output tensors */
    static
    void MakeLink(XTensor * h, TensorList * list, int id);

    /* add a parameter */
    static
    void AddParamToHead(XTensor * h, DTYPE param);

    /* add an integer parameter */
    static
    void AddParamToHeadInt(XTensor * h, int param);

    /* add a MATRIX_TRANS_TYPE parameter */
    static
    void AddParamToHeadTrans(XTensor * h, MATRIX_TRANS_TYPE param);

    /* add a boolean parameter */
    static
    void AddParamToHeadBool(XTensor * h, bool param);

    /* add a pointer parameter */
    static
    void AddParamToHeadPointer(XTensor * h, void * param);

    /* replace a node with another, i.e., we redirect the links to the new node */
    static 
    void Replace(const XTensor * oldOne, XTensor * newOne);

    /* copy a node with another, i.e., we add the links to the new node */
    static
    void Copy(const XTensor * reference, XTensor * target);

    /* copy links of a given node */
    static
    void CopyIncoming(const XTensor * reference, XTensor * target);

    /* check the correctness of the network encoded in a root node (tensor) */
    static
    void CheckNetwork(XTensor * root);

    /* show a node */
    static
    void ShowNode(FILE * file, XTensor * node);

    /* search a node in a top-down manner by its name */
    static
    XTensor * SearchNode(XTensor * top, const char * name);
};
    
} // namespace nts(NiuTrans.Tensor)

#endif // __XLINK_H__
