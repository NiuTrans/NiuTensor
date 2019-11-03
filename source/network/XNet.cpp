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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-12
 */

#include "XNet.h"
#include "XNoder.h"
#include "XBackwardLoss.h"
#include "XBackwardMath.h"
#include "XBackwardFunc.h"
#include "XBackwardShape.h"
#include "../tensor/XName.h"

namespace nts{

unsigned int netIDGlobal = 0;
MUTEX_HANDLE netMutex;

/* generate a network id */
unsigned int MakeNetID()
{
    if(netIDGlobal == 0)
        MUTEX_INIT(netMutex);

    MUTEX_LOCK(netMutex);
    netIDGlobal += 3;
    unsigned int id = netIDGlobal;
    MUTEX_UNLOCK(netMutex);

    return id;
}

void XNetClearAll()
{
    MUTEX_DELE(netMutex);
}

/* constructor */
XNet::XNet()
{
    nodes.Clear();
    isGradEfficient = true;
}

/* de-constructor */
XNet::~XNet()
{
}

/* clear the network */
void XNet::Clear()
{
    nodes.Clear();
    gradNodes.Clear();
    outputs.Clear();
    inputs.Clear();
}

/* 
backward propagation to obtain gradient
>> root - root node (output) of the network
>> loss - name of loss function
*/
void XNet::Backward(XTensor &root)
{
    TensorList roots(1);
    roots.Add(&root);

    Backward(roots);
}

/* 
backward propagation to obtain gradient wrt. the loss/error function
with a number of root nodes 
>> roots - a list of root nodes (output) of the network
*/
void XNet::Backward(TensorList &roots)
{
    Traverse(roots);

    /* label tensors where the backward computation is neccessary */
    if(isGradEfficient)
        MakeEfficientNet();

    for(int i = 0; i < nodes.count; i++){
        XTensor * node = (XTensor*)nodes.Get(i);
        node->visitMark = NODE_UNFINISHED;
    }

    /* back-propagation from output to input */
    for(int i = nodes.count - 1; i >= 0; i--){
        XTensor * node = (XTensor*)nodes.Get(i);

        if(node->mem != NULL){
            CheckNTErrors(node->mem->bufUsed < BUF_PITCH, "Illegal access of buffer!");
        }

        if(node->visitMark != NODE_FINISHED)
            BackwardNode(node, isGradEfficient); 

        if(isGradEfficient){
            XLink & outgo = node->outgo;
            for(int i = 0; i < outgo.tailNum; i++){
                XTensor * parent = outgo.tails[i];
                ClearGrad(parent);
            }

            if(XNoder::IsLeaf(node))
                ClearGrad(node);
        }
    }
}

/* 
backward computation for a given node 
>> node - the node keeps the result of an operation (e.g., activation function)
>> isEfficient - indicates whether the back-propagation is compuated in an
                 efficient manner
*/
void XNet::BackwardNode(XTensor * node, bool isEfficent)
{
    if(node == NULL || node->visitMark == NODE_FINISHED)
        return;

    if(!XNoder::IsLeaf(node)){
        /* post processing for parent nodes */
        BackwardNodePost(node, isEfficent);

        /* process the current node */
        if(XMathGrad::IsMathOP(node))
            XMathGrad::MakeGrad(node, isEfficent);
        else if(XFuncGrad::IsFunc(node))
            XFuncGrad::MakeGrad(node, isEfficent);
        else if(XShapeGrad::IsShapeOP(node))
            XShapeGrad::MakeGrad(node, isEfficent);
        else if(XLossGrad::IsLossOP(node))
            XLossGrad::MakeGrad(node, isEfficent);
        else{
            ShowNTErrors("Wrong node type!");
        }
    }
    else{
        node->visitMark = NODE_FINISHED;
    }
}

/* 
backward computation (in post processing) for a given node 
>> node - the node whose parent nodes are not processed yet. So
          we do the job at the child node.
*/
void XNet::BackwardNodePost(XTensor * node, bool isEfficent)
{
    bool isSplitList = false;
    XLink &outgo = node->outgo;
    for(int i = 0; i < outgo.tailNum; i++){
        if(outgo.tails[i]->income.typeID == SHAPE_SPLIT_LIST)
            isSplitList = true;
    }

    if(isSplitList)
        XShapeGrad::PostProcessing(node, SHAPE_SPLIT_LIST, isEfficent);
}

/* 
traverse the net and find the topological order by 
depth-first search (Tarjan's algorithm) 
>> root - root node (or output of the net)
*/
void XNet::Traverse(XTensor &root)
{
    TensorList roots(1);
    roots.Add(&root);

    Traverse(roots);
}

/* 
traverse the net and find the topological order by 
depth-first search (Tarjan's algorithm) 
>> roots - a list of roots (or output nodes)
*/
void XNet::Traverse(TensorList &roots)
{
    id = MakeNetID();
    nodes.Clear();
 
    for (int i = 0; i < roots.count; i++)
        TarjanVisit((XTensor*)roots.Get(i), nodes, id);

    for(int i = 0; i < nodes.count; i++){
        XTensor * node = (XTensor*)nodes.Get(i);
        if(XNoder::IsRoot(node))
            outputs.Add(node);
        if(XNoder::IsLeaf(node))
            inputs.Add(node);
        if(XNoder::IsGrad(node))
            gradNodes.Add(node);
    }
}

/* 
depth-first search given a node (Tarjan's algorithm for topological ordering)
>> node - the node to visit (mark 0:unvisited, 1:visiting, 2:done)
>> orders - topological order of the nodes
>> code - code of the network
*/
void XNet::TarjanVisit(XTensor * node, TensorList &orders, const unsigned int code)
{
    if(node == NULL)
        return;

    //fprintf(stderr, "%d\n", node->id);
    if(node->visitMark == code + 1){
        ShowNTErrors("There is a circle in the network\n");
    }
    else if(node->visitMark <= code){
        node->visitMark = code + 1;
        XLink &income = node->income;
        for(int i = 0; i < income.tailNum; i++){
            XTensor * child = income.tails[i];
            if(child == NULL)
                continue;
            TarjanVisit(child, orders, code);
        }
        node->visitMark = code + 2;
        orders.Add(node);
    }
    else if(node->visitMark == code + 2){
    }
}

/* 
dump network information 
>> file - the file for dumping
*/
void XNet::Dump(FILE * file)
{
    for(int i = 0; i < nodes.count; i++){
        XTensor * node =  (XTensor*)nodes.Get(i);
        fprintf(file, "node %d: %d\n", i, node->id);
        node->Dump(file, "tensor: ");
        if(node->grad != NULL)
            node->grad->Dump(file, "grad: ");
        else
            fprintf(file, "no gradient!\n");
        fprintf(file, "\n");
    }
}

/* 
set the flag of gradient-efficient 
>> flag - the flag
*/
void XNet::SetGradEfficientFlag(bool flag)
{
    isGradEfficient = flag;
}

/* generate the gradient-efficient flag for every node */
void XNet::MakeEfficientNet()
{
    /* back-propagation from output to input */
    for(int i = 0; i < nodes.count; i++){
        XTensor * node = (XTensor*)nodes.Get(i);
        XLink &income = node->income;
        for(int j = 0; j < income.tailNum; j++){
            XTensor * child = income.tails[j];
            if(child->isGrad || child->isVar){
                node->SetGradFlag(true);
                break;
            }

        }
    }
}

/* 
clear the graident information if the node is no use 
>> node - the node that we want to clear
*/
void XNet::ClearGrad(XTensor * node)
{
    if(node->isVar)
        return;
    if(node->grad == NULL)
        return;
    if(node->visitMark != NODE_FINISHED)
        return;

    XLink & income = node->income;

    bool finished = true;
    for(int i = 0; i < income.tailNum; i++){
        XTensor * child = income.tails[i];
        if(child->visitMark != NODE_FINISHED){
            finished = false;
            break;
        }
    }

    if(finished){
        //fprintf(stderr, "del %d %ld\n", node->id, node->grad->unitNum);
        delete node->grad;
        node->grad = NULL;
    }
}

/* 
show network topology 
>> file - file to dump information
>> node - pointer to the node
*/
void XNet::ShowNetwork(FILE * file, XTensor * node)
{
    TensorList roots(1);
    roots.Add(node);

    Traverse(roots);

    XLink::ShowNode(file, node);

    /* go over nodes in its topological order */
    for(int i = nodes.count - 1; i >= 0; i--){
        XTensor * n = (XTensor*)nodes.Get(i);
        XLink::ShowNode(file, n);
    }
}

/*
search for a node in a top-down manner by its name
>> top - the top most node
<< return - the node we found
*/
//XTensor * XNet::SearchNode(XTensor * top, const char * name)
//{
    //return XLink::SearchNode(top, name);
//}

}
