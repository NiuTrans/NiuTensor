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
 * We expected a heavy rain today but a drizzle came down. Should I
 * take a big umbrella?
 */

#include "../tensor/XTensor.h"
#include "../tensor/function/FHeader.h"
#include "../tensor/loss/LHeader.h"

#ifndef __XNET_H__
#define __XNET_H__

namespace nts{

/* management of tensor net (or graph) */
struct XNet
{
    /* id of the network */
    unsigned int id;

    /* tensor nodes of the network (in order) */
    TensorList nodes;

    /* tensor nodes to keep gradient for output (e.g., SGD)*/
    TensorList gradNodes;

    /* output nodes of the network */
    TensorList outputs;

    /* input nodes of the network */
    TensorList inputs;

    /* indicates whether the network just keeps the gradient for parameter tensors */
    bool isGradEfficient;

    /* constructor */
    XNet();

    /* de-constructor */
    ~XNet();

    /* clear the network */
    void Clear();

    /* backward propagation to obtain gradient */
    void Backward(XTensor &root);

    /* backward propagation to obtain gradient wrt. the loss/error function
       with a number of root nodes */
    void Backward(TensorList &roots);

    /* backward computation for a given node */
    void BackwardNode(XTensor * node, bool isEfficent = false);

    /* backward computation (in post processing) for a given node */
    void BackwardNodePost(XTensor * node, bool isEfficent = false);

    /* traverse the net and find the topological order by 
       depth-first search (Tarjan's algorithm) */
    void Traverse(XTensor &root);

    /* traverse the net and find the topological order by 
       depth-first search (Tarjan's algorithm) */
    void Traverse(TensorList &roots);

    /* depth-first search given a node (Tarjan's algorithm for topological ordering) */
    void TarjanVisit(XTensor * node, TensorList &orders, const unsigned int code);

    /* dump network information */
    void Dump(FILE * file);

    /* set the flag of gradient-efficient */
    void SetGradEfficientFlag(bool flag = true);

    /* generate the gradient-efficient flag for every node */
    void MakeEfficientNet();

    /* clear the graident information if the node is no use */
    void ClearGrad(XTensor * node);

    /* show network topology */
    void ShowNetwork(FILE * file, XTensor * node);

    /* search a node in a top-down manner by its name */
    //static
    //XTensor * SearchNode(XTensor * top, const char * name);
};

/* we make a unique id for every tensor */
extern unsigned int netIDGlobal;
extern MUTEX_HANDLE netMutex;
extern unsigned int MakeNetID();
extern void XNetClearAll();

}

#endif