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
* $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-07-04
*/

#include "../../XUtility.h"
#include "../../XName.h"
#include "Select.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/*
generate a tensor with selected data in index along the given dimension

c = select(a)

>> a - input tensor
>> c - result tensor
>> index - the selected index
>> dim - the dimension along with which we do the job
*/

void _Select(const XTensor * a, XTensor * c, int* index, int dim)
{
    CheckNTErrors(a != NULL && c != NULL, "empty tensors!");
    CheckNTErrors(a->order == c->order, "The input and output tensors must in the same order!");
    CheckNTErrors(dim >= 0 && dim < a->order, "The input dimension is out of bounds!");
    CheckNTErrors(a->dataType == c->dataType, "The tensor must be of the same data type!");
    int stride = 1;
    for (int i = dim + 1; i < a->order; i++)
        stride *= a->dimSize[i];
    int copyTimes = 1;
    for (int i = 0; i < dim; i++)
    {
        copyTimes *= a->dimSize[i];
    }
    int cot = c->dimSize[dim];
    int blockSize = stride * a->unitSize;
    int stepSizeS = stride * a->dimSize[dim] * a->unitSize;
    int stepSizeT = stride * c->dimSize[dim] * a->unitSize;
    char * s = (char*)a->data;
    char * t = (char*)c->data;
    for (int i = 0; i < copyTimes; i++) {
        for (int j = 0; j < cot; ++j) {
            XMemCopy(t + j * blockSize, c->devID, s + index[j] * blockSize, a->devID, blockSize);
        }
        s += stepSizeS;
        t += stepSizeT;
    }
}

/*
generate a tensor with selected data in index along the given dimension

c = select(a)

>> a - input tensor
>> c - result tensor
>> index - the selected index
>> dim - the dimension along with which we do the job
*/
void _Select(const XTensor * a, XTensor * c, XTensor* index, int dim)
{
    if (index->devID >= 0)
    {
        int* indexCPU = new int[index->unitNum];
        XMemCopy(indexCPU, -1, index->data,index->devID, index->unitNum * sizeof(int));

        _Select(a, c, indexCPU, dim);
        delete[] indexCPU;
    }
    else
    {
        _Select(a, c, (int *)index->data, dim);
    }
}

/*
c = select(a)

>> a - input tensor
>> index - the selected index
>> dim - the dimension along with which we do the job 
<< return - the result of the generated tensor with selected data
*/
XTensor Select(const XTensor &a, XTensor &index, int dim)
{
    int order = a.order;
    int * dimSize = new int[order];

    CheckNTErrors(dim >= 0 && dim < a.order, "The input dimension is out of bounds!");

    for (int i = 0; i < a.order; i++) {
        if (i == dim) {
            dimSize[i] = index.dimSize[0];
        }
        else
            dimSize[i] = a.dimSize[i];
    }

    float dr = (!a.isSparse) ? 1.0F : a.denseRatio;
    XTensor c(order, dimSize, a.dataType, dr, a.devID, a.mem);
    c.SetTMPFlag();

    /* call _SelectRange function */
    _Select(&a, &c, &index, dim);

    /* tensor connection */
    if (a.enableGrad) {
        XLink::MakeLink(&a, &index, &c, GETANDSET_SELECT);
        XLink::AddParamToHeadInt(&c, dim);
    }

    /* destroy variables */
    delete[] dimSize;

    return c;
}

/* 
generate a tensor with selected data in range[low,high] along the given dimension 

c = select(a) 

>> a - input tensor
>> c - result tensor
>> dim - the dimension along with which we do the job
>> low - lower bound
>> high - higher bound.
Note that range [1,3] means that we select 1 and 2.
*/
void _SelectRange(const XTensor * a, XTensor * c, int dim, int low, int high)
{
    CheckNTErrors(a != NULL && c != NULL, "empty tensors!");
    CheckNTErrors(a->order == c->order, "The input and output tensors must in the same order!");
    CheckNTErrors(dim >= 0 && dim < a->order, "The input dimension is out of bounds!");
    CheckNTErrors(a->dataType == c->dataType, "The tensor must be of the same data type!");
    
    if(low >= high)
        return;

    for(int i = 0; i < a->order; i++){
        if(i == dim){
            CheckNTErrors(low >= 0 && low < a->dimSize[dim], "Illegal range specified!");
            CheckNTErrors(high > 0 && high <= a->dimSize[dim], "Illegal range specified!");
        }
        else{
            CheckNTErrors(a->dimSize[i] == c->dimSize[i], "The size of the dimensions should be same!");
        }
    }

    int stride = 1;
    for(int i = dim + 1; i < a->order; i++)
        stride *= a->dimSize[i];

    int copyTimes = 1;
    for (int i = 0; i < dim; i++) 
        copyTimes *= a->dimSize[i];

    int blockSize = stride * (high - low) * a->unitSize;
    int stepSizeS = stride * a->dimSize[dim] * a->unitSize;
    int stepSizeT = stride * c->dimSize[dim] * a->unitSize;
    char * s = (char*)a->data + stride * low * a->unitSize;
    char * t = (char*)c->data;
    for(int i = 0; i < copyTimes; i++){
        XMemCopy(t, c->devID, s, a->devID, blockSize);
        s += stepSizeS;
        t += stepSizeT;
    }
}

/* 
generate a tensor with selected data in range[low,high] along the given dimension (return an XTensor structure)
make a new tensor to keep the result and return it

c = select(a) 

>> a - input tensor
>> dim - the dimension along with which we do the job
>> low - lower bound
>> high - higher bound.
   Note that range [1,3] means that we select 1 and 2.
<< return - the result of the generated tensor with selected data
*/
XTensor SelectRange(const XTensor &a, int dim, int low, int high)
{
    int order = a.order;
    int * dimSize = new int[order];

    CheckNTErrors(dim >= 0 && dim < a.order, "The input dimension is out of bounds!");
    CheckNTErrors(low < high, "Illegal range specified!");
    
    for(int i = 0; i < a.order; i++){
        if(i == dim){
            CheckNTErrors(low >= 0 && low < a.dimSize[dim], "Illegal range specified!");
            CheckNTErrors(high > 0 && high <= a.dimSize[dim], "Illegal range specified!");
            dimSize[i] = high - low;
        }
        else
            dimSize[i] = a.dimSize[i];
    }

    float dr = (!a.isSparse) ? 1.0F : a.denseRatio;
    XTensor c(order, dimSize, a.dataType, dr, a.devID, a.mem);
    c.SetTMPFlag();

    /* call _SelectRange function */
    _SelectRange(&a, &c, dim, low, high);

    /* tensor connection */
    if (a.enableGrad) {
        XLink::MakeLink(&a, NULL, &c, GETANDSET_SELECT);
        XLink::AddParamToHeadInt(&c, dim);
        XLink::AddParamToHeadInt(&c, low);
        XLink::AddParamToHeadInt(&c, high);
    }

    /* destroy variables */
    delete[] dimSize;

    return c;
}


} // namespace nts(NiuTrans.Tensor)
