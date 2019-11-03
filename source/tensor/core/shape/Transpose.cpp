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
 * $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-07-28
 * It is extreamly hot these days and i cannot sleep well. Fortunately we had 
 * good lunch of Steamed Cold Noodles. This made me feel much better!
 */

#include "Transpose.h"
#include "Merge.h"
#include "../../XUtility.h"
#include "../../XName.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/*
tensor transposition of dimensions i and j
b = transposed(a) 

For a input tensor a, we tranpose the dimensions i and j of it.
E.g., let a be a tensor of size x * y * z, i = 0, j = 2, 
then the output will be a tensor of size z * y * x.

>> a - the input tensor
>> b - the output tensor by transpose tensor a with specified dimensions i and j
>> i - the transposed dimension
>> j - the transposed dimension
*/
void _Transpose(const XTensor * a, XTensor * b, const int i, const int j)
{
    CheckNTErrors(a && b, "Empty tensors");
    CheckNTErrors(a->order == b->order, "Wrong tensor orders");
    CheckNTErrors(a->unitNum == b->unitNum && a->unitSize == b->unitSize, "Wrong tensor sizes");
    CheckNTErrors(a->order > i && i >= 0, "index of dimension is out of scope!");
    CheckNTErrors(a->order > j && j >= 0, "index of dimension is out of scope!");

    for(int k = 0; k < a->order; k++){
        if(k == i){
            CheckNTErrors(a->dimSize[k] == b->dimSize[j], "Wrong dimension size in transposition");
        }
        else if(k == j){
            CheckNTErrors(a->dimSize[k] == b->dimSize[i], "Wrong dimension size in transposition");
        }
        else{
            CheckNTErrors(a->dimSize[k] == b->dimSize[k], "Wrong dimension size in transposition");
        }
    }

    if(i == j){
        XMemCopy(b->data, b->devID, a->data, a->devID, b->unitNum * b->unitSize);
    }
    else{
        int I = MIN(i, j);
        int J = MAX(i, j);
        int * dims = new int[a->order + 1];

        for(int k = 0; k <= J; k++)
            dims[k] = a->dimSize[k];
        dims[J + 1] = -1;
        for(int k = J + 1; k < a->order; k++)
            dims[k + 1] = a->dimSize[k];

        /* reshape tensor a form (..., n_I, ..., n_J, ...) => (..., n_I, ..., n_J, 1, ...)*/
        XTensor * aTMP =  new XTensor(a->order + 1, dims, a->dataType, a->denseRatio, a->devID, a->mem);
        aTMP->data = a->data;

        for(int k = 0; k < I; k++)
            dims[k] = a->dimSize[k];
        for(int k = I + 1; k <= J; k++)
            dims[k - 1] = a->dimSize[k];
        dims[J] = a->dimSize[I];
        for(int k = J + 1; k < a->order; k++)
            dims[k] = a->dimSize[k];

        /* reshape tensor b form (..., m_I, ..., m_J, ...) => (..., m_J, m_I, ...) */
        b->Reshape(b->order, dims);

        /* tensor (..., n_I, ..., n_J, 1, ...) => tensor (..., m_J, m_I, ...) */
        _Merge(aTMP, b, J + 1, I);

        memcpy(dims, a->dimSize, sizeof(int) * a->order);
        dims[I] = a->dimSize[J];
        dims[J] = a->dimSize[I];

        /* reshape tensor b form (..., m_J, m_I, ...) => (..., m_J, ..., m_I, ...) =>  */
        b->Reshape(b->order, dims);

        aTMP->data = NULL;
        delete[] dims;
        delete aTMP;
    }
}

/*
tensor transposition of dimensions i and j (return an XTensor structure).
make a new tensor to keep the result and return it.
b = transposed(a)

For a input tensor a, we tranpose the dimensions i and j of it.
E.g., let a be a tensor of size x * y * z, i = 0, j = 2, 
then the output will be a tensor of size z * y * x.

>> a - the input tensor
>> i - the transposed dimension
>> j - the transposed dimension
<< return - the output tensor by transpose tensor a with specified dimensions i and j
*/
XTensor Transpose(const XTensor &a, const int i, const int j)
{
    CheckNTErrors(a.order > i && i >= 0, "index of dimension is out of scope!");
    CheckNTErrors(a.order > j && j >= 0, "index of dimension is out of scope!");

    int order = a.order;
    int * dimSize = new int[order];
    for(int k = 0; k < order; k++){
        if(k == i)
            dimSize[k] = a.dimSize[j];
        else if(k == j)
            dimSize[k] = a.dimSize[i];
        else
            dimSize[k] = a.dimSize[k];
    }

    float dr = (!a.isSparse) ? 1.0F : a.denseRatio;
    XTensor b(order, dimSize, a.dataType, dr, a.devID, a.mem);
    b.SetTMPFlag();

    /* call _Transpose function */
    _Transpose(&a, &b, i, j);
    
    /* tensor connection */
    if (a.enableGrad) {
        XLink::MakeLink(&a, NULL, &b, SHAPE_TRANSPOSE);
        XLink::AddParamToHeadInt(&b, i);
        XLink::AddParamToHeadInt(&b, j);
    }

    /* destroy variables */
    delete[] dimSize;

    return b;
}

}
