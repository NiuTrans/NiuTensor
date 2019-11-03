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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-12-17
 */

#include "OnehotAndIndex.h"
#include "OnehotAndIndex.cuh"
#include "SetData.h"

namespace nts{ // namespace nts(NiuTrans.Tensor)

/* 
convert onehot tensor to index tensor 

>> onehot - onehot tensor, which value is 0 or 1
>> index - index tensor, which value is an integer num
>> size - the last dimension size of the onehot tensor
*/
void _OnehotToIndex(const XTensor * onehot, XTensor * index, int size)
{
    CheckNTErrors(onehot->GetDim(-1) == size, "Illegal tensor dimension!");
    CheckNTErrors(onehot->order == index->order + 1, "Illegal tensor order!");
    CheckNTErrors(onehot->dataType == X_INT, "The onehot tensor must be in X_INT!")
    CheckNTErrors(index->dataType == X_INT, "The index tensor must be in X_INT!")

    for (int i = 0; i < index->order; i++)
        CheckNTErrors(index->GetDim(i) == onehot->GetDim(i), "Illegal tensor order!");

#ifdef USE_CUDA
    if(onehot->devID >= 0 && index->devID >= 0) {
        _CudaOnehotToIndex(onehot, index, size);
        return;
    }
#endif

    int blockNum = index->unitNum;
    int stride = size;

    int * onehotData = (int *)onehot->data;
    int * indexData = (int *)index->data;

    for (int i = 0; i < blockNum; i++) {
        int * od = onehotData + i * stride;
        int record = -1;
        for (int j = 0; j < stride; j++) {
            if (od[j] != 0) {
                if (record == -1)
                    record = j;
                else
                    ShowNTErrors("The value of onehot tensor is illegal!");
            }
        }
        indexData[i] = record;
    }

}

/* 
convert onehot tensor to index tensor (return an XTensor structure)
make a new tensor to keep the result and return it 

>> onehot - onehot tensor, which value is 0 or 1
>> size - the last dimension size of the onehot tensor
<< return - the index tensor
*/
XTensor OnehotToIndex(const XTensor & onehot, int size)
{
    CheckNTErrors(onehot.GetDim(-1) == size, "Illegal tensor dimension!");
    CheckNTErrors(onehot.dataType == X_INT, "The onehot tensor must be in X_INT!")

    XTensor index;
    InitTensorV2(&index, onehot.order - 1, onehot.dimSize, X_INT, 1.0F, onehot.devID, onehot.mem);
    index.SetTMPFlag();

    _OnehotToIndex(&onehot, &index, size);

    return index;
}

/* 
convert index tensor to onehot tensor 

>> index - index tensor, which value is an integer num
>> onehot - onehot tensor, which value is 0 or 1
>> size - the last dimension size of the onehot tensor
*/
void _IndexToOnehot(const XTensor * index, XTensor * onehot, 
                    int size, float labelSmoothingP)
{
    CheckNTErrors(onehot->GetDim(-1) == size, "Illegal tensor dimension!");
    CheckNTErrors(onehot->order == index->order + 1, "Illegal tensor order!");
    //CheckNTErrors(onehot->dataType == X_INT, "The onehot tensor must be in X_INT!")
    CheckNTErrors(index->dataType == X_INT, "The index tensor must be in X_INT!")

    for (int i = 0; i < index->order; i++)
        CheckNTErrors(index->GetDim(i) == onehot->GetDim(i), "Illegal tensor order!");

    //onehot->SetZeroAll();

    float confidence = 1 - labelSmoothingP;
    float lowconfidence = labelSmoothingP / size;

    _SetDataFixedFloat(onehot, lowconfidence);

#ifdef USE_CUDA
    if(onehot->devID >= 0 && index->devID >= 0) {
        _CudaIndexToOnehot(index, onehot, size, confidence, lowconfidence);
        return;
    }
#endif

    int blockNum = index->unitNum;
    int stride = size;

    int * indexData = (int *)index->data;
    DTYPE * onehotData = (DTYPE *)onehot->data;

    for (int i = 0; i < blockNum; i++) {
        int id = indexData[i];
        DTYPE * od = onehotData + i * stride;
        od[id] = confidence;
    }

}

/*
convert index tensor to onehot tensor

>> index - index tensor, which value is an integer num
>> onehot - onehot tensor, which value is 0 or 1
>> size - the last dimension size of the onehot tensor
*/
void _IndexToOnehot(int * index, int n, XTensor * onehot, int size, float labelSmoothingP)
{
    /*CheckNTErrors(onehot->GetDim(-1) == size, "Illegal tensor dimension!");
    CheckNTErrors(onehot->dataType == X_INT, "The onehot tensor must be in X_INT!")


        onehot->SetZeroAll();

#ifdef USE_CUDA
    if (onehot->devID >= 0) {
        
        delete[] cudaIndex;
        return;
    }
#endif

    int blockNum = n;
    int stride = size;

    int * indexData = (int *)index;
    int * onehotData = (int *)onehot->data;

    for (int i = 0; i < blockNum; i++) {
        int id = indexData[i];
        int * od = onehotData + i * stride;
        od[id] = 1;
    }*/
    XTensor* cudaIndex = NewTensor1DV2(n, X_INT, onehot->devID);
    cudaIndex->SetData(index, n);
    _IndexToOnehot(cudaIndex, onehot, size, labelSmoothingP);
    delete[] cudaIndex;

}

/* 
convert onehot tensor to index tensor (return an XTensor structure)
make a new tensor to keep the result and return it 

>> index - index tensor, which value is an integer num
>> size - the last dimension size of the onehot tensor
>> confidence - labelsmoothing
<< return - the onehot tensor
*/
XTensor IndexToOnehot(const XTensor & index, int size, float labelSmoothingP)
{
    CheckNTErrors(index.dataType == X_INT, "The onehot tensor must be in X_INT!")

    XTensor onehot;
    onehot.SetTMPFlag();
    
    int order = index.order;
    int * dim = new int[order + 1];
    memcpy(dim, index.dimSize, order * sizeof(int));
    dim[order] = size;
    InitTensorV2(&onehot, index.order + 1, dim, X_FLOAT, 1.0F, index.devID, index.mem);

    _IndexToOnehot(&index, &onehot, size, labelSmoothingP);

    delete[] dim;

    return onehot;
}

} // namespace nts(NiuTrans.Tensor)