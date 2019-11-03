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
* $Created by: LI Yinqiao (li.yin.qiao.2012@hotmail.com) 2018-06-14
*/

#include "FlushToMem.cuh"
#include "../../XUtility.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/*
flush a list of XTensor to GPU memory
>> mList - list of the tensors
>> devID - target GPU id
>> GPUMem - memory pool for the GPU
*/
void CudaCPUToGPUFlush(TensorList * mList, int devID, XMem * GPUMem)
{
    if (mList == NULL || mList->count == 0)
        return;

#ifdef USE_CUDA
    int size = 0, p = 0;
    int reqiredSize = 0;

    /* compute the requried memory size */
    for (int i = 0; i < mList->count; i++) {
        XTensor * m = (XTensor*)mList->GetItem(i);

        CheckNTErrors((m->devID < 0), "Cannot do gpu-flush on matrices that are already on GPUs.");

        if (m->isSparse)
            reqiredSize = sizeof(int) + (sizeof(int) + m->unitSize) * m->unitNumNonZero;
        else
            reqiredSize = m->unitSize * m->unitNum;

        size += reqiredSize;
    }

    char * data = new char[size];
    char * GPUData = GPUMem != NULL ? (char*)GPUMem->Alloc(GPUMem->devID, size):
                                      (char*)XMemAlloc(devID, size);
    int pSize = 0;

    /* place the data in a memory block */
    for (int i = 0; i < mList->count; i++) {
        XTensor * m = (XTensor*)mList->GetItem(i);

        if (m->isSparse)
            pSize = sizeof(int) + (sizeof(int) + m->unitSize) * m->unitNumNonZero;
        else
            pSize = m->unitSize * m->unitNum;

        reqiredSize = pSize;

        memcpy(data + p, m->data, pSize);

        if (m->dataHost != NULL)
            delete[](char*)m->dataHost;

        if(m->mem == NULL)
            delete[] (char*)m->data;

        m->dataHost = NULL;
        m->data = GPUData + p;
        m->devID = GPUMem != NULL ? GPUMem->devID : devID;
        m->mem = GPUMem;

        p += reqiredSize;
    }

    /* copy from CPU memory to GPU memory */
    cudaMemcpy(GPUData, data, size, cudaMemcpyHostToDevice);

    delete[] data;
#endif
}

/* copy the data from GPU memory to CPU memory */
void CudaGPUToCPUFlush(XTensor * tensor)
{
    CheckNTErrors((sizeof(DTYPE) == tensor->unitSize), "Unsupported data type.");

    if (tensor->dataHost != NULL)
        delete[](char*)tensor->dataHost;

    if (tensor->isSparse) {
        int num = int(tensor->unitNum * tensor->denseRatio + 1);
        cudaMemcpy(&num, (DTYPE*)tensor->data, sizeof(int), cudaMemcpyDeviceToHost);

        int tupleSize = sizeof(int) + sizeof(DTYPE);
        int size = sizeof(int) + tupleSize*(num);

        CheckNTErrors((size >= 0), "Illegal data size in the sparse matrix!");

        tensor->dataHost = new char[size];
        cudaMemcpy(tensor->dataHost, tensor->data, size, cudaMemcpyDeviceToHost);
    }
    else {
        tensor->dataHost = new char[tensor->unitNum * tensor->unitSize];
        if (tensor->data != NULL)
            XMemCopy(tensor->dataHost, -1, tensor->data, tensor->devID, tensor->unitNum * tensor->unitSize);
        else
            memset(tensor->dataHost, 0, tensor->unitNum * tensor->unitSize);
    }
}
#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)