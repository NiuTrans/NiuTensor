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

#include "SetAscendingOrder.cuh"
#include "../../XDevice.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/* 
set the cell to the ascending order along a given dimension (kernel code)
>> data - the data array
>> stride - how many items we go ove when move to the next item along the dimension
>> strideNum - size of the given dimension
>> blockNum - block number
*/
__global__
void KernelSetAscendingOrder(int * data, int stride, int strideNum, int blockNum)
{
    __shared__ int iBlock[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ int iOffset[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    /* index along the "stride" dimension */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* index along the leading dimension */
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i >= stride * blockNum || j >= strideNum)
        return;

    if(threadIdx.y == 0){
        iBlock[threadIdx.x] = i / stride;
        iOffset[threadIdx.x] = i % stride;
    }
    __syncthreads();
    
    int * d = (int*)data + (iBlock[threadIdx.x] * strideNum + j) * stride + iOffset[threadIdx.x];
    *d = j;
}

/* 
set the cell to the ascending order along a given dimension
>> a - the tensor
>> dim - the dimension
*/
void CudaSetAscendingOrder(XTensor * a, int dim)
{
    CheckNTErrors((a->dataType == X_INT), "TODO!");

	int stride = 1;
    int blockNum = 1;
    int strideNum = a->dimSize[dim];
    for(int i = 0; i < dim; i++)
        blockNum *= a->dimSize[i];

    for(int i = dim + 1; i < a->order; i++)
        stride *= a->dimSize[i];

    int gridSize[3];
    int blockSize[3];

    GDevs.GetCudaThread2D(a->devID, strideNum, stride * blockNum, MAX_INT, gridSize, blockSize);

    int devIDBackup;
    ProtectCudaDev(a->devID, devIDBackup);

    KernelSetAscendingOrder<<<dim3(gridSize[1], gridSize[0]), dim3(blockSize[1], blockSize[0])>>>
                            ((int*)a->data, stride, strideNum, blockNum);

    BacktoCudaDev(a->devID, devIDBackup);
}
#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)