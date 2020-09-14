/* NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2017, Natural Language Processing Lab, Northeastern University.
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
 * $Created by: Xu Chen (email: hello_master1954@163.com) 2018-11-30
 */

#include "CopyIndexed.cuh"
#include "../../XDevice.h"
#include "../../XUtility.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

#ifdef USE_CUDA

/*
copy selected sub-tensors where indeces are kept in tensors (kenerl version)

>> s - the source tensor
>> t - the target tensor
>> dim - the leading dimension to define "sub-tensors"
         e.g., for a tensor of size (3, 2, 4) and dim = 2, 
         we have 4 sub-tensors of size (3, 2)
>> srcIndex - the tensor to save the index of the source sub-tensors
>> tgtIndex - the tensor to save the index of the target sub-tensors
>> copyNum - number of the sub-tensors we copy for each source index, 
             e.g., for srcIndex = [1,4] and copyNum = 2,
             we actually copy the source sub-tensors 1, 2, 4, 5
*/
template <class T>
__global__
void KernelCopyIndexed(T * sData, T * tData, int * sIndex, int * tIndex,
 
                       int blockNum, int blockSizeSrc, int blockSizeTgt, 
                       int stride, int indexSize, int copyNum)
{
    __shared__ T * sp[MAX_CUDA_THREAD_NUM_PER_BLOCK];
    __shared__ T * tp[MAX_CUDA_THREAD_NUM_PER_BLOCK];

    /* block id */
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    /* offset in each block */
    int offset = blockDim.y * blockIdx.y + threadIdx.y;

    if(i >= blockNum * indexSize * copyNum || offset >= stride)
        return;

    int realIndexSize = indexSize * copyNum;

    int realBlockNum = i / realIndexSize;
    int realIndex = i % realIndexSize;

    int realSrcIndex = sIndex[realIndex / copyNum] + realIndex % copyNum;
    int realTgtIndex = tIndex[realIndex / copyNum] + realIndex % copyNum;

    if(threadIdx.y == 0){
        sp[threadIdx.x] = sData + realBlockNum * blockSizeSrc + realSrcIndex * stride;
        tp[threadIdx.x] = tData + realBlockNum * blockSizeTgt + realTgtIndex * stride;
    }

    __syncthreads();

    T * s = sp[threadIdx.x];
    T * t = tp[threadIdx.x];

    t[offset] = s[offset];
}

/*
copy selected sub-tensors where indeces are kept in tensors

>> s - the source tensor
>> t - the target tensor
>> dim - the leading dimension to define "sub-tensors"
         e.g., for a tensor of size (3, 2, 4) and dim = 2, 
         we have 4 sub-tensors of size (3, 2)
>> srcIndex - the tensor to save the index of the source sub-tensors
>> tgtIndex - the tensor to save the index of the target sub-tensors
>> copyNum - number of the sub-tensors we copy for each source index, 
             e.g., for srcIndex = [1,4] and copyNum = 2,
             we actually copy the source sub-tensors 1, 2, 4, 5
*/
void _CudaCopyIndexed(const XTensor * s, XTensor * t, int dim,
                      const XTensor * srcIndex, const XTensor * tgtIndex,
                      int copyNum)
{
    int devID = s->devID;
    int order = s->order;
    int indexSize = srcIndex->unitNum;

    int blockNum = 1;
    int stride = 1;
    int blockSizeSrc = 1;
    int blockSizeTgt = 1;

    for (int i = 0; i < dim; i++)
        blockNum *= s->GetDim(i);
    
    for (int i = dim + 1; i < order; i++)
        stride *= s->GetDim(i);

    blockSizeSrc = stride * s->GetDim(dim);
    blockSizeTgt = stride * t->GetDim(dim);

    int cudaGrids[3];
    int cudaBlocks[3];

    int devIDBackup;
    ProtectCudaDev(devID, devIDBackup);

    GDevs.GetCudaThread2D(devID, blockNum * indexSize * copyNum, stride, MAX_INT, cudaGrids, cudaBlocks);

    dim3 blocks(cudaGrids[0], cudaGrids[1]);
    dim3 threads(cudaBlocks[0], cudaBlocks[1]);

    if (s->dataType == DEFAULT_DTYPE && t->dataType == DEFAULT_DTYPE) {
        DTYPE * sData = (DTYPE*)s->data;
        DTYPE * tData = (DTYPE*)t->data;

        int * sIndex = (int*)srcIndex->data;
        int * tIndex = (int*)tgtIndex->data;

        KernelCopyIndexed<<<blocks, threads >>>(sData, tData, sIndex, tIndex,
                                                blockNum, blockSizeSrc, blockSizeTgt,
                                                stride, indexSize, copyNum);
    }
    else if (s->dataType == X_FLOAT16 && t->dataType == X_FLOAT16) {
        half * sData = (half*)s->data;
        half * tData = (half*)t->data;

        int * sIndex = (int*)srcIndex->data;
        int * tIndex = (int*)tgtIndex->data;
        KernelCopyIndexed<<<blocks, threads>>>(sData, tData, sIndex, tIndex,
                                               blockNum, blockSizeSrc, blockSizeTgt,
                                               stride, indexSize, copyNum);
    }
    else {
        ShowNTErrors("Unsupported dataType!");
    }
    BacktoCudaDev(devID, devIDBackup);

}

#endif // USE_CUDA

} // namespace nts(NiuTrans.Tensor)