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
* $Created by: LI Yinqiao (email: li.yin.qiao.2012@hotmail.com) 2019-10-23
*/

#include "../../XTensor.h"
#include "SetAscendingOrder.cuh"
#include "SetAscendingOrder.h"

namespace nts { // namespace nts(NiuTrans.Tensor)

/* 
set the cell to the ascending order along a given dimension 
>> tensor - input tensor
>> dim - the dimension specified
*/
void SetAscendingOrder(XTensor & tensor, int dim)
{
    CheckNTErrors(dim < tensor.order, "Wrong dimension specified!");
    CheckNTErrors(tensor.dataType == X_INT, "TODO!");

    if(dim < 0){        
        int o = tensor.order;
        int ds[MAX_TENSOR_DIM_NUM];
        memcpy(ds, tensor.dimSize, sizeof(int) * tensor.order);

        tensor.Reshape(tensor.unitNum);
        SetAscendingOrder(tensor, 0);
        tensor.Reshape(o, ds);

        return;
    }

    if(tensor.devID >= 0){
#ifdef USE_CUDA
        CudaSetAscendingOrder(&tensor, dim);
#else
        ShowNTErrors("Plesae specify USE_CUDA and recompile the code!");
#endif
    }
    else{
        int stride = 1;
        int blockNum = 1;
        int strideNum = tensor.dimSize[dim];
        for(int i = 0; i < dim; i++)
            blockNum *= tensor.dimSize[i];

        for(int i = dim + 1; i < tensor.order; i++)
            stride *= tensor.dimSize[i];

        for(int k = 0; k < blockNum; k++){
            for(int j = 0; j < strideNum; j++){
                int * d = (int*)tensor.data + stride * strideNum * k + stride * j;
                for(int i = 0; i < stride; i++)
                    d[i] = j;
            }
        }
    }
}

} // namespace nts(NiuTrans.Tensor)