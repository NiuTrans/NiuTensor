/*
 * NiuTrans.Tensor - an open-source tensor library
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
 * $Created by: XIAO Tong (email: xiaotong@mail.neu.edu.cn) 2018-05-08
 */

#include <math.h>
#include "SetData.h"
#include "SetData.cuh"
#include "../../XUtility.h"
#include "../movement/CopyValues.h"

#if !defined( WIN32 ) && !defined( _WIN32 )
    #include "sys/time.h"
    #include "time.h"
    #include "iconv.h"
#else
    #include "time.h"
    #include "windows.h"
    #include "process.h"
#endif

namespace nts{ // namespace nts(NiuTrans.Tensor)

/*
Fills the input Tensor or Variable with values according to the method described in 
"Understanding the difficulty of training deep feedforward neural networks" - Glorot, X. & Bengio, Y. (2010), 
using a uniform distribution. The resulting tensor will have values sampled from :math:`U(-a, a)` 
where :math:`a = gain \times \sqrt{2 / (fan\_in + fan\_out)} \times \sqrt{3}`. Also known as Glorot initialisation.

>> tensor - the tensor whose data array would be initialized
>> gain - an optional scaling factor
*/
void _SetDataFanInOut(XTensor * tensor, DTYPE gain)
{
    CheckNTErrors(tensor->dataType == X_FLOAT, "the tensor must be in X_FLOAT!");
    CheckNTErrors(tensor->order >= 2, "the tensor dimension must be no less than 2!");

    int fanIn = 1;
    int fanOut = 1;

    int order = tensor->order;
    if (order == 2) {
        fanIn = tensor->dimSize[1];
        fanOut = tensor->dimSize[0];
    }
    else {
        int numInputFmaps = tensor->dimSize[1];
        int numOutputFmaps = tensor->dimSize[0];
        int receptiveFieldSize = 0;
        for (int i = 2; i < order; i++)
            receptiveFieldSize += tensor->dimSize[i];
        fanIn = numInputFmaps * receptiveFieldSize;
        fanOut = numOutputFmaps * receptiveFieldSize;
    }

    DTYPE std = gain * (float)sqrt(2.0 / (fanIn + fanOut));
    DTYPE a = (DTYPE)sqrt(3.0F) * std;
    tensor->SetDataRand(-a, a);
    //_SetDataRand(tensor, -finfout, finfout);
}

/* 
generate data items with a fixed value p 
>> tensor - the tensor whose data array would be initialized
>> p - pointer to the number for initializing the tensor
*/
void _SetDataFixed(XTensor * tensor, void * valuePointer)
{
    int num = tensor->unitNum;

    if(tensor->dataType == X_INT){
        int p = *(int*)valuePointer;
        if(tensor->devID < 0){
            int * d = (int*)tensor->data;
            if(num % 4 == 0){
                for(int i = 0; i < num; i += 4){
                    d[i] = p;
                    d[i + 1] = p;
                    d[i + 2] = p;
                    d[i + 3] = p;
                }
            }
            else{
                for(int i = 0; i < num; i++)
                    d[i] = p;
            }
        }
        else{
#ifdef USE_CUDA
            _CudaSetDataFixedInt(tensor, p);
#endif
        }
    }
    else if(tensor->dataType == X_FLOAT){
        float p = *(float*)valuePointer;
        if(tensor->devID < 0){
            float * d = (float*)tensor->data;
            if(num % 4 == 0){
                for(int i = 0; i < num; i += 4){
                    d[i] = p;
                    d[i + 1] = p;
                    d[i + 2] = p;
                    d[i + 3] = p;
                }
            }
            else{
                for(int i = 0; i < num; i++)
                    d[i] = p;
            }
        }
        else{
#ifdef USE_CUDA
            _CudaSetDataFixedFloat(tensor, p);
#endif
        }
    }
    else if(tensor->dataType == X_DOUBLE){
        double p = *(double*)valuePointer;
        if(tensor->devID < 0){
            double * d = (double*)tensor->data;
            if(num % 4 == 0){
                for(int i = 0; i < num; i += 4){
                    d[i] = p;
                    d[i + 1] = p;
                    d[i + 2] = p;
                    d[i + 3] = p;
                }
            }
            else{
                for(int i = 0; i < num; i++)
                    d[i] = p;
            }
        }
        else{
#ifdef USE_CUDA
            _CudaSetDataFixedDouble(tensor, p);
#endif
        }
    }
    else{
        ShowNTErrors("TODO");
    }
}

/* 
generate data items with a fixed value p (in default type) 
>> tensor - the tensor whose data array would be initialized
>> p - number in default type
*/
void SetDataFixed(XTensor &tensor, DTYPE p)
{
    _SetDataFixed(&tensor, &p);
}
    
/*
generate data items with a fixed value p (in integer)
>> tensor - the tensor whose data array would be initialized
>> p - an integer
*/
void SetDataFixedInt(XTensor &tensor, int p)
{
    CheckNTErrors(tensor.dataType == X_INT, "An integer tensor is required!");
    _SetDataFixed(&tensor, &p);
}

/* 
generate data items with a fixed value p (in integer) 
>> tensor - the tensor whose data array would be initialized
>> p - an int-valued number
*/
void _SetDataFixedInt(XTensor * tensor, int p)
{
    CheckNTErrors(tensor->dataType == X_INT, "the tensor must be in X_INT!");

    if(p == 0)
        tensor->SetZeroAll();
    else
        _SetDataFixed(tensor, &p);
}

/*
generate data items with a fixed value p (in float) 
>> tensor - the tensor whose data array would be initialized
>> p - a float-valued number
*/
void _SetDataFixedFloat(XTensor * tensor, float p)
{
    CheckNTErrors(tensor->dataType == X_FLOAT, "the tensor must be in X_FLOAT!");

    if(p == 0)
        tensor->SetZeroAll();
    else
        _SetDataFixed(tensor, &p);
}

/* 
generate data items with a fixed value p (in double) 
>> tensor - the tensor whose data array would be initialized
>> p - a double-valued number
*/
void _SetDataFixedDouble(XTensor * tensor, double p)
{
    CheckNTErrors(tensor->dataType == X_DOUBLE, "the tensor must be in X_DOUBLE!");

    if(p == 0)
        tensor->SetZeroAll();
    else
        _SetDataFixed(tensor, &p);
}

/* 
generate data items with a fixed value p only if 
the condition entry is non-zero 
>> tensor - the tensor whose data array would be initialized
>> condition - the condition tensor whose entries would be checked
               for set the corresponding entries in "tensor"
>> p - a given value
*/
void _SetDataFixedCond(XTensor * tensor, XTensor * condition, DTYPE p)
{
    int num = tensor->unitNum;

    CheckNTErrors(num == condition->unitNum, "Wrong size of the condition tensor!");
    CheckNTErrors(condition->unitSize == sizeof(float), "TODO!");

    if(tensor->dataType == DEFAULT_DTYPE){
        if(tensor->devID < 0){
            DTYPE * data = (DTYPE*)tensor->data;
            DTYPE * cond = (DTYPE*)condition->data;
            for(int i = 0; i < num; i++){
                if(cond[i] != 0)
                    data[i] = p;
            }
        }
        else{
#ifdef USE_CUDA
            _CudaSetDataFixedCondFloat(tensor, condition, p);
#else
            ShowNTErrors("Please specify USE_CUDA and recompile the code");
#endif
        }
    }
    else{
        ShowNTErrors("the tensor should be in integer typed!");
    }
}

/* 
generate data items with a fixed value p only if 
the condition entry is non-zero 
>> tensor - the tensor whose data array would be initialized
>> condition - the condition tensor whose entries would be checked
               for set the corresponding entries in "tensor"
>> p - a given value
*/
void _SetDataFixedCondInt(XTensor * tensor, XTensor * condition, int p)
{
    int num = tensor->unitNum;

    CheckNTErrors(num == condition->unitNum, "Wrong size of the condition tensor!");
    CheckNTErrors(condition->unitSize == sizeof(float), "TODO!");

    if(tensor->dataType == DEFAULT_DTYPE){
        if(tensor->devID < 0){
            int * data = (int*)tensor->data;
            int * cond = (int*)condition->data;
            for(int i = 0; i < num; i++){
                if(cond[i] != 0)
                    data[i] = p;
            }
        }
        else{
#ifdef USE_CUDA
            _CudaSetDataFixedCondInt(tensor, condition, p);
#else
            ShowNTErrors("Please specify USE_CUDA and recompile the code");
#endif
        }
    }
    else{
        ShowNTErrors("TODO!");
    }
}

/* 
set data items along with a given dimension (and keep the remaining items unchanged) 
>> tensor - the tensor whose data array would be initialized
>> beg - the beginning position
>> len - length along with the given dimension
>> dim - the dimension along which we set the data
e.g., given a 3 * 3 tensor 
      1 2 3
      4 5 6
      7 8 9
      when beg = 1, len = 1, dim = 0 and p = 0, we have
      1 2 3
      0 0 0
      7 8 9
      i.e., we set all entries of row 1 to 0
*/
void _SetDataDim(XTensor * tensor, int beg, int len, int dim, DTYPE p)
{
    int n = tensor->order;

    CheckNTErrors(tensor->dataType == DEFAULT_DTYPE, "TODO!");
    CheckNTErrors(dim < n && dim >= 0, "Illegal dimension!");
    CheckNTErrors(beg >= 0 && beg < tensor->GetDim(dim), "Illegal beginning position!");
    CheckNTErrors(beg + len >= 0 && beg + len < tensor->GetDim(dim), "Illegal length!");
    
    if(tensor->devID < 0){
        int stride = 1;
        int blockSize = 1;
        int blockNum  = 1;
        for(int i = n - 1; i > dim; i--){
            stride *= tensor->GetDim(i);
        }
        blockSize = stride * tensor->GetDim(dim);
        blockNum = tensor->unitNum / blockSize;

        int l = len * stride;

        for(int i = 0; i < blockNum; i++){
            DTYPE * d = (DTYPE*)tensor->data + blockSize * i + beg * stride;    
            for(int j = 0; j < l; j++)
                d[j] = p;
        }
    }
    else{
#ifdef USE_CUDA
        _CudaSetDataDim(tensor, beg, len, dim, p);
#endif
    }
}

/* 
modify data items along with a given index and dimension (and keep the remaining items unchanged) 
>> source - the tensor whose data array would be modified
>> modify - the tensor whose data array would be used to modify the source tensor
>> dim - the dimension along which we modify the tensor
>> index - index of the given dimension
e.g., given a source tensor (3, 3)
      1 2 3
      4 5 6
      7 8 9
      given a modified tensor (3)
      1 2 3
      when dim = 0, index = 1, we have
      1 2 3
      1 2 3
      7 8 9
      i.e., we set entries of row 1 to {1, 2, 3}
*/
void _SetDataIndexed(XTensor * source, XTensor * modify, int dim, int index)
{
    int order = source->order;
    int size = source->GetDim(dim);

    CheckNTErrors(source->dataType == DEFAULT_DTYPE, "TODO!");
    CheckNTErrors(dim >= 0 && dim < order, "Illegal dimension!");
    CheckNTErrors(index >= 0 && index < size, "Illegal index!");
    
    for(int i = 0; i < order - 1; i++){
        if(i < dim){
            CheckNTErrors(modify->GetDim(i) == source->GetDim(i), "Illegal dimension!");
        }
        else if(i >= dim){
            CheckNTErrors(modify->GetDim(i) == source->GetDim(i+1), "Illegal dimension!");
        }
    }

    if(source->devID < 0 && modify->devID < 0){
        int stride = 1;
        int blockSize = 1;
        int blockNum  = 1;

        for(int i = order - 1; i > dim; i--){
            stride *= source->GetDim(i);
        }

        blockSize = stride * source->GetDim(dim);
        blockNum = source->unitNum / blockSize;

        for(int i = 0; i < blockNum; i++){
            DTYPE * d = (DTYPE*)source->data + blockSize * i + index * stride;
            DTYPE * p = (DTYPE*)modify->data + stride * i;
            for(int j = 0; j < stride; j++)
                d[j] = p[j];
        }
    }
    else if(source->devID >= 0 && modify->devID >= 0) {
#ifdef USE_CUDA
        _CudaSetDataIndexed(source, modify, dim, index);
#else
        ShowNTErrors("Please specify USE_CUDA and recompile the code!");
#endif
    }
    else{
        ShowNTErrors("TODO!");
    }
}

/* 
generate data as lower triangular matrics for last two dimensions 
>> tensor - the tensor whose data to be set
>> p - the value for each entry of the lower triangular matrics
>> shift - the offset from diagonal
e.g., for a 3 * 3 tensor, 
      when p = 1 ans shift = 0, we have
      1 0 0
      1 1 0
      1 1 1
      when p = 2 and shift = -1, we have
      0 0 0
      2 0 0
      2 2 0
*/
void _SetDataLowTri(XTensor * tensor, DTYPE p, int shift)
{
    int n = tensor->order;

    CheckNTErrors(tensor->dataType == DEFAULT_DTYPE, "TODO!");
    CheckNTErrors(n >= 2, "The tensor must have a order no less than 2!");
    CheckNTErrors(tensor->GetDim(n - 1) == tensor->GetDim(n - 2), 
                 "The last two dimensions must be of the same size!");

    if(tensor->devID < 0){
        int l = tensor->GetDim(-1);
        int blockNum = 1;
        int blockSize = l * l;
        for(int i = 0; i < n - 2; i++)
            blockNum *= tensor->GetDim(i);

        for(int i = 0; i < blockNum; i++){
            DTYPE * d = (DTYPE*)tensor->data + i * blockSize;
            for(int row = 0; row < l; row++){
                for(int col = 0; col <= row + shift; col++){
                    d[row * l + col] = p;
                }
                for(int col = MAX(0, row + shift + 1); col < l; col++){
                    d[row * l + col] = 0;
                }
            }
        }
    }
    else{
#ifdef USE_CUDA
        _CudaSetDataLowTri(tensor, p, shift);
#endif
    }
}

/* generate data items with a uniform distribution in [0, 1] */
void _SetDataRand(XTensor * tensor, int rNum, int cNum)
{
    if (tensor == NULL || tensor->isInit == false || tensor->order !=2 ) {
        InitTensor2DV2(tensor, rNum, cNum);
    }

    _SetDataRand(tensor, 0.0F, 1.0F);
}

/*
generate data items with a uniform distribution in [lower, upper]
>> tensor - the tensor whose data array would be initialized
>> lower - lower value of the range
>> upper - upper value of the range
*/
void _SetDataRand(XTensor * tensor, DTYPE lower, DTYPE upper)
{
    CheckNTErrors(upper > lower, "the high value must be greater than low value!");

    if(tensor == NULL)
        return;
    
    /* CPU code */
    if(tensor->devID < 0){
        DTYPE variance = upper - lower;
        
        if(tensor->dataType == X_FLOAT){
            float * d = (float*)tensor->data;
            for(int i = 0; i < tensor->unitNum; i++){
                d[i] = variance * ((float)rand()/RAND_MAX) + lower;
            }
        }
        else if(tensor->dataType == X_DOUBLE){
            double * d = (double*)tensor->data;
            for(int i = 0; i < tensor->unitNum; i++){
                d[i] = variance * ((double)rand()/RAND_MAX) + lower;
            }
        }
        else{
            ShowNTErrors("TODO");
        }
    }
    /* 
    GPU code
    The trick here is that initialize the data on a temperary tensor on CPU.
    The CPU data is then copied to GPU.
    TODO: generate data points on GPUs straightforwardly.
    */
    else{
#ifdef USE_CUDA
        _CudaSetDataRand(tensor, lower, upper);
#endif
        //XTensor * t2 = NewTensorV2(tensor->order, tensor->dimSize, tensor->dataType, tensor->denseRatio, -1);
        //_SetDataRand(t2, low, high);
        //_CopyValues(t2, tensor);
        //delete t2;
    }
}

/* generate data items with a range by start, end and the step
>> tensor - the tensor whose data array would be initialized
>> start - the begin of the array
>> end - the end of the array (not included self)
>> step - the step of two items
*/
void _SetDataRange(XTensor * tensor, DTYPE lower, DTYPE upper, DTYPE step)
{
    CheckNTErrors((tensor->order == 1), "Tensor must be 1 dimension!");

    /* compute the true length according to the (start, end, step) */
    DTYPE size = fabs(upper - lower);
    int num = ceil(size / fabs(step));
    CheckNTErrors((tensor->unitNum == num), "Unit number of the tensor is not matched.");

    /* init a integer array to store the sequence */
    void * data = NULL;
    if (tensor->dataType == X_INT) {
        data = new int[num];
        for (int i = 0; i < num; i++)
            *((int*)data + i) = lower + i * step;
    }
    else if (tensor->dataType == X_FLOAT) {
        data = new float[num];
        for (int i = 0; i < num; i++)
            *((float*)data + i) = lower + i * step;
    }
    else {
        ShowNTErrors("TODO!");
    }

    /* set the data from the array */
    tensor->SetData(data, num);

    delete[] data;
}

/* 
generate data items with a uniform distribution in [lower, upper] and set
the item to a pre-defined value if the item >= p, set the item to 0 otherwise
>> tensor - the tensor whose data array would be initialized
>> lower - lower value of the range
>> upper - upper value of the range
>> p - the threshold
>> value - the value we intend to assign to the item
*/
void _SetDataRandP(XTensor * tensor, DTYPE lower, DTYPE upper, DTYPE p, DTYPE value)
{
    CheckNTErrors(tensor->dataType == DEFAULT_DTYPE, "TODO");

    if (tensor->devID < 0) {
        _SetDataRand(tensor, lower, upper);

        DTYPE * data = (DTYPE*)tensor->data;
        for (int i = 0; i < tensor->unitNum; i++) {
            if (data[i] >= p)
                data[i] = value;
            else
                data[i] = 0;
        }
    }
    else {
#ifdef USE_CUDA
        _CudaSetDataRandP(tensor, lower, upper, p, value);
#else
        ShowNTErrors("Please recompile the code by specifying USE_CUDA");
#endif // USE_CUDA
    }
}
    
/*
generate data items with a normal distribution with specified mean and standard deviation 
>> tensor - the tensor that keeps the data
>> mean - mean or expectation of the distribution
>> standardDeviation - standard deviation of the distribution
*/
void _SetDataRandN(XTensor * tensor, DTYPE mean, DTYPE standardDeviation)
{
    // TODO: rewrite it and add cuda code!!!!!!!
    tensor->SetDataRandn(mean, standardDeviation);
}

/* 
set the data with an array of offsets 
>> tensor - the tensor that keeps the data
>> offsets - offset for each data item
>> num - number of the data items
>> value - value of the data items
*/
void _SetDataWithOffset(XTensor * tensor, MTYPE * offsets, DTYPE value, MTYPE num)
{
    CheckNTErrors(tensor->dataType == X_FLOAT, "Data type is incorrect!");

    if (tensor->devID < 0) {
        DTYPE * d = (DTYPE*)tensor->data;
        for (int i = 0; i < num; i++) {
            d[offsets[i]] = value;
        }
    }
    else {
#ifdef USE_CUDA
        XMem * mem = tensor->mem;
        MTYPE size = num * sizeof(MTYPE);
        MTYPE * offsetsCuda = mem != NULL ? (MTYPE*)mem->AllocBuf(mem->devID, size) : (MTYPE*)XMemAlloc(tensor->devID, size);
        XMemCopy(offsetsCuda, tensor->devID, offsets, -1, num * sizeof(MTYPE));

        _CudaSetDataWithOffset(tensor, offsetsCuda, value, num);
        
        if (mem != NULL)
            mem->ReleaseBuf(mem->devID, size);
        else
            XMemFree(tensor->devID, offsetsCuda);
#else
        ShowNTErrors("Please recompile the code with USE_CUDA");
#endif
    }
}

/* 
set the data with an array of values 
>> tensor - the tensor that keeps the data
>> offsets - offset for each data item
>> values - value for each data item
>> num - number of the data items
*/
void _SetDataWithOffsetAndValue(XTensor * tensor, MTYPE * offsets, void * values, MTYPE num)
{
    if (tensor->devID < 0) {
        for (int i = 0; i < num; i++) {
            if (tensor->dataType == X_INT)
                *((int *)tensor->data + offsets[i]) = *((int *)values + i);
            else if (tensor->dataType == X_FLOAT)
                *((float *)tensor->data + offsets[i]) = *((float *)values + i);
            else 
                ShowNTErrors("TO DO!!!");
        }
    }
    else {
#ifdef USE_CUDA
        if(tensor->devID >= 0) {
            _CudaSetDataWithOffsetAndValue(tensor, offsets, values, num);
            return;
        }
#else
        ShowNTErrors("Please recompile the code with USE_CUDA");
#endif
    }
}

} // namespace nts(NiuTrans.Tensor)

