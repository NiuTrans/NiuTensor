/*
 * NiuTrans.Tensor - an open-source tensor library
 * Copyright (C) 2018, Natural Language Processing Lab, Northeastern University.
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
generate data items according to the method
described in `Understanding the difficulty 
of training deep feedforward neural networks`
- Glorot, X. & Bengio, Y. (2010), using a normal 
distribution. The resulting tensor will have values sampled from
:math:`\mathcal{N}(0, \text{std}^2)` where

.. math::
\text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

Also known as Glorot initialization.
>> tensor - the tensor whose data array would be initialized
>> gain - an optional scaling factor
*/
void _SetDataXavierNormal(XTensor * tensor, DTYPE gain)
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

    DTYPE std = gain * (float)sqrt(2.0 / (float)(fanIn + fanOut));
    
    tensor->SetDataRandn(0, std);
}
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

    DTYPE std = gain * (float)sqrt(2.0 / (float)(fanIn + fanOut));
    DTYPE a = (DTYPE)sqrt(3.0F) * std;
    tensor->SetDataRand(-a, a);
    //_SetDataRand(tensor, -finfout, finfout);
}

/*
set a data array with a fixed value

>> d - pointer to the data array
>> v - the initial value
>> size - size of the array
*/
template<class T>
void ArraySetDataFixed(T * d, T v, int size)
{
    if (size % 4 == 0) {
        for (int i = 0; i < size; i += 4) {
            d[i] = v;
            d[i + 1] = v;
            d[i + 2] = v;
            d[i + 3] = v;
        }
    }
    else {
        for (int i = 0; i < size; i++)
            d[i] = v;
    }
}

/*
generate data items with a fixed value

>> tensor - the tensor for initialization
>> value - the initial value
*/
template<class T>
void _SetDataFixed(XTensor * tensor, T value)
{
    if (tensor->devID >= 0) {
#ifdef USE_CUDA
        _CudaSetDataFixed(tensor, value);
        return;
#else
        ShowNTErrors("Please specify USE_CUDA and recompile the code!");
#endif
    }

    int num = tensor->unitNum;

    if (tensor->dataType == X_INT)
        ArraySetDataFixed((int*)tensor->data, (int)value, num);
    else if (tensor->dataType == X_FLOAT)
        ArraySetDataFixed((float*)tensor->data, (float)value, num);
    else if (tensor->dataType == X_DOUBLE)
        ArraySetDataFixed((double*)tensor->data, (double)value, num);
    else
        ShowNTErrors("TODO! Unsupported datatype!")
}
template void _SetDataFixed<int>(XTensor*, int);
template void _SetDataFixed<float>(XTensor*, float);
template void _SetDataFixed<double>(XTensor*, double);

/*
generate data items with a fixed value p only if the condition entry is non-zero

>> d - pointer to the data array
>> c - pointer to the condition array
>> v - the initial value
>> size - size of the array
*/
template<class T>
void ArraySetDataFixedCond(T* d, T* c, T v, int size)
{
    for (int i = 0; i < size; i++) {
        if (c[i] != 0)
            d[i] = v;
    }
}

/* 
generate data items with a fixed value p only if the condition entry is non-zero 

>> tensor - the tensor whose data array would be initialized
>> condition - the condition tensor whose entries would be checked
               for set the corresponding entries in "tensor"
>> value - a given value
*/
template<class T>
void _SetDataFixedCond(XTensor * tensor, XTensor * condition, T value)
{
    CheckDev(tensor->devID, condition->devID);
    CheckDataType(tensor->dataType, condition->dataType);

    if (tensor->devID >= 0) {
#ifdef USE_CUDA
        _CudaSetDataFixedCond(tensor, condition, value);
        return;
#else
        ShowNTErrors("Please specify USE_CUDA and recompile the code!");
#endif
    }

    int num = tensor->unitNum;

    if (tensor->dataType == X_INT)
        ArraySetDataFixedCond((int*)tensor->data, (int*)condition->data, (int)value, num);
    else if (tensor->dataType == X_FLOAT)
        ArraySetDataFixedCond((float*)tensor->data, (float*)condition->data, (float)value, num);
    else if (tensor->dataType == X_DOUBLE)
        ArraySetDataFixedCond((double*)tensor->data, (double*)condition->data, (double)value, num);
    else
        ShowNTErrors("TODO! Unsupported datatype!")
}
template void _SetDataFixedCond<int>(XTensor*, XTensor*, int);
template void _SetDataFixedCond<float>(XTensor*, XTensor*, float);
template void _SetDataFixedCond<double>(XTensor*, XTensor*, double);

/* 
set data items along with a given dimension (and keep the remaining items unchanged) 

>> tensor - the tensor for initialization
>> beg - the beginning position
>> len - length along with the given dimension
>> dim - the dimension along which we set the data
   e.g., given a 3 * 3 tensor 
         1 2 3
         4 5 6
         7 8 9
         when beg = 1, len = 1, dim = 0 and value = 0, we have
         1 2 3
         0 0 0
         7 8 9
         i.e., we set all entries of row 1 to 0
>> value - the given value
*/
template<class T>
void _SetDataDim(XTensor * tensor, int beg, int len, int dim, T value)
{
    int order = tensor->order;
    int size = tensor->GetDim(dim);
    if (dim < 0)
        dim = order + dim; 

    CheckNTErrors(dim < order && dim >= 0, "Illegal dimension!");
    CheckNTErrors(beg >= 0 && beg < size, "Illegal beginning position!");
    CheckNTErrors(len >= 0 && beg + len <= size, "Illegal length!");

    if (tensor->devID >= 0) {
#ifdef USE_CUDA
        _CudaSetDataDim(tensor, beg, len, dim, (DTYPE)value);
        return;
#else
        ShowNTErrors("Please specify USE_CUDA and recompile the code!");
#endif
    }

    int stride = 1;
    int blockSize = 1;
    int blockNum  = 1;

    for (int i = order - 1; i > dim; i--)
        stride *= tensor->GetDim(i);
    blockSize = stride * size;
    blockNum = tensor->unitNum / blockSize;

    int initNum = len * stride;

    for(int i = 0; i < blockNum; i++) {
        if (tensor->dataType == X_INT) {
            int* d = (int*)tensor->data + blockSize * i + beg * stride;
            for (int j = 0; j < initNum; j++)
                d[j] = (int)value;
        }
        else if (tensor->dataType == X_FLOAT) {
            float* d = (float*)tensor->data + blockSize * i + beg * stride;
            for (int j = 0; j < initNum; j++)
                d[j] = (float)value;
        }
        else if (tensor->dataType == X_DOUBLE) {
            double* d = (double*)tensor->data + blockSize * i + beg * stride;
            for (int j = 0; j < initNum; j++)
                d[j] = (double)value;
        }
        else
            ShowNTErrors("TODO! Unsupported datatype!")
    }
}
template void _SetDataDim<int>(XTensor*, int, int, int, int);
template void _SetDataDim<float>(XTensor*, int, int, int, float);
template void _SetDataDim<double>(XTensor*, int, int, int, double);

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
void _SetDataIndexed(XTensor * tensor, XTensor * modify, int dim, int index)
{
    int order = tensor->order;
    int size = tensor->GetDim(dim);
    if (dim < 0)
        dim = order + dim;

    CheckDev(tensor->devID, modify->devID);
    CheckNTErrors(dim >= 0 && dim < order, "Illegal dimension!");
    CheckNTErrors(index >= 0 && index < size, "Illegal index!");
    
    for(int i = 0; i < order - 1; i++) {
        if(i < dim) {
            CheckNTErrors(modify->GetDim(i) == tensor->GetDim(i), "Illegal dimension!");
        }
        else if(i >= dim) {
            CheckNTErrors(modify->GetDim(i) == tensor->GetDim(i+1), "Illegal dimension!");
        }
    }

    if (tensor->devID >= 0) {
#ifdef USE_CUDA
        _CudaSetDataIndexed(tensor, modify, dim, index);
        return;
#else
        ShowNTErrors("Please specify USE_CUDA and recompile the code!");
#endif
    }

    if(tensor->devID < 0) {
        int stride = 1;
        int blockSize = 1;
        int blockNum  = 1;

        for (int i = order - 1; i > dim; i--) {
            stride *= tensor->GetDim(i);
        }

        blockSize = stride * tensor->GetDim(dim);
        blockNum = tensor->unitNum / blockSize;

        for (int i = 0; i < blockNum; i++) {
            DTYPE * d = (DTYPE*)tensor->data + blockSize * i + index * stride;
            DTYPE * p = (DTYPE*)modify->data + stride * i;
            for(int j = 0; j < stride; j++)
                d[j] = p[j];
        }
    }
}

/* 
generate data as lower triangular matrics for last two dimensions 

>> tensor - the tensor whose data to be set
>> value - the value for each entry of the lower triangular matrics
>> shift - the offset from diagonal

   e.g., for a 3 * 3 tensor, 
         when value = 1 ans shift = 0, we have
         1 0 0
         1 1 0
         1 1 1
         when value = 2 and shift = -1, we have
         0 0 0
         2 0 0
         2 2 0
*/
void _SetDataLowTri(XTensor * tensor, DTYPE value, int shift)
{
    int n = tensor->order;

    CheckNTErrors(n >= 2, "The tensor must have a order no less than 2!");
    CheckNTErrors(tensor->GetDim(n - 1) == tensor->GetDim(n - 2), 
                 "The last two dimensions must be of the same size!");

    tensor->SetZeroAll();
    if (tensor->devID >= 0) {
#ifdef USE_CUDA
        _CudaSetDataLowTri(tensor, value, shift);
        return;
#else
        ShowNTErrors("Please specify USE_CUDA and recompile the code!");
#endif
    }

    int size = tensor->GetDim(-1);
    int blockSize = size * size;
    int blockNum = tensor->unitNum / blockSize;

    for (int i = 0; i < blockNum; i++) {
        for (int row = 0; row < size; row++) {
            if (tensor->dataType == X_INT) {
                int * d = (int*)tensor->data + i * blockSize;
                for (int col = 0; col <= row + shift; col++) {
                    d[row * size + col] = (int)value;
                }
                /*for (int col = MAX(0, row + shift + 1); col < size; col++) {
                    d[row * size + col] = 0;
                }*/
            }
            else if (tensor->dataType == X_FLOAT) {
                float * d = (float*)tensor->data + i * blockSize;
                for (int col = 0; col <= row + shift; col++) {
                    d[row * size + col] = (float)value;
                }
                /*for (int col = MAX(0, row + shift + 1); col < size; col++) {
                    d[row * size + col] = 0;
                }*/
            }
            else if (tensor->dataType == X_DOUBLE) {
                double * d = (double*)tensor->data + i * blockSize;
                for (int col = 0; col <= row + shift; col++) {
                    d[row * size + col] = (double)value;
                }
                /*for (int col = MAX(0, row + shift + 1); col < size; col++) {
                    d[row * size + col] = 0;
                }*/
            }
            else 
                ShowNTErrors("TODO! Unsupported datatype!")
        }
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
    CheckNTErrors(upper >= lower, "the high value must be greater than low value!");

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
        else if (tensor->dataType == X_FLOAT16) {
            unsigned short* d = (unsigned short*)tensor->data;
            for (int i = 0; i < tensor->unitNum; i++) {
                d[i] = variance * ((unsigned short)rand() / RAND_MAX) + lower;
            }
        }
        else if(tensor->dataType == X_DOUBLE){
            double * d = (double*)tensor->data;
            for(int i = 0; i < tensor->unitNum; i++){
                d[i] = variance * ((double)rand()/RAND_MAX) + lower;
            }
        }
        else{
            ShowNTErrors("TODO! Unsupported datatype!")
        }
    }
    else{
#ifdef USE_CUDA
        /*
        GPU code
        The trick here is that initialize the data on a temperary tensor on CPU.
        The CPU data is then copied to GPU.
        TODO: generate data points on GPUs straightforwardly.
        */
        //_CudaSetDataRand(tensor, lower, upper);
        int num = tensor->unitNum;
        DTYPE variance = upper - lower;

        void * d = NULL;
        if (tensor->dataType == X_FLOAT) {
            d = new float[num];
            for (int i = 0; i < num; i++) 
                *((float*)d + i) = lower + variance * (float)rand() / RAND_MAX;
        }
        else if (tensor->dataType == X_DOUBLE) {
            d = new double[num];
            for (int i = 0; i < num; i++) 
                *((double*)d + i) = (double)lower + variance * rand() / RAND_MAX;
        }
        else {
            ShowNTErrors("Data type must be X_FLOAT or X_Double!");
        }

        tensor->SetData(d, num);

        if (tensor->dataType == X_FLOAT) {
            delete[](float*)d;
        }
        else {
            delete[](double*)d;
        }
#endif
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
    DTYPE size = (DTYPE)fabs(upper - lower);
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
        ShowNTErrors("TODO! Unsupported datatype!")
    }

    /* set the data from the array */
    tensor->SetData(data, num);

    if (tensor->dataType == X_INT) {
        delete[] (int*)data;
    }
    else if (tensor->dataType == X_FLOAT) {
        delete[] (float*)data;
    }
    else {
        ShowNTErrors("TODO! Unsupported datatype!")
    }
}

/* 
generate data items with a uniform distribution in [lower, upper] and 
set the item to a pre-defined value if the item >= p, 
set the item to 0 otherwise

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

/* a gauss distribution (Box-Muller method) */
double GaussRand(DTYPE mean, DTYPE standardDeviation)
{
    static double u, v;
    static int phase = 0;
    double z;
    double pi = 3.141592654;

    if (phase == 0) {
        u = (rand() + 1.0) / (RAND_MAX + 1.0);
        v = (rand() + 1.0) / (RAND_MAX + 1.0);
        z = sqrt(-2.0 * log(u)) * sin(2.0 * pi * v);
    }
    else {
        z = sqrt(-2.0 * log(u)) * cos(2.0 * pi * v);
    }

    phase = 1 - phase;
    return mean + (z * standardDeviation);
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
    int num = tensor->unitNum;

    void * d = NULL;
    if (tensor->dataType == X_FLOAT) {
        d = new float[num];
        for (int i = 0; i < num; i++)
            *((float*)d + i) = (float)GaussRand(mean, standardDeviation);
    }
    else if (tensor->dataType == X_DOUBLE) {
        d = new double[num];
        for (int i = 0; i < num; i++)
            *((double*)d + i) = GaussRand(mean, standardDeviation);
    }
    else {
        ShowNTErrors("TODO! Unsupported datatype!")
    }

    tensor->SetData(d, num);

    if (tensor->dataType == X_FLOAT) {
        delete[](float*)d;
    }
    else {
        delete[](double*)d;
    }
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

