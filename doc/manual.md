# NiuTrans.Tensor张量计算库

## NiuTrans.Tensor
NiuTrans.Tensor是小牛开源项目所开发的一个工具包，提供了完整的张量定义及计算功能，可以被用于深度学习相关研究及工业系统的开发。NiuTrans.Tensor具有以下特点：

* 简单小巧，易于修改
* c语言编写，代码高度优化
* 同时支持CPU和GPU设备
* 丰富的张量计算接口
* 支持C/C++、Python等调用方式

## 安装NiuTrans.Tensor

在开始创建您的项目并使用NiuTrans.Tensor工具包时，需要注意的是：

* 所创建项目如在CPU上运行，我们的系统支持高性能的数学运算库，推荐安装[Intel® MKL](https://software.intel.com/en-us/mkl)或[OpenBLAS](http://www.openblas.net/)。
* 所创建项目如需在GPU上运行，需安装 [NVIDIA®CUDA®Toolkit](https://developer.nvidia.com/cuda-downloads)，CUDA版本需求为9.0及以上，CUDA工具为创建高性能GPU加速应用程序提供了开发环境。

小牛开源项目所开发的NiuTrans.Tensor工具包采用源程序编译方法，在Windows和Linux环境下的安装方法如下所示。

### Windows

若在Windows上使用NiuTrans.Tensor工具包：

* 首先需要将NiuTrans.Tensor代码包含在所创建的项目中
* 在所创建项目中需要引用XTensor.h、core里的CHeader.h和function里的FHeader.h这三个头文件：
    * 通过XTensor.h可以获取我们需要操作的XTensor类
    * 通过core里的CHeader.h可以对Tensor进行一些张量运算
    * 通过function里的FHeader.h可以调用一些激活函数
* 在所创建项目中使用命名空间nts

此外，一些必须的环境配置方法请参考 [NiuTrans.Tensor环境配置](http://47.105.50.196/NiuTrans/NiuTrans.Tensor/blob/linye/doc/Configuration.md)。

### Linux

若在Linux上使用NiuTrans.Tensor工具包，直接执行make.sh即可在同级目录下生成tensorCPU和tensorGPU，分别对应于NiuTrans.Tensor的CPU以及GPU的可执行文件。以前馈神经网络语言模型为例，输入以下命令即可在GPU上执行提供的测试用例：
>./tensorGPU -test

## 什么是张量

在计算机科学中，张量（Tensor）通常被定义为\\(n\\)维空间中的一种量，它具有\\(n\\)个分量，这种张量本质上是一个多维数组（ multidimensional array）。张量的阶或秩是这个多维数组的维度，或者简单理解为索引张量里的每个元素所需要的索引个数。通常来说，0阶张量被定义为标量（Scalar），1阶张量被定义为向量（vector），而2阶张量被定义为矩阵（matrix）。比如，在一个三维空间中，1阶张量就是空间中点所表示的向量\\((x,y,z)\\)，其中\\(x\\)、\\(y\\)、\\(z\\)分别表示这个点在三个轴上的坐标。

张量是一种高效的数学建模工具，它可以将复杂的问题通过统一、简洁的方式进行表达。比如，姜英俊同学做饭需要2斤牛肉、5斤土豆，市场上牛肉每斤32元、土豆每斤2元，那么购买这些食物总共花费\\(2 \times 32 + 5 \times 2 = 74\\)元。如果用张量来描述，我们可以用一个1阶张量\\(a=(2,5)\\)表示所需不同食物的重量。然后用另一个1阶张量\\(b=(32,2)\\)表示不同食物的价格。最后，我们用一个0阶张量\\(c\\)表示购买这些食物的总价，计算如下

$$
\begin{aligned}
  c & = a \times b^T \\\\
    & = \left(\begin{matrix}2 & 5\end{matrix}\right) \times \left(\begin{matrix}32 \\\\ 2\end{matrix}\right) \\\\
    & = 2 \times 32 + 5 \times 2 \\\\
    & = 74
\end{aligned}
$$

其中\\(b^T\\)表示行向量\\(b\\)的转置 - 列向量，\\(\times\\)表示向量的乘法。第二天，姜英俊同学换了一个市场，这里牛肉每斤35元、土豆每斤1元。如果要知道在两个市场分别购物的总价，可以把\\(b\\)重新定义为一个2阶张量\\(\left(\begin{matrix}32 & 2 \\\\ 35 & 1\end{matrix}\right)\\)，总价\\(c\\)定义为一个2阶张量。同样有

$$
\begin{aligned}
  c & = a \times b^T \\\\
    & = \left(\begin{matrix}2 & 5\end{matrix}\right) \times \left(\begin{matrix}32 & 35 \\\\ 2 & 1\end{matrix}\right) \\\\
    & = \left(\begin{matrix}74 & 75\end{matrix}\right)
\end{aligned}
$$

即，在两个市场分别花费74元和75元。可以看出，利用张量可以对多样、复杂的问题进行建模，比如，可以进一步扩展上述问题中\\(a\\)、\\(b\\)、\\(c\\)的定义，把它们定义成更高阶的张量，处理不同时间、不同市场、不同菜谱的情况，但是不论情况如何变化，都可以用同一个公式\\(c = a \times b^T\\)来描述问题。

许多现实世界的问题都可以被描述为张量表达式（expression），也就是把张量的组合、计算描述为算数表达式。这种建模方式也构成了现代神经网络模型及深度学习方法的基础。在许多机器学习工具中，张量计算已经成为了神经网络前向、反向传播等过程的基本单元，应用十分广泛。

## 如何定义张量

如果你是一名C/C++或者Python的使用者，那么在程序中使用NiuTrans.Tensor定义张量将非常简单。首先，下载NiuTrans.Tensor的工具包，并解压到任意目录，比如~/NiuTrans.Tensor目录。我们会在NiuTrans.Tensor这个目录中找到source子目录，它是存放源代码的目录。对于source子目录的结构，信息如下：

* ~/NiuTrans.Tensor/source/tensor/XTensor.h - 定义了张量结构XTensor，以及构建和销毁XTensor的接口
* ~/NiuTrans.Tensor/source/tensor/core - 存放张量计算的函数声明及函数体实现的源文件
    * arithmetic - 存放有关算术运算的源文件
    * getandset - 存放有关算术存取的源文件
    * math - 存放有关数学运算的源文件
    * movement - 存放有关数据移动的源文件
    * reduce - 存放有关规约操作的源文件
    * shape - 存放有关形状转换的源文件
    * sort - 存放有关排序操作的源文件
* ~/NiuTrans.Tensor/source/tensor/function - 存放各种激活函数的源文件
* ~/NiuTrans.Tensor/source/tensor/test - 存放单元测试的源文件
* ~/NiuTrans.Tensor/source/tensor/*.h(cpp) - 与张量定义不相关，后文介绍 :)

以C/C++为例，仅需要在源程序中引用XTensor.h头文件就可以完成张量的定义。下面是一个简单的示例程序sample.cpp
```
#inlucde "XTensor.h"          // 引用XTensor定义的头文件

using namepsace nt;           // 使用XTensor所在的命名空间nt

int main(int argc, const char ** argv)
{
    // 声明一个变量tensor，它的类型是XTensor
    XTensor tensor;                         
    
    // 初始化这个变量为50列*100行的矩阵(2阶张量)      
    InitTensor2D(&tensor, 50, 100, X_FLOAT);
    
    // 之后可以使用张量tensor了
    
    return 0;
}
```

下一步，编译以上源程序，这个过程需要指定XTensor.h头文件所在目录。比如，使用g++编译sample.cpp

```
g++ sample.cpp -I~/NiuTrans.Tensor/source/tensor -o sample
```

在sample.cpp中使用了XTensor，它是NiuTrans.Tensor里的一个类，这个类定义了张量所需的数据结构。我们可以使用这个类完成对张量的计算、拷贝等各种操作。XTensor类型的变量被声明后，这个变量需要被初始化，或者说被真正指定为一个张量，比如，指定张量各个维度的大小、张量中每个单元的数据类型、给张量分配内存空间等。InitTensor2D()就是一个张量初始化函数，它把张量初始化为一个矩阵，有四个参数：指向被初始化的张量的指针，矩阵的列数，矩阵的行数，数据单元的类型。这里X_FLOAT，是NiuTrans.Tensor自定义的枚举类型，它表示单精度浮点数。我们也可以使用X_INT或者X_DOUBLE，将数据类型指定为32bit整数或者双精度浮点数。

NiuTrans.Tensor也提供了其它方式定义张量。比如可以直接调用一个函数完成张量的创建，而且可以显性释放张量。下面是一段示例代码（sample2.cpp）：

```
#inlucde "XTensor.h"         // 引用XTensor定义的头文件

using namepsace nt;          // 使用XTensor所在的命名空间nt

int main(int argc, const char ** argv)
{
    // 构建一个单精度浮点类型张量，它是一个50列*100行的矩阵
    XTensor * tensor = NewTensor2D(50, 100, X_FLOAT);  
    
    // 之后可以使用张量tensor了
    
    // 释放这个张量
    DelTensor(tensor);
    
    return 0;
}
```

sample2.cpp中使用的NewTensor2D和DelTensor是一组函数，前者生成张量并返回指向这个张量的指针，后者释放指针所指向张量的内容。这种方法比较适合C/C++风格的开发。

> 注意，在NiuTrans.Tensor中所有张量默认都是“稠密”张量，也就是张量中所有的单元都会被分配空间，而且这些空间是连续的。有些情况下，张量里的单元仅有少数为非零单元，对于这类张量，可以使用“稀疏"的表示方法，这样可以有效的节省存储空间。

如果要定义稀疏张量，需要在原有的参数基础上额外指定一个参数 - 稠密度。所谓稠密度是指非零单元的比例，他是介于0和1之间的一个实数，0表示所有单元全为零，1表示全为非零单元。默认所有张量的稠密度都是1。下面是不同类型张量的定义方法示例（sample3.cpp）


```
#inlucde "XTensor.h"         // 引用XTensor定义的头文件

using namepsace nt;          // 使用XTensor所在的命名空间nt

int main(int argc, const char ** argv)
{
    // 构建一个单精度浮点类型张量，它是一个50列*100行的矩阵
    // 这个张量是稠密的
    XTensor * tensor0 = NewTensor2D(50, 100, X_FLOAT);
    
    // 构建一个单精度浮点类型张量，它是一个50列*100行的矩阵
    // 这个张量是稠密的
    XTensor * tensor1 = NewTensor2D(50, 100, X_FLOAT, 1.0F);
    
    // 构建一个单精度浮点类型张量，它是一个50列*100行的矩阵
    // 这个张量是稀疏的，有10%的单元非零
    XTensor * tensor2 = NewTensor2D(50, 100, X_FLOAT, 0.1F);  
    
    // 之后可以使用张量tensor0，tensor1和tensor2了
    
    // 释放这些张量
    DelTensor(tensor0);
    DelTensor(tensor1);
    DelTensor(tensor2);
    
    return 0;
}
```

以下是关于张量定义的基础函数：

| 功能 | 函数| 参数 |
| - | - | - |
| 初始化张量 | void InitTensor(<br>XTensor * tensor, const int myOrder, <br> const int * myDimSize, <br> const TENSOR_DATA_TYPE myDataType = X_FLOAT, <br> const float myDenseRatio, <br>const int myDevID = -1, XMem * myMem = NULL) <br> <br> <br> | tensor - 指向被初始化张量的指针 <br> myOrder - 张量的维度 <br> myDimSize - 张量每一维的大小，索引0表示第一维 <br> myDenseRatio - 张量的稠密度，1表示稠密张量 <br> myDataType - 张量的数据类型 <br> myDevID - 张量所在的设备ID <br> myMem - 张量所使用的内存池 |
| 初始化稠密张量 | void InitTensor(<br>XTensor * tensor, const int myOrder, <br> const int * myDimSize, <br> const TENSOR_DATA_TYPE myDataType = X_FLOAT, <br>const int myDevID = -1, XMem * myMem = NULL) <br> <br> | tensor - 指向被初始化张量的指针 <br> myOrder - 张量的维度 <br> myDimSize - 张量每一维的大小，索引0表示第一维 <br> myDataType - 张量的数据类型 <br> myDevID - 张量所在的设备ID <br> myMem - 张量所使用的内存池 |
| 创建空张量 | XTensor * NewTensor() | N/A |
| 创建张量 | XTensor * NewTensor(<br>const int myOrder, <br> const int * myDimSize, const float myDenseRatio, <br> const TENSOR_DATA_TYPE myDataType = X_FLOAT, <br>const int myDevID = -1, XMem * myMem = NULL) <br> <br> | myOrder - 张量的维度 <br> myDimSize - 张量每一维的大小，索引0表示第一维 <br> myDenseRatio - 张量的稠密度，1表示稠密张量 <br> myDataType - 张量的数据类型 <br> myDevID - 张量所在的设备ID <br> myMem - 张量所使用的内存池 |
| 创建稠密张量 | XTensor * NewTensor(<br>const int myOrder, <br> const int * myDimSize, <br> const TENSOR_DATA_TYPE myDataType = X_FLOAT, <br>const int myDevID = -1, XMem * myMem = NULL) <br> | myOrder - 张量的维度 <br> myDimSize - 张量每一维的大小，索引0表示第一维 <br> myDataType - 张量的数据类型 <br> myDevID - 张量所在的设备ID <br> myMem - 张量所使用的内存池 |
| 销毁张量 | void DelTensor(const XTensor * tensor)   | tensor - 指向要被销毁的张量的指针 |

上述函数中需要说明的是

* 设备ID是指张量所申请的空间所在CPU或者GPU设备的编号，-1表示CPU
* XMem是NiuTrans.Tensor中的一个内存/显存池类，它负责内存（或显存）的统一管理。关于设备ID和XMem的进一步说明，请参见下一节内容。
* TENSOR_DATA_TYPE定义了张量的数据类型，包括：

| 类型 | 说明 |
| - | - |
| X_INT | 32bit整数 |
| X_FLOAT | 32bit浮点数 |
| X_DOUBLE | 64bit浮点数 |
| X_INT8 | 8bit整数（计划支持）|
| X_FLOAT16 | 16bit浮点数（计划支持） |

此外，NiuTrans.Tensor也提供了更多种类的张量初始化和创建方法：

| 功能 | 函数 | 参数 |
| - | - | - |
| 初始化为稠密向量 | void InitTensor1D(<br>XTensor * tensor, const int num, <br> const TENSOR_DATA_TYPE myDataType = X_FLOAT, <br>const int myDevID = -1, XMem * myMem = NULL) <br> <br> | tensor - 指向被初始化张量的指针 <br>  num - 向量维度大小 <br> myDataType - 张量的数据类型 <br> myDevID - 张量所在的设备ID <br> myMem - 张量所使用的内存池 |
| 初始化为稠密矩阵 | void InitTensor2D(<br>XTensor * tensor, const int colNum, const int rowNum, <br> const TENSOR_DATA_TYPE myDataType = X_FLOAT, <br>const int myDevID = -1, XMem * myMem = NULL) <br> <br> <br> | tensor - 指向被初始化张量的指针 <br>  colNum - 矩阵列数 <br> rowNum - 矩阵行数 <br> myDataType - 张量的数据类型 <br> myDevID - 张量所在的设备ID <br> myMem - 张量所使用的内存池 |
| 初始化为3维稠密张量 | void InitTensor3D(<br>XTensor * tensor, <br> const int d0, const int d1, const int d2, <br> const TENSOR_DATA_TYPE myDataType = X_FLOAT, <br>const int myDevID = -1, XMem * myMem = NULL) <br> <br> <br> | tensor - 指向被初始化张量的指针 <br>  d0 - 张量第一维大小 <br>  d1 - 张量第二维大小 <br>  d2 - 张量第三维大小 <br> myDataType - 张量的数据类型 <br> myDevID - 张量所在的设备ID <br> myMem - 张量所使用的内存池 |
| 初始化为4维稠密张量 | void InitTensor4D(<br>XTensor * tensor, <br> const int d0, const int d1, const int d2, const int d3, <br> const TENSOR_DATA_TYPE myDataType = X_FLOAT, <br>const int myDevID = -1, XMem * myMem = NULL) <br> <br> <br> <br> | tensor - 指向被初始化张量的指针 <br>  d0 - 张量第一维大小 <br>  d1 - 张量第二维大小 <br>  d2 - 张量第三维大小 <br>  d3 - 张量第四维大小 <br> myDataType - 张量的数据类型 <br> myDevID - 张量所在的设备ID <br> myMem - 张量所使用的内存池 |
| 初始化为5维稠密张量 | void InitTensor5D(<br>XTensor * tensor, <br> const int d0, const int d1, const int d2, <br> const int d3, const int d4, <br> const TENSOR_DATA_TYPE myDataType = X_FLOAT, <br>const int myDevID = -1, XMem * myMem = NULL) <br> <br> <br> <br> | tensor - 指向被初始化张量的指针 <br>  d0 - 张量第一维大小 <br>  d1 - 张量第二维大小 <br>  d2 - 张量第三维大小 <br>  d3 - 张量第四维大小 <br>  d4 - 张量第五维大小 <br> myDataType - 张量的数据类型 <br> myDevID - 张量所在的设备ID <br> myMem - 张量所使用的内存池 |
| 创建稠密向量 | XTensor * NewTensor1D(<br>const int num, <br> const TENSOR_DATA_TYPE myDataType = X_FLOAT, <br>const int myDevID = -1, XMem * myMem = NULL) <br> | num - 向量维度大小 <br> myDataType - 张量的数据类型 <br> myDevID - 张量所在的设备ID <br> myMem - 张量所使用的内存池 |
| 创建稠密矩阵 | XTensor * NewTensor2D(<br>const int colNum, const int rowNum, <br> const TENSOR_DATA_TYPE myDataType = X_FLOAT, <br>const int myDevID = -1, XMem * myMem = NULL) <br> <br> | colNum - 矩阵列数 <br> rowNum - 矩阵行数 <br> myDataType - 张量的数据类型 <br> myDevID - 张量所在的设备ID <br> myMem - 张量所使用的内存池 |
| 创建3维稠密张量 | XTensor * NewTensor3D(<br> const int d0, const int d1, const int d2, <br> const TENSOR_DATA_TYPE myDataType = X_FLOAT, <br>const int myDevID = -1, XMem * myMem = NULL) <br> <br> <br> | d0 - 张量第一维大小 <br>  d1 - 张量第二维大小 <br>  d2 - 张量第三维大小 <br> myDataType - 张量的数据类型 <br> myDevID - 张量所在的设备ID <br> myMem - 张量所使用的内存池 |
| 创建4维稠密张量 | XTensor * NewTensor4D(<br>const int d0, const int d1, const int d2, const int d3, <br> const TENSOR_DATA_TYPE myDataType = X_FLOAT, <br>const int myDevID = -1, XMem * myMem = NULL) <br> <br> <br> <br> | d0 - 张量第一维大小 <br>  d1 - 张量第二维大小 <br>  d2 - 张量第三维大小 <br>  d3 - 张量第四维大小 <br> myDataType - 张量的数据类型 <br> myDevID - 张量所在的设备ID <br> myMem - 张量所使用的内存池 |
| 创建5维稠密张量 | XTensor * NewTensor5D(<br>const int d0, const int d1, const int d2, <br> const int d3, const int d4, <br> const TENSOR_DATA_TYPE myDataType = X_FLOAT, <br>const int myDevID = -1, XMem * myMem = NULL) <br> <br> <br> <br> | d0 - 张量第一维大小 <br>  d1 - 张量第二维大小 <br>  d2 - 张量第三维大小 <br>  d3 - 张量第四维大小 <br>  d4 - 张量第五维大小 <br> myDataType - 张量的数据类型 <br> myDevID - 张量所在的设备ID <br> myMem - 张量所使用的内存池 |

## 访问张量中的内容

在C/C++中，我们通过XTensor.h访问张量中的内容，并且仅需要在源程序中引用XTensor.h头文件就可以完成张量的定义。
在此部分，我们主要对用户在访问张量内容时涉及到的成员变量及方法进行说明，更详细的说明请见附录。
在XTensor.h头文件中定义的成员变量说明：

| 成员变量 | 功能 |
| - | - |
| XMem * mem | 张量所使用的内存池 |
| void * data | 保存元素的数据数组 |
| int devID | 设备ID，指张量所申请的空间所在CPU或者GPU设备的编号，-1表示CPU |
| int order | 张量的维度，例如：一个矩阵（维度为2）是一个二维张量 |
| int dimSize[ ] | 张量中每一维度的大小，索引0表示第1维 |
| TENSOR_DATA_TYPE dataType | 每个数据单元的数据类型 |
| int unitSize | 数据单元的大小，类似于sizeof() |
| int unitNum | 数据单元的数量 |
| bool isSparse | 是否稠密，一个n * m稠密矩阵的数据量大小为n * m,而稀疏（非稠密）矩阵的数据量大小则取决于矩阵中非零元素个数。|
| float denseRatio | 稠密度，指非零单元的比例，是介于0和1之间的一个实数，0表示所有单元全为零，1表示全为非零单元。|

在XTensor.h头文件中定义的部分方法说明，详情参见附录：

| 功能 | 函数  | 参数 |
| - | - | - |
| 设置张量每一维度的大小 | void SetDim(int * myDimSize) |myDimSize - 张量每一维度的大小 |
| 得到张量中给定的维度大小 | int GetDim(const int dim) | dim - 张量的维度 |
| 重新调整矩阵维度 | void Reshape(<br> const int order, const int * myDimSize) | order - 张量的维度 <br> myDimSize - 张量每一维的大小 |
| 得到张量中元素数量 | int GetSize() | N/A |
| 用数组赋值张量 | void SetData(<br> const void * d, int num, int beg = 0) | d - 赋值数组  <br> num - 数组大小 <br> beg - 赋值时从张量的第几位开始 |
| 张量中所有元素设置为0 | void SetZeroAll(XStream * stream = NULL) | stream - 多线程流|
| 获取二维张量的值 | DTYPE Get2D(int ni, int mi = 0) | ni - 行值 <br> mi - 列值 |
| 设置二维张量中<br> 的单元值 | bool Set2D(DTYPE value, int ni, int mi = 0) | value - 单元值 <br> ni - 行值 <br> mi - 列值 |
| 增加二维张量中<br> 的单元值 | bool Add2D(DTYPE value, int ni, int mi = 0) | value - 单元值 <br> ni - 行值 <br> mi - 列值 |
| 将矩阵重置为特定大小 | bool Resize(<br> const int myOrder, <br> const int * myDimSize, <br> const TENSOR_DATA_TYPE myDataType = DEFAULT_DTYPE, <br> const float myDenseRatio = 1.0F) | myOrder - 张量的维度 <br> myDimSize - 张量每一维的大小,索引0表示第一维 <br> myDataType - 张量的数据类型 <br> myDenseRatio - 张量的稠密度，1表示稠密张量 |
| 将矩阵重置为<br> 另一矩阵大小 | bool Resize(<br> const XTensor * myTensor) | myTensor - 重置矩阵大小的参考矩阵 |
| 依据给定张量<br>释放数据空间 | void DelTensor(<br>const XTensor * tensor) | tensor - 给定张量 |

## 张量计算

NiuTrans.Tensor提供关于张量计算的函数功能，主要包括一些基本的张量运算以及激活函数，在本节中，主要对这些函数及其用法用例进行介绍。我们以点乘(Multiply)操作为例介绍NiuTrans.Tensor的几种函数定义形式：

* _Multiply: 需指定输出张量，只支持前向操作
* Multiply: 输出张量与输入张量相同，只支持前向操作
* MultiplyMe: 输出张量需返回给上层，同时支持前向和反向操作

### 代数计算(arithmetic)

此部分主要包括各种数学运算，加、减、乘、除、取负、取绝对值等。

#### 取绝对值（Absolute）

##### 什么是张量的取绝对值运算？
利用张量的取绝对值运算可以将张量中每一元素取绝对值并得到一个新的张量，一个维度分别为\\(2 \times 3\\)的矩阵取绝对值过程如下所示：

$$
\left(\begin{matrix}-1.0 & 2.0 & 3.0\\\\-4.0 & 5.0 & 6.0\end{matrix}\right) \rightarrow 
\left(\begin{matrix}1.0 & 2.0 & 3.0\\\\4.0 & 5.0 & 6.0\end{matrix}\right)
$$

##### 张量取绝对值的调用

NiuTrans.Tensor提供了张量的取绝对值操作，在NiuTrans.Tensor/Tensor/core/arithmetic中定义

张量取绝对值的调用方式以及参数说明如下所示:
```
void _Absolute(const XTensor * a, XTensor * b)

void _AbsoluteMe(XTensor * a)

XTensor Absolute(const XTensor & a)
```
Parameters: 

* a - 输入张量
* b - 输出张量

##### 张量取绝对值片段示例

用Absolute进行张量取绝对值操作的示例代码为：
```
/* call Absolute function */
b = Absolute(*a);
```
有关张量取绝对值的详细代码示例：

NiuTrans.Tensor/Tensor/test/TAbsolute.cpp

#### 矩阵乘法（MatrixMul）

##### 什么是张量间矩阵乘法？
利用矩阵乘法可以将矩阵想乘并得到一个新的结果矩阵，两个维度分别为\\(2 \times 3\\)和\\(3 \times 2\\)的矩阵相乘过程如下所示，结果矩阵的维度为\\(2 \times 2\\)：

$$
\left(\begin{matrix}1.0 & 2.0 & 3.0\\\\-4.0 & 5.0 & 6.0\end{matrix}\right) × 
\left(\begin{matrix}0.0 & -1.0\\\\1.0 & 2.0\\\\2.0 & 1.0\end{matrix}\right) \rightarrow 
\left(\begin{matrix}8.0 & 6.0\\\\17.0 & 20.0\end{matrix}\right)
$$

##### 矩阵乘法的调用

NiuTrans.Tensor提供了矩阵乘法的计算操作，在NiuTrans.Tensor/Tensor/core/arithmetic中定义，函数定义为：
>c_{i,j} = trans(ai) * trans(bj) * alpha + c_{i,j} * beta

矩阵乘法的调用方式以及参数说明如下所示:
```
void _MatrixMul(XTensor * a, MATRIX_TRANS_TYPE transposedA, XTensor * b, MATRIX_TRANS_TYPE transposedB, XTensor * c, DTYPE alpha = (DTYPE)1.0, DTYPE beta = 0)

XTensor MatrixMul(const XTensor &a, MATRIX_TRANS_TYPE transposedA, const XTensor &b, MATRIX_TRANS_TYPE transposedB, DTYPE alpha = (DTYPE)1.0, DTYPE beta = 0)
```
Parameters: 

* a - 输入张量1
* transposedA - a是否进行转置
* b - 输入张量2
* transposedB - b是否进行转置
* c - 输出张量
* alpha - 系数α
* beta - 系数β

##### 矩阵乘法片段示例

我们以最基本的二维矩阵乘法为例，用MatrixMul进行矩阵乘法操作的示例代码为：
```
/* call MatrixMul function */
t = MatrixMul(*s1, X_NOTRANS, *s2, X_NOTRANS);
```
有关矩阵乘法的详细代码示例：

NiuTrans.Tensor/Tensor/test/TMatrixMul.cpp

#### 点乘（Multiply）

##### 什么是张量点乘？

利用张量间的点乘操作可以进行张量间元素的按位置依次相乘，两个维度分别为\\(2 \times 2\\)的张量点乘过程如下所示：

$$
\left(\begin{matrix}0.0 & 1.0\\\\2.0 & 3.0\end{matrix}\right)  ·
\left(\begin{matrix}0.0 & 1.0\\\\2.0 & 3.0\end{matrix}\right)  \rightarrow 
\left(\begin{matrix}0.0 & 1.0\\\\4.0 & 9.0\end{matrix}\right)
$$

##### 张量点乘的调用

NiuTrans.Tensor提供了张量点乘的计算操作，用来计算张量中元素点乘结果，该函数在NiuTrans.Tensor/Tensor/core/arithmetic中定义，张量点乘的调用方式以及参数说明如下所示:
```
_Multiply(XTensor * a, XTensor * b, XTensor * c, int leadingDim, DTYPE alpha = 0)

void _MultiplyMe(XTensor * a, const XTensor * b, DTYPE alpha = 0, int leadingDim = 0)

XTensor Multiply(const XTensor &a, const XTensor &b, DTYPE alpha = 0, int leadingDim = 0)
```
Parameters: 

* a - 输入张量1
* b - 输入张量2
* c - 输出张量
* leadingDim - 沿着指定维度进行点乘操作
* alpha - 系数

##### 张量点乘片段示例

用Multiply进行s1和s2张量间的点乘操作的调用示例如下所示，计算结果存入t中：
```
/* call multiply function */
t = Multiply(*s1, *s2, 0);
```
有关矩阵乘法的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TMultiply.cpp

#### 取负（Negate）

##### 什么是张量的取负操作？

在进行张量的取负操作时，张量中每一元素都进行取负得到新的元素，所有新元素的组合得到新的结果张量，一个维度为\\(3 \times 2\\)的张量取负操作过程如下所示：

$$
\left(\begin{matrix}1.0 & -2.0\\\\-3.0 & 4.0\\\\5.0 & -6.0\end{matrix}\right) \rightarrow 
\left(\begin{matrix}-1.0 & 2.0\\\\3.0 & -4.0\\\\-5.0 & 6.0\end{matrix}\right)
$$

##### 张量取负的调用

NiuTrans.Tensor提供了张量取负的计算操作，进行张量的按元素位置进行取负操作，该函数在NiuTrans.Tensor/Tensor/core/arithmetic中定义，张量取负的调用方式以及参数说明如下所示:
```
void _Negate(const XTensor * a, XTensor * b)

void _NegateMe(XTensor * a)

XTensor Negate(const XTensor & a)
```
Parameters: 

* a - 输入张量
* b - 输出张量

##### 张量取负片段示例

用Negate进行张量取负操作的调用示例如下所示，其中a为我们要进行处理的张量，b为得到的结果张量：
```
/* call negate function */
b = Negate(*aGPU);
```
有关张量取负的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TNegate.cpp

#### 符号函数（Sign）

##### 什么是张量的符号函数？

张量的符号函数用来取得张量中每一元素的符号，一个维度为\\(3 \times 2\\)的张量符号函数操作过程如下所示：

$$
\left(\begin{matrix}1.0 & -2.0\\\\0.0 & 4.0\\\\5.0 & -6.0\end{matrix}\right) \rightarrow 
\left(\begin{matrix}1.0 & -1.0\\\\0.0 & 1.0\\\\1.0 & -1.0\end{matrix}\right)
$$

##### 张量符号函数的调用

NiuTrans.Tensor提供了张量的符号函数，该函数在NiuTrans.Tensor/Tensor/core/arithmetic中定义，张量符号函数的调用方式以及参数说明如下所示:
```
void _Sign(const XTensor * a, XTensor * b)

void _SignMe(XTensor * a)

XTensor Sign(const XTensor & a)
```
Parameters: 

* a - 输入张量
* b - 输出张量

##### 张量符号函数片段示例

用Sign进行张量符号函数的调用示例如下所示，其中a为我们要进行处理的张量，b为得到的结果张量：
```
/* call Sign function */
b = Sign(*a);
```
有关张量符号函数的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TSign.cpp

#### 加法（Sum）

##### 什么是张量加法？
张量加法的目的是将n个张量相加得到一个新的结果张量，结果张量某一位置的元素数值为进行操作的张量在该位置上元素的求和，在张量加法的计算过程中进行操作的张量与结果张量的维度相同，两个维度为\\(2\times 3\\)的张量相加过程如下所示：

$$
\left(\begin{matrix}0.0 & 1.0 & 2.0 \\\\ 3.0 & 4.0 & 5.0\end{matrix}\right) + 
\left(\begin{matrix}0.5 & 1.5 & 2.5 \\\\ 3.5 & 4.5 & 5.5\end{matrix}\right) \rightarrow
\left(\begin{matrix}0.5 & 2.5 & 4.5 \\\\ 6.5 & 8.5 & 10.5\end{matrix}\right)
$$

##### 张量加法的调用

NiuTrans.Tensor提供了张量加法的计算操作，在NiuTrans.Tensor/Tensor/core/arithmetic中定义，该操作用来进行张量之间的按元素位置相加，并得到相加的结果张量，张量加法的调用方法为：
```
void _Sum(const XTensor * a, const XTensor * b, XTensor * c, DTYPE beta = (DTYPE)1.0)

void _SumMe(XTensor * a, const XTensor * b, DTYPE beta = (DTYPE)1.0)

XTensor Sum(const XTensor &a, const XTensor &b, DTYPE beta = (DTYPE)1.0)
```
其中a和b为输入张量，c为结果张量，若c为NULL则将相加结果存入a中，beta为一个缩放参数，缩放公式为：c = a + b * beta，beta默认为1.0，NiuTrans.Tensor中张量加法的调用方式以及参数说明如下所示:

Parameters: 

* a - 输入张量1
* b - 输入张量2
* c - 输出张量
* beta - 缩放参数

##### 张量加法片段示例

调用Sum进行张量间的求和操作如下所示，在此例中将张量相加结果存入c中：
```
/* call sum function */
c = Sum(*a, *b);
```
详细代码示例见：

NiuTrans.Tensor/Tensor/test/TSum.cpp

#### SumByColumnTV

##### 什么是SumByColumnTV？

SumByColumnTV的作用是将一个Tensor和一个Vector按列相加，所得结果维度与Tensor一致，一个\\(2 \times 4\\)的Tensor和一个\\(2 \times 1\\)的Vector的SumByColumnTV操作过程如下所示：

$$
\left(\begin{matrix}0.0 & 1.0 & 2.0 & 3.0\\\\4.0 & 5.0 & 6.0 & 7.0\end{matrix}\right) + \left(\begin{matrix}1.0\\\\0.0\end{matrix}\right) \rightarrow 
\left(\begin{matrix}1.0 & 2.0 & 3.0 & 4.0\\\\4.0 & 5.0 & 6.0 & 7.0\end{matrix}\right)
$$ 

##### SumByColumnTV的调用

NiuTrans.Tensor提供了张量的SumByColumnTV操作，调用方法及参数说明如下所示:
```
void _SumByColumnTV(XTensor * a, XTensor * b, XTensor * c, DTYPE beta)
```
Parameters:

* a - 输入张量
* b - 输入向量
* c - 输出张量
* beta - 缩放参数

调用SumByColumnTV进行的运算为c_col = a_col + b * \beta

#####  SumByColumnTV片段示例

SumByColumnTV示例代码如下，其中a为输入的张量，b为输入的向量，c为a和b按列相加所得结果：
```
/* call SumByColumnTV function */
void _SumByColumnTV(a, b, c);
```
有关张量SumByColumnTV的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TSumByColumnTV.cpp

#### SumByColumnVT

##### 什么是SumByColumnVT？

SumByColumnVT的作用是将一个Vector和一个Tensor按列相加，所得结果维度与Vector一致，一个\\(2 \times 1\\)的Vector和一个\\(2 \times 4\\)的Tensor的SumByColumnVT操作过程如下所示：

$$
\left(\begin{matrix}1.0\\\\0.0\end{matrix}\right) + \left(\begin{matrix}0.0 & 1.0 & 2.0 & 3.0\\\\4.0 & 5.0 & 6.0 & 7.0\end{matrix}\right) \rightarrow 
\left(\begin{matrix}7.0\\\\22.0\end{matrix}\right)
$$ 

##### SumByColumnVT调用

NiuTrans.Tensor提供了张量的SumByColumnVT操作，调用方法及参数说明如下所示:
```
_SumByColumnVT(XTensor * a, XTensor * b, XTensor * c, DTYPE beta)
```
Parameters:

* a - 输入向量
* b - 输入张量
* c - 输出向量
* beta - 缩放参数

调用SumByColumnVT进行的运算为c = a + \sum{col} b_col * \beta

#####  SumByColumnVT片段示例

SumByColumnVT示例代码如下，其中a为输入的向量，b为输入的张量，c为a和b按列相加所得结果：
```
/* call SumByColumnVT function */
_SumByColumnVT(a, b, c);
```
有关张量SumByColumnVT的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TSumByColumnVT.cpp

### 张量存取(getandset)

此部分包括各种数据类型转化，设置数据、取数据等操作。

#### ConvertDataType

##### 什么是ConvertDataType？

ConvertDataType的作用是将张量中每个元素的数据类型转换为另一数据类型。

##### ConvertDataType调用

NiuTrans.Tensor提供了张量的ConvertDataType操作，调用方法及参数说明如下所示:

```
void _ConvertDataType(const XTensor * input, XTensor * output)
```
Parameters:

* input - 输入张量
* output - 输出张量

#####  ConvertDataType片段示例

ConvertDataType示例代码如下，本例中将张量中元素数据类型由flaot32转换为int32。

首先，创建张量时a为flaot32类型，b为int32类型：
```
/* create tensors */
XTensor * a = NewTensor(aOrder, aDimSize);
XTensor * b = NewTensor(aOrder, aDimSize, X_INT);
```
调用ConvertDataType函数
```
/* call ConvertDataType function */
_ConvertDataType(a, b);
```
有关张量ConvertDataType的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TConvertDataType.cpp

#### 选择（Select）

##### 什么是张量的选择操作？

Select时按张量指定维度上的指定位置对张量进行选择的操作，一个\\(2 \times 2 \times 4\\)的张量选择过程如下所示，本例中是选择张量维度2上位置索引为1和2的元素并存入目标张量，得到一个维度为\\(2 \times 2 \times 2\\)的张量：

$$
\begin{aligned}
\Biggl( 
& \left( 
\begin{matrix}0.0 & 1.0 & 2.0 & 3.0\\\\4.0 & 5.0 & 6.0 & 7.0\end{matrix}\right),\\\\ 
& \left( 
\begin{matrix}1.0 & 2.0 & 3.0 & 4.0\\\\5.0 & 6.0 & 7.0 & 8.0\end{matrix}
\right)
\Biggr)
\end{aligned} \rightarrow 
\begin{aligned}
\Biggl( 
& \left( 
\begin{matrix}1.0 & 2.0\\\\5.0 & 6.0\end{matrix}
\right),\\\\ 
& \left( 
\begin{matrix}2.0 & 3.0\\\\6.0 & 7.0\end{matrix}
\right)  
\Biggr)
\end{aligned}
$$

##### 张量选择的调用

NiuTrans.Tensor提供了张量的选择操作，调用方法及参数说明如下所示:

第一种选择操由一个0，1构成的index矩阵对张量进行选择：
```
void _Select(const XTensor * a, XTensor * c, XTensor * indexCPU)

XTensor Select(const XTensor &a, XTensor &indexCPU)
```

Parameters:

* a - 输入张量
* c - 输出张量
* indexCPU - 张量选择标志

第二种调用方式是按位置范围对张量进行选择：
```
void _SelectRange(const XTensor * a, XTensor * c, int dim, int low, int high)

XTensor SelectRange(const XTensor &a, int dim, int low, int high)
```
Parameters:

* a - 输入张量
* dim - 在哪一维对张量进行张量选择操作
* low - 张量选择范围的下限
* high - 张量选择范围的上限
* c - 输出张量

>需要注意的是，当张量选择的取值范围为[1,3]时意味着选择的是索引位置为1和2的值

#####  张量选择片段示例

张量选择示例代码如下，其中s为输入的待操作张量，t输出结果张量，在第三维上按范围[1,3]进行张量的选择操作：
```
/* call SelectRange function */
t = SelectRange(*s, 2, 1, 3);
```
有关张量选择的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TSelect.cpp

#### SetData

##### 什么是SetData？

SetData的作用是将张量在一定取值范围内随机进行初始化设置，一个\\(2 \times 4\\)的张量在[0.0,1.0]的取值范围SetData过程如下所示：

$$
\left(\begin{matrix}0.0 & 0.0 & 0.0 & 0.0\\\\0.0 & 0.0 & 0.0 & 0.0\end{matrix}\right) \rightarrow 
\left(\begin{matrix}0.1 & 0.5 & 0.3 & 0.9\\\\0.8 & 0.5 & 0.5 & 0.2\end{matrix}\right)
$$ 

##### SetData调用

NiuTrans.Tensor提供了张量的SetData操作，调用方法及参数说明如下所示:

设置张量为固定值：
```
void _SetDataFixed(XTensor * tensor, void * valuePointer)

void SetDataFixed(XTensor &tensor, DTYPE p)
```
Parameters:

* tensor - 输入张量
* valuePointer - 指向数据的指针
* p - 设置的值

设置张量为整型值：
```
void _SetDataFixedInt(XTensor * tensor, int p)
```
Parameters:

* tensor - 输入张量
* p - 固定整型值

设置张量为单精度浮点值：
```
void _SetDataFixedFloat(XTensor * tensor, float p)
```
Parameters:

* tensor - 输入张量
* p - 固定单精度浮点值

设置张量为双精度浮点值：
```
void _SetDataFixedDouble(XTensor * tensor, double p)
```
Parameters:

* tensor - 输入张量
* p - 固定双精度浮点值

设置张量为随机分布：
```
void _SetDataRand(XTensor * tensor, DTYPE low, DTYPE high)
```
* tensor - 输入张量
* low - 取值下限
* high - 取值上限

设置张量为正态分布：
```
void _SetDataRandN(XTensor * tensor, DTYPE mean, DTYPE standardDeviation)
```
Parameters:

* tensor - 输入张量
* mean - 均值
* standardDeviation - 标准差

#####  SetData片段示例

SetData示例代码如下，本例中是在[0.0,1.0]取值范围内对张量s进行随机初始化：
```
/* call SetData function */
s->SetDataRand(0.0, 1.0);
```
有关张量SetData的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TSetData.cpp

### 数学运算(math)

此部分包括各种非基本代数操作，包括：log、exp、power等。

#### 对数运算（Log）

##### 什么是张量的对数运算？

张量的对数运算即将张量中每一元素都取对数从而得到一个新的张量。

##### Log调用

NiuTrans.Tensor提供了张量的Log操作，调用方法及参数说明如下所示:
```
void _Log(const XTensor * a, XTensor * b)

void _LogMe(XTensor * a)

XTensor Log(const XTensor & a)
```
Parameters:

* a - 输入张量
* b - 输出张量
  
#####  Log片段示例

Log示例代码如下所示：
```
/* call Log function */
b = Log(*a);
```
有关Log的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TLog.cpp

#### 标准化（Normalize）

##### 什么是张量的标准化？

神经网络需要标准化处理（Normalize），这样做是为了弱化某些变量的值较大而对模型产生影响，Normalize函数定义为：
>y = a * (x-mean)/sqrt(variance+\epsilon) + b

##### Normalize调用

NiuTrans.Tensor提供了张量的Normalize操作，调用方法及参数说明如下所示:
```
void _Normalize(const XTensor * input, XTensor * output, int dim, const XTensor * mean, const XTensor * var, const XTensor * a, const XTensor * b, DTYPE epsilon)

void _NormalizeMe(XTensor * input, int dim, const XTensor * mean, const XTensor * var, const XTensor * a, const XTensor * b, DTYPE epsilon)

XTensor Normalize(const XTensor &input, int dim, const XTensor &mean, const XTensor &var, const XTensor &a, const XTensor &b, DTYPE epsilon)
```
Parameters:

* input - 输入张量
* output - 输出张量
* dim - 沿着指定维度产生均值和方差
* mean - 均值
* var - 方差
* a - 缩放
* b - 偏置
* epsilon - 防止方差为0的参数

#####  Normalize片段示例

Normalize示例代码如下所示：
```
/* call normalize function */
t = Normalize(*s, 0, *mean, *var, *a, *b, 0.0F);
```
有关Normalize的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TNormalize.cpp

#### 幂运算（Power）

##### 什么是张量的幂运算操作？

幂运算是一种关于幂的数学运算，张量的幂运算是将张量中的每个元素都进行幂运算从而得到新的张量，一个维度为\\(3 \times 2\\)的幂为2.0的张量幂运算过程如下所示：


$$
\left(\begin{matrix}1.0 & 2.0\\\\3.0 & 4.0\\\\5.0 & 6.0\end{matrix}\right) \rightarrow 
\left(\begin{matrix}1.0 & 4.0\\\\9.0 & 16.0\\\\25.0 & 36.0\end{matrix}\right)
$$

##### 张量幂运算的调用

NiuTrans.Tensor提供了张量幂运算的操作，用来进行张量的按元素位置进行幂运算的操作，调用方法为：
```
void _Power(const XTensor * a, XTensor * b, DTYPE p)

void _PowerMe(XTensor * a, DTYPE p)

XTensor Power(const XTensor & a, DTYPE p)
```
其中a为进行操作的张量，p为次方数，张量幂运算的参数说明如下所示:

Parameters: 

* a - 输入张量
* b - 输出张量
* p - 次方数

##### 张量幂运算片段示例

下面是调用Power进行a的幂为2.0的幂运算操作的一段示例代码：
```
/* call power function */
b = Power(*a, 2.0F);
```
有关张量幂运算的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TPower.cpp

#### 缩放和偏移（Scale and Shift）

##### 什么是张量的缩放和偏移？

张量的缩放和偏移计算公式为：p = p * scale + shift，其中scale和shift分别为张量缩放和偏移的参数，一个\\(2 \times 4\\)的张量进行缩放和偏移的过程如下所示，缩放参数取2.0，偏移参数取0.5：

$$
\left(\begin{matrix}0.0 & 1.0 & 2.0 & 3.0\\\\4.0 & 5.0 & 6.0 & 7.0\end{matrix}\right) \rightarrow 
\left(\begin{matrix}0.5 & 2.5 & 4.5 & 6.5\\\\8.5 & 10.5 & 12.5 & 14.5\end{matrix}\right)
$$

##### 张量缩放和偏移的调用

NiuTrans.Tensor提供了张量的缩放和偏移操作，调用方法为：
```
void _ScaleAndShift(const XTensor * a, XTensor * b, DTYPE scale, DTYPE shift = 0)

void _ScaleAndShiftMe(XTensor * a, DTYPE scale, DTYPE shift = 0)

XTensor ScaleAndShift(const XTensor &a, DTYPE scale, DTYPE shift = 0)
```
张量的缩放和偏移操作结果为：p = p * scale + shift，其中scale和shift分别为张量的缩放和偏移参数，张量缩放和偏移操作的参数说明如下表所示:

Parameters:

* a - 输入张量
* b - 输出张量
* scale - 缩放参数
* shift - 偏移参数

##### 张量缩放和偏移片段示例

张量缩放和偏移示例代码如下，input为输入的待操作张量，scaleFactor为缩放参数，shiftFactor为偏移参数：
```
/* call ScaleAndShift function */
t = ScaleAndShift(*s, scaleFactor, shiftFactor);
```
有关张量缩放和偏移的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TScaleAndShift.cpp

### 数据移动(movement)

此部分主要是介绍有关数据拷贝函数。

#### CopyIndexed

##### 什么是张量的CopyIndexed操作？

CopyIndexed，即按指定索引位置拷贝张量，一个\\(2 \times 2 \times 3\\)的张量拷贝过程如下所示，本例中是对张量维度2上起始位置索引为0和2的1个元素进行拷贝，所得张量维度为\\(2 \times 2 \times 2\\)：

$$
\begin{aligned}
\Biggl( 
& \left( 
\begin{matrix}0.0 & -1.0 & 2.0\\\\2.0 & 1.0 & 3.0\end{matrix}\right),\\\\ 
& \left( 
\begin{matrix}1.0 & 2.0 & 4.0\\\\3.0 & 1.0 & 2.0\end{matrix}
\right),\\\\ 
& \left( 
\begin{matrix}-1.0 & 3.0 & 2.0\\\\1.0 & -1.0 & 0.0\end{matrix}
\right)  
\Biggr)
\end{aligned} \rightarrow 
\begin{aligned}
\Biggl( 
& \left( 
\begin{matrix}0.0 & 2.0\\\\2.0 & 3.0\end{matrix}\right),\\\\ 
& \left( 
\begin{matrix}1.0 & 4.0\\\\3.0 & 2.0\end{matrix}
\right),\\\\ 
& \left( 
\begin{matrix}-1.0 & 2.0\\\\1.0 & 0.0\end{matrix}
\right)  
\Biggr)
\end{aligned}
$$

##### 张量CopyIndexed的调用

NiuTrans.Tensor提供了张量的CopyIndexed操作，调用方法及参数说明如下所示:
```
void _CopyIndexed(const XTensor * s, XTensor * t, int dim, int * srcIndex, int indexSize, int * tgtIndex, int copyNum)

XTensor CopyIndexed(const XTensor &s, int dim, int * srcIndex,int indexSize, int * tgtIndex, int copyNum)
```
Parameters:

* s - 输入张量
* t - 输出张量
* dim - 在哪一维对张量进行CopyIndexed操作
* srcIndex - 源索引，即在指定dim上进行赋值的值的索引
* indexSize - 源索引的个数
* tgtIndex - 目标索引，所赋值的值在输出张量中的索引
* copyNum - 以源索引为起始位置拷贝的元素个数

#####  张量CopyIndexed片段示例

CopyIndexed示例代码如下，其中s为输入的待操作张量，t输出结果张量，在指定维度上按起始位置索引拷贝一个元素到目标张量：
```
/* call CopyIndexed function */
t = CopyIndexed(*s, dim, srcIndex, indexSize, tgtIndex, copyNum);
```
有关CopyIndexed的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TCopyIndexed.cpp

#### 拷贝（CopyValues）

##### 什么是张量的拷贝操作？

拷贝，即将一个张量的值赋给另一个张量，也就是对张量进行拷贝操作，一个\\(2 \times 4\\)的张量拷贝过程如下所示：

$$
\left(\begin{matrix}5.0 & 1.0 & 2.0 & 8.0\\\\4.0 & 3.0 & 7.0 & 6.0\end{matrix}\right) \rightarrow
\left(
\begin{matrix}5.0 & 1.0 & 2.0 & 8.0\\\\4.0 & 3.0 & 7.0 & 6.0\end{matrix}\right)
$$

##### 张量拷贝操作的调用

NiuTrans.Tensor提供了张量的拷贝操作，调用方法及参数说明如下所示:
```
void _CopyValues(const XTensor * s, XTensor * t, XStream * stream = NULL)

XTensor CopyValues(const XTensor &s, XStream * stream = NULL)
```
Parameters:

* s - 输入张量
* t - 输出张量
* stream - 多线程流

#####  张量拷贝片段示例

张量拷贝示例代码如下，其中s为输入的待操作张量，t输出结果张量：
```
/* call CopyValues function */
t = CopyValues(*s);
```
有关张量拷贝的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TCopyValues.cpp

### 规约操作(reduce)

#### 归约取最大值（ReduceMax）

##### 什么是张量的归约取最大值？

张量的归约取最大值操作是沿着张量的某一维度，取得该向量在该维度中的最大值,一个\\(2 \times 4\\)的张量在维度0和维度1进行取最大值操作的过程分别如下所示：

$$
\left(\begin{matrix}0.0 & 1.0 & 2.0 & 3.0\\\\4.0 & 5.0 & 6.0 & 7.0\end{matrix}\right) \rightarrow 
\left(\begin{matrix}4.0 & 5.0 & 6.0 & 7.0\end{matrix}\right)
$$

$$
\left(\begin{matrix}0.0 & 1.0 & 2.0 & 3.0\\\\4.0 & 5.0 & 6.0 & 7.0\end{matrix}\right) \rightarrow 
\left(\begin{matrix}3.0\\\\7.0\end{matrix}\right)
$$

##### 张量归约取最大值操作的调用

NiuTrans.Tensor提供了张量的ReduceMax操作，用来获得张量中沿指定维度取得的最大值，张量归约取最大值操作的调用方式及参数说明如下所示:
```
void _ReduceMax(const XTensor * input, XTensor * output, int dim)

XTensor ReduceMax(const XTensor &input, int dim)
```
Parameters:

* input - 输入张量
* output - 输出张量
* dim - 沿着指定维度进行取最大值操作

##### 张量归约取最大值片段示例

调用ReduceMax进行张量归约取最大值操作的示例代码如下所示，代码中两行分别表示沿着维度0和维度1进行取值：
```
/* call reduce max function */
t = ReduceMax(*s, 0);
t = ReduceMax(*s, 1);
```
有关张量归约取最大值的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TReduceMax.cpp

#### 归约取均值（ReduceMean）

##### 什么是张量的归约取均值操作？

张量的归约取均值操作是沿着张量的某一维度，计算该张量在该维度的均值,一个\\(2 \times 4\\)的张量在维度0和维度1进行取均值操作的过程分别如下所示：

$$
\left(\begin{matrix}0.0 & 1.0 & 2.0 & 3.0\\\\4.0 & 5.0 & 6.0 & 7.0\end{matrix}\right) \rightarrow 
\left(\begin{matrix}2.0 & 3.0 & 4.0 & 5.0\end{matrix}\right)
$$

$$
\left(\begin{matrix}1.0 & 1.0 & 3.0 & 3.0\\\\4.0 & 4.0 & 6.0 & 6.0\end{matrix}\right) \rightarrow 
\left(\begin{matrix}2.0\\\\5.0\end{matrix}\right)
$$

##### 张量归约取均值操作的调用

NiuTrans.Tensor提供了张量的ReduceMean操作，调用方法为：
```
void _ReduceMean(const XTensor * input, XTensor * output, int dim)

XTensor ReduceMean(const XTensor &input, int dim)
```
ReduceMean用来获得张量中沿指定维度取得的数值均值，张量归约取均值的参数说明如下所示:

Parameters:

* input - 输入张量
* output - 输出张量
* dim - 沿着指定维度进行取平均值操作

##### 张量归约取均值片段示例

调用ReduceMean进行张量归约取均值操作的示例代码如下所示，代码中两行分别表示沿着维度0和维度1进行取值：
```
/* call reduce mean function */
t = ReduceMean(*s, 0);
t = ReduceMean(*s, 1);
```
有关张量归约取均值的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TReduceMean.cpp

#### 归约求和（ReduceSum）

##### 什么是张量的归约求和操作？

张量的归约求和操作是沿着张量的某一维度，计算该张量在该维度的和,一个\\(2 \times 4\\)的张量在维度0和维度1进行求和操作的过程分别如下所示：

$$
\left(\begin{matrix}0.0 & 1.0 & 2.0 & 3.0\\\\4.0 & 5.0 & 6.0 & 7.0\end{matrix}\right) \rightarrow 
\left(\begin{matrix}4.0 & 6.0 & 8.0 & 10.0\end{matrix}\right)
$$

$$
\left(\begin{matrix}0.0 & 1.0 & 2.0 & 3.0\\\\4.0 & 5.0 & 6.0 & 7.0\end{matrix}\right) \rightarrow 
\left(\begin{matrix}6.0\\\\22.0\end{matrix}\right)
$$

##### 张量归约求和操作的调用

NiuTrans.Tensor提供了张量的ReduceSum操作，调用方法为：
```
void _ReduceSum(const XTensor * input, XTensor * output, int dim, const XTensor * shift = NULL, DTYPE power = (DTYPE)1.0F, bool isExp = false)

XTensor ReduceSum(const XTensor &input, int dim, const XTensor &shift = NULLTensor, DTYPE power = (DTYPE)1.0F, bool isExp = false)
```
其中shift默认为NULL，power默认为1.0F，isExp默认为false，张量归约求和操作的参数说明如下所示:

Parameters:

* input - 输入张量
* output - 输出张量
* dim - 沿着指定维度进行取最大值操作
* shift - 输入的偏移，默认为NULL
* power - 元素的幂，默认为1.0F
* isExp - 是否取指，默认为false

##### 张量归约求和片段示例

调用ReduceSum进行张量归约求和操作的示例代码如下所示，代码中两行分别表示沿着维度0和维度1进行取值：
```
/* call reduce sum function */
t1 = ReduceSum(*s, 0, *shift1);
t2 = ReduceSum(*s, 1, *shift2);
```
有关张量归约求和的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TReduceSum.cpp

#### 归约取方差（ReduceSumSquared）

##### 什么是张量的归约取方差操作？

张量的归约取方差操作是沿着张量的某一维度，计算该张量在该维度的方差,一个\\(2 \times 4\\)的张量在维度0进行取方差操作的过程如下所示：

$$
\left(\begin{matrix}0.0 & 1.0 & 2.0 & 3.0\\\\4.0 & 5.0 & 6.0 & 7.0\end{matrix}\right) \rightarrow 
\left(\begin{matrix}8.0 & 8.0 & 8.0 & 8.0\end{matrix}\right)
$$

##### 张量归约取方差操作的调用

NiuTrans.Tensor提供了张量的ReduceSumSquared操作，调用方法为：
```
void _ReduceSumSquared(const XTensor * input, XTensor * output, int dim, const XTensor * shift)

XTensor ReduceSumSquared(const XTensor &input, int dim, const XTensor &shift)
```
ReduceSumSquared用来计算张量的沿着某一维度元素的方差，张量归约取方差操作的参数说明如下所示:

Parameters:

* input - 输入张量
* output - 输出张量
* dim - 沿着指定维度进行取平均值操作
* shift - 输入的偏移

##### 张量归约取方差片段示例

调用ReduceSumSquared进行张量归约取方差操作的示例代码如下所示：
```
/* call reduce sum squared function */
t = ReduceSumSquared(*s, 0, *shift);
```
有关张量归约取方差的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TReduceSumSquared.cpp

#### 归约取标准差（ReduceVariance）

##### 什么是张量的归约取标准差操作？

张量的归约取标准差操作是沿着张量的某一维度，计算该张量在该维度的标准差,一个\\(2 \times 4\\)的张量在维度0进行取标准差操作的过程如下所示：

$$
\left(\begin{matrix}0.0 & 1.0 & 2.0 & 3.0\\\\4.0 & 5.0 & 6.0 & 7.0\end{matrix}\right) \rightarrow 
\left(\begin{matrix}4.0 & 4.0 & 4.0 & 4.0\end{matrix}\right)
$$

##### 张量归约取标准差操作的调用

NiuTrans.Tensor提供了张量的ReduceVariance操作，调用方法为：
```
void _ReduceVariance(const XTensor * input, XTensor * output, int dim, const XTensor * mean)

XTensor ReduceVariance(const XTensor &input, int dim, const XTensor &mean)
```
ReduceVariance用来计算张量的沿着某一维度元素的标准差，张量归约取标准差操作的参数说明如下所示:

Parameters:

* input - 输入张量
* output - 输出张量
* dim - 沿着指定维度进行取标准差操作
* mean - 均值

##### 张量归约取标准差片段示例

调用ReduceVariance进行张量归约取标准差操作的示例代码如下所示：
```
/* call reduce variance function */
t = ReduceVariance(*s, 0, *mean);
```
有关张量归约取标准差的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TReduceVariance.cpp

### 形状转换(shape)

此部分主要包括关于形状改变的函数，比如：split、merge、reshape等。

#### 级联（Concatenate）

##### 什么是张量的级联操作？

张量间的级联操作是沿着张量的某一维度，将一系列张量或是一个列表中的所有张量连接在一起组成一个更大的张量，将维度分别为\\(2 \times 1\\)和\\(2 \times 2\\)的两个张量进行级联过程如下所示：

$$
\left(\begin{matrix}0.0\\\\1.0\end{matrix}\right) +
\left(\begin{matrix}2.0 & 3.0\\\\4.0 & 5.0\end{matrix}\right) \rightarrow 
\left(\begin{matrix}0.0 & 2.0 & 3.0\\\\1.0 & 4.0 & 5.0\end{matrix}\right)
$$

##### 张量级联的调用

NiuTrans.Tensor提供了张量间的级联操作，调用方法为：

第一种调用方法中的操作对象是列表，将进行级联操作的张量存入列表smalls中，级联结果存入张量big中：
```
void _Concatenate(const XList * smalls, XTensor * big, int dim)

XTensor Concatenate(const XList &smalls, int dim)
```
Parameters:

* smalls - 进行级联张量的列表
* big - 输出张量
* dim - 在指定维度进行级联

第二种方法操作对象不再是列表中的张量而是直接对一系列张量进行级联操作：
```
void _Concatenate(const XTensor * smallA, const XTensor * smallB, XTensor * big, int dim)

XTensor Concatenate(const XTensor &smallA, const XTensor &smallB, int dim)
```
Parameters:

* smallA - 输入张量1
* smallB - 输入张量2
* big - 输出张量
* dim - 进行级联的维度

##### 张量级联片段示例

通过操作张量列表进行张量的级联操作片段示例如下所示，sList为存放进行级联张量的列表，t为结果张量：
```
/* call concatenate function */
t = Concatenate(*sList, 1);
```
直接通过操作一系列张量进行张量的级联操作片段示例如下所示，s1、s2为需要进行级联的张量，t为结果张量：
```
/* call concatenate function */
t = Concatenate(*s1, *s2, 1);
```
有关张量级联的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TConcatenate.cpp

#### 合并（Merge）

##### 什么是张量的合并操作？

张量间的合并操作与级联有些类似，是沿着张量的某一维度，可以将一个张量合并为另一个维度不同的张量，也可以将一个列表中的所有张量合并在一起组成一个更大的张量。

在第一种情况下将维度为\\(2 \times 2 \times 3\\)的张量在维度1进行合并，进行合并的维度为0，得到维度为\\(4 \times 3\\)的张量的过程如下所示：

$$
\begin{aligned}
\Biggl( & \left( 
\begin{matrix}0.0 & 1.0 & 2.0\\\\3.0 & 4.0 & 5.0\end{matrix}\right),
\\\\ & \left( 
\begin{matrix}0.1 & 1.1 & 2.1\\\\3.1 & 4.1 & 5.1\end{matrix}
\right) \Biggr)
\end{aligned} \rightarrow 
\left(\begin{matrix}0.0 & 1.0 & 2.0\\\\3.0 & 4.0 & 5.0\\\\0.1 & 1.1 & 2.1\\\\3.1 & 4.1 & 5.1\end{matrix}\right)
$$

在第二种情况下将两个维度均为\\(2 \times 3\\)的张量沿着维度0合并为维度为\\(4 \times 3\\)的张量的过程如下所示：

$$
\left(\begin{matrix}0.0 & 2.0 & 3.0\\\\1.0 & 4.0 & 5.0\end{matrix}\right) + \left(\begin{matrix}0.1 & 1.1 & 2.1\\\\3.1 & 4.1 & 5.1\end{matrix}\right) \rightarrow 
\left(\begin{matrix}0.0 & 1.0 & 2.0\\\\3.0 & 4.0 & 5.0\\\\0.1 & 1.1 & 2.1\\\\3.1 & 4.1 & 5.1\end{matrix}\right)
$$ 

##### 张量合并操作的调用

NiuTrans.Tensor提供了张量的合并操作，调用方法为：

在第一种调用方法中是将源张量中的某一维度进行Merge操作，Merge结果为张量t，whereToMerge为指定进行Merge操作的维度，leadingDim为指定将哪一维度Merge，例如：(N/2, 2, M) -> (N, M)，参数说明如下表所示:
```
void _Merge(const XTensor * s, XTensor * t, int whereToMerge, int leadingDim = -1)

XTensor Merge(const XTensor &s, int whereToMerge, int leadingDim = -1)
```
Parameters:

* s - 输入张量
* t - 输出张量
* whereToMerge - 沿着指定维度进行Merge操作
* leadingDim - 把指定维度进行Merge操作

在第二种调用方法中是将所操作张量存入列表smalls中，操作结果为张量big，whereToMerge为指定进行Merge操作的维度，例如：2 * (N/2, M) -> (N, M)，参数说明如下表所示:
```
void _Merge(const XList * smalls, XTensor * big, int whereToMerge)

XTensor Merge(const XList &smalls, int whereToMerge)
```
Parameters:

* smalls - 存放进行合并张量的列表
* big - 结果张量
* whereToMerge - 沿着指定维度进行Merge操作

##### 张量合并片段示例

上述第一种张量合并片段示例如下所示，s为进行合并的张量，t为结果张量，1表示在维度1进行合并操作，0表示将维度0进行合并操作：
```
/* call merge function */
t = Merge(*s, 1, 0);
```
上述第二种张量合并片段示例如下所示，sList为要进行合并的张量列表，t为结果张量，0表示沿着维度0进行合并操作：
```
/* call merge function */
t = Merge(*sList, 0);
```

有关张量合并的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TMerge.cpp

#### 切分（Split）

##### 什么是张量的切分操作？

张量间的切分操作是沿着张量的某一维度，可以将一个张量切分成另一张量，也可以将一个大的张量切分成n个小的张量集合的列表。

第一种情况下将维度为\\(4 \times 3\\)张量沿着维度0进行切分，切分份数为2，得到维度为\\(2 \times 2 \times 3\\)的张量的过程如下所示：

$$
\left(\begin{matrix}0.0 & 1.0 & 2.0\\\\3.0 & 4.0 & 5.0\\\\0.1 & 1.1 & 2.1\\\\3.1 & 4.1 & 5.1\end{matrix}\right) \rightarrow 
\begin{aligned}
\Biggl( & \left( 
\begin{matrix}0.0 & 1.0 & 2.0\\\\3.0 & 4.0 & 5.0\end{matrix}\right),
\\\\ & \left( 
\begin{matrix}0.1 & 1.1 & 2.1\\\\3.1 & 4.1 & 5.1\end{matrix}
\right) \Biggr)
\end{aligned}
$$

在第二种情况下将维度为\\(4 \times 3\\)张量沿着维度0进行切分，切分份数为2，得到两个维度均为\\(2 \times 3\\)的张量的过程如下所示：

$$
\left(\begin{matrix}0.0 & 1.0 & 2.0\\\\3.0 & 4.0 & 5.0\\\\0.1 & 1.1 & 2.1\\\\3.1 & 4.1 & 5.1\end{matrix}\right) \rightarrow 
\left(\begin{matrix}0.0 & 2.0 & 3.0\\\\1.0 & 4.0 & 5.0\end{matrix}\right) + \left(\begin{matrix}0.1 & 1.1 & 2.1\\\\3.1 & 4.1 & 5.1\end{matrix}\right)
$$

##### 张量切分的调用

NiuTrans.Tensor提供了两种张量切分操作，调用方法为：

在第一种调用方法中是将源张量中的某一维度进行Split操作，Split结果为张量t，whereToSplit为在哪一维度进行split操作，splitNum表示分成多少份，例如：(N, M) -> (N/3, M, 3)，参数说明如下所示:
```
void _Split(const XTensor * s, XTensor * t, int whereToSplit, int splitNum)

XTensor Split(const XTensor &s, int whereToSplit, int splitNum)
```
Parameters:

* s - 输入张量
* t - 输出张量
* whereToSplit - 在指定维度进行split操作
* splitNum - 分成多少份

在第二种调用方法中是将所操作张量big按某一维度whereToSplit进行Split操作，操作结果为包含若干更小维度张量的列表smalls，splitNum表示分成多少份，例如：(N, M) -> 2 * (N/2, M)，参数说明如下所示:
```
void _Split(const XTensor * big, XList * smalls, int whereToSplit, int splitNum)

XList SplitList(const XTensor &big, int whereToSplit, int splitNum)
```
Parameters:

* big - 输入张量
* smalls - 存放切分出张量的列表
* whereToSplit - 在指定维度进行split操作
* splitNum - 分成多少份

##### 张量切分片段示例

上述第一种张量切分片段示例如下所示，s为进行切分的张量，t为结果张量，0表示沿着维度0进行切分操作，2表示切分份数为2：

```
/* call split function */
t = Split(*s, 0, 2);
```

上述第二种张量切分片段示例如下所示，s为进行切分的张量，tList为存放结果张量的列表，1表示沿着维度1进行切分操作，2表示切分份数为2：

```
/* call split function */
Split(*s, tList, 1, 2);
```

有关张量切分的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TSplit.cpp

#### Unsqueeze

##### 什么是Unsqueeze？

Unsqueeze的作用是通过对张量进行操作，返回一个新的在指定维度插入新维度的张量，这个返回的张量与源张量共享相同的基础数据，一个\\(2 \times 3\\)的张量在维度1和2分别进行Unsqueeze的操作如下所示，插入新的维度大小均为2：

$$
\left(\begin{matrix}0.0 & 1.0 & 2.0\\\\3.0 & 4.0 & 5.0\end{matrix}\right) \rightarrow 
\begin{aligned}
\Biggl( & \left( 
\begin{matrix}0.0 & 1.0 & 2.0\\\\0.0 & 1.0 & 2.0\end{matrix}\right),
\\\\ & \left( 
\begin{matrix}3.0 & 4.0 & 5.0\\\\3.0 & 4.0 & 5.0\end{matrix}
\right) \Biggr)
\end{aligned}
$$

$$
\left(\begin{matrix}0.0 & 1.0 & 2.0\\\\3.0 & 4.0 & 5.0\end{matrix}\right) \rightarrow  
\begin{aligned}
\Biggl( & \left( 
\begin{matrix}0.0 & 0.0\\\\1.0 & 1.0\\\\2.0 & 2.0\end{matrix}\right),
\\\\ & \left( 
\begin{matrix}3.0 & 3.0\\\\4.0 & 4.0\\\\5.0 & 5.0\end{matrix}
\right) \Biggr)
\end{aligned}
$$

##### Unsqueeze的调用

NiuTrans.Tensor提供了张量的Unsqueeze操作，调用方法及参数说明如下所示:
```
void _Unsqueeze(const XTensor * a, XTensor * b, int dim, int dSize)

XTensor Unsqueeze(const XTensor &a, int dim, int dSize)
```
Parameters:

* a - 输入张量
* b - 输出张量
* dim - 在指定维度进行Unsqueeze操作
* dSize - 插入维度的大小

#####  Unsqueeze片段示例

Unsqueeze示例代码如下，其中s为输入的待操作张量，t1、t2代表输出结果张量，以下两行分别表示在维度1和维度2上插入的维度大小为2：

```
/* call Unsqueeze function */
t1 = Unsqueeze(*s, 1, 2);
t2 = Unsqueeze(*s, 2, 2);
```

有关张量Unsqueeze的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TUnsqueeze.cpp

### 排序操作(sort)

此部分主要介绍排序相关的函数，如：sort、topk等。

#### Sort

##### 什么是Sort？

Sort操作是对张量中元素沿着指定的维度进行排序，一个\\(2 \times 4\\)的张量沿着维度0进行Sort操作过程如下所示：

$$
\left(\begin{matrix}0.0 & 1.0 & 2.0 & 3.0\\\\4.0 & 5.0 & 6.0 & 7.0\end{matrix}\right) \rightarrow 
\left(\begin{matrix}4.0 & 5.0 & 6.0 & 7.0\\\\0.0 & 1.0 & 2.0 & 3.0\end{matrix}\right)
$$

##### Sort的调用

NiuTrans.Tensor提供了张量的Sort操作，调用方法及参数说明如下所示:

```
void _Sort(const XTensor * a, XTensor * b, XTensor * index, int dim)

void _SortMe(XTensor * a, XTensor * index, int dim)

void Sort(XTensor & a, XTensor & b, XTensor & index, int dim)
```

Parameters:

* a - 输入张量
* b- 输出张量
* index - 输出张量中元素的索引
* dim - 沿着指定维度进行Sort操作

#####  Sort片段示例

Sort示例代码如下所示，a为进行操作的张量，index为结果张量中元素的索引，本例中沿着维度0进行Sort操作：

```
/* call Sort function */
Sort(*a, b, *index, 0)
```

有关Sort的详细代码示例见                              NiuTrans.Tensor/Tensor/test/TSort.cpp

#### TopK

##### 什么是TopK？

TopK操作是通过对张量中元素进行排序，得到最大或最小的k个元素值及其对应的索引值，在张量中，可以沿着某一维度进行TopK操作，一个\\(2 \times 4\\)的张量沿着维度0进行Top-2操作过程如下所示：

$$
\left(\begin{matrix}5.0 & 1.0 & 2.0 & 8.0\\\\4.0 & 3.0 & 7.0 & 6.0\end{matrix}\right) \rightarrow 
\begin{aligned}
outputAnswer: & \left(
\begin{matrix}0.5 & 2.5 & 4.5 & 6.5\\\\8.5 & 10.5 & 12.5 & 14.5\end{matrix}\right)\\\\ +
\\\\ indexAnswer: & \left(
\begin{matrix}0 & 1 & 1 & 0\\\\1 & 0 & 0 & 1\end{matrix}\right)
\end{aligned}
$$

##### TopK的调用

NiuTrans.Tensor提供了张量的TopK操作，调用方法及参数说明如下所示:

```
void _TopK(const XTensor * a, XTensor * b, XTensor * index, int dim, int k)

void TopK(XTensor &a, XTensor &b, XTensor &index, int dim, int k)
```

Parameters:

* a - 输入张量
* b - 输出张量
* index - 输出结果索引
* dim - 沿着指定维度进行TopK操作
* k - TopK中k代表取最大的k个值

#####  TopK片段示例

TopK示例代码如下，s为输入的待操作张量，t输出结果张量，index为输出结果索引，本例中沿着维度dim取Top-k：

```
/* call TopK function */
int dim = 0;
int k = inputDimSize[dim];
TopK(s, t, index, dim, k);
```

有关TopK的详细代码示例见                              NiuTrans.Tensor/Tensor/test/TTopK.cpp

### 激活函数(function)

此部分主要介绍一些激活函数和损失函数。

#### HardTanH

##### 什么是HardTanH？

HardTanH是一种激活函数，HardTanH函数定义为：
>y =  1 &nbsp;&nbsp;if x > 1 <br />
&nbsp;&nbsp; &nbsp;&nbsp;&nbsp; x &nbsp;&nbsp;if -1 <= x <= 1 <br />
&nbsp;&nbsp; &nbsp; -1 &nbsp;&nbsp;if x < -1

##### HardTanH调用

NiuTrans.Tensor提供了张量的HardTanH激活函数，调用方法及参数说明如下所示:

```
void _HardTanH(const XTensor * x, XTensor * y)

XTensor HardTanH(const XTensor &x)
```

Parameters:

* x - 输入张量
* y - 输出张量

#####  HardTanH片段示例

HardTanH示例代码如下，其中x为输入的向量，y为输入的张量：

```
/* call hardtanh function */
y = HardTanH(*x);
```

有关HardTanH的详细代码示例见：

NiuTrans.Tensor/Tensor/test/THardTanH.cpp

#### Identity

##### 什么是Identity？

Identity是一种激活函数，Identity函数定义为：
>y = x

##### Identity调用

NiuTrans.Tensor提供了张量的Identity激活函数，调用方法及参数说明如下所示:

```
void _Identity(const XTensor * x, XTensor * y)

XTensor Identity(const XTensor &x)
```

Parameters:

* x - 输入张量
* y - 输出张量

#####  Identity片段示例

Identity示例代码如下，其中x为输入的向量，y为输入的张量：

```
/* call Identity function */
y = Identity(*x);
```

有关Identity的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TIdentity.cpp

#### LogSoftmax

##### 什么是LogSoftmax？

LogSoftmax是一种激活函数，LogSoftmax函数定义为：
>y = log(e^x / \sum_{i} e^{x_i})

##### LogSoftmax调用

NiuTrans.Tensor提供了张量的LogSoftmax激活函数，调用方法及参数说明如下所示:

```
void _LogSoftmax(const XTensor * x, XTensor * y, int leadDim)

XTensor LogSoftmax(const XTensor &x, int leadDim)
```

Parameters:

* x - 输入张量
* y - 输出张量
* leadDim - 沿着指定维度进行操作

#####  LogSoftmax片段示例

LogSoftmax示例代码如下，其中x为输入的向量，y为输入的张量，本例中沿着维度1进行LogSoftmax操作：

```
/* call LogSoftmax function */
y = LogSoftmax(*x, 1);
```

有关LogSoftmax的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TLogSoftmax.cpp

#### Loss

##### 什么是Loss？

Loss Function(损失函数)是用来衡量神经网络模型效果及优化目标的一种损失函数，函数定义为：
>squared error : loss = sum_{i} 0.5*(gold_i - output_i)^2 <br />
cross entropy : loss = sum_{i} (-gold_i * log(output_i)) <br />
one hot error : loss = sum_{i} e_i <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; where e_i = 0.5*(t_i - y_i)^2 &nbsp;&nbsp;if t_i = 1, <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;e_i = 0 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; otherwise


##### Loss调用

NiuTrans.Tensor提供了张量的Loss激活函数，调用方法及参数说明如下所示:

```
DTYPE _LossCompute(XTensor * gold, XTensor * output, LOSS_FUNCTION_NAME LFName, bool isLogOutput, int leadDim, int gBeg, int gLen, int oBeg)
```

Parameters:

* gold - 标准答案
* output - 输出的模型预测结果
* LFName - 损失函数名称
* isLogOutput - 输出是否log
* leadDim - 沿着指定维度进行输出
* gBeg - 沿着指定维度leadDim从指定位置取标准答案
* gLen - 从指定位置gBeg开始标准答案的偏移
* oBeg - 沿着指定维度leadDim从指定位置开始输出模型预测结果

#####  Loss片段示例

Loss示例代码如下所示：

```
/* call LossCompute function */
error = _LossCompute(gold, output, SQUAREDERROR, false, 0, 0, dimSize[0], 0);
```

有关Loss的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TLoss.cpp

#### Rectify

##### 什么是Rectify？

Rectify是一种激活函数，Rectify函数定义为：
>y = max(0, x)

##### Rectify调用

NiuTrans.Tensor提供了张量的Rectify激活函数，调用方法及参数说明如下所示:

```
void _Rectify(const XTensor * x, XTensor * y)

XTensor Rectify(const XTensor &x)
```


Parameters:

* x - 输入张量
* y - 输出张量

#####  Rectify片段示例

Rectify示例代码如下，其中x为输入的向量，y为输入的张量：
```
/* call Rectify function */
y = Rectify(*x);
```
有关Rectify的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TRectify.cpp

#### Sigmoid

##### 什么是Sigmoid？

Sigmoid是一种激活函数，Sigmoid函数定义为：
>y = 1/(1+exp(-x))

##### Sigmoid调用

NiuTrans.Tensor提供了张量的Sigmoid激活函数，调用方法及参数说明如下所示:
```
void _Sigmoid(const XTensor * x, XTensor * y)

XTensor Sigmoid(const XTensor &x)
```
Parameters:

* x - 输入张量
* y - 输出张量

#####  Sigmoid片段示例

Sigmoid示例代码如下，其中x为输入的向量，y为输入的张量：
```
/* call Sigmoid function */
y = Sigmoid(*x);
```
有关Sigmoid的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TSigmoid.cpp

#### Softmax

##### 什么是Softmax？

Softmax是一种激活函数，Softmax函数定义为：
>y = e^x / \sum_{i} e^{x_i}

##### Softmax调用

NiuTrans.Tensor提供了张量的Softmax激活函数，调用方法及参数说明如下所示:
```
void _Softmax(const XTensor * x, XTensor * y, int leadDim)

XTensor Softmax(const XTensor &x, int leadDim)
```
Parameters:

* x - 输入张量
* y - 输出张量
* leadDim - 沿着指定维度进行操作

#####  Softmax片段示例

Softmax示例代码如下，其中x为输入的向量，y为输入的张量，本例中沿着维度1进行Softmax操作：
```
/* call Softmax function */
y = Softmax(*x, 1);
```
有关Softmax的详细代码示例见：

NiuTrans.Tensor/Tensor/test/TSoftmax.cpp

## 高级技巧

### 内存池

内存作为计算机软件运行过程中不可或缺的一项重要资源，在软件开发过程中具有十分重要的地位。对于一个软件系统而言，如何更高效地进行内存管理将对系统整体性能，尤其是运行速度方面产生很大程度的影响。虽然目前而言，主流编程语言均会为开发人员提供相应的系统级接口（如C语言中的malloc和free，C++中的new和delete等），但这类接口在设计的时候由于需要考虑各种使用情况，因此并不一定能够最适用于目前的使用需求（如对速度具有较高要求等），因此直接使用系统级的内存管理接口存在以下弊端：

1. 内存申请、释放时间消耗大：由于操作系统在进行内存管理的时候需要保证内存空间得到有效地使用，因此在执行内存申请或释放操作的时候，系统会对候选内存块进行一定程度的选择和合并，这些操作给相应的操作带来了许多额外的时间开销，导致频繁地对内存进行操作耗时较大。 
2. 程序执行效率低：由于所申请内存块的大小不定，当频繁使用系统级接口进行内存管理的时候容易在存储空间中产生大量内存碎片，拖慢系统的执行效率。
3. 易发生内存泄漏：使用系统级接口对内存空间进行申请的时候，一般来说需要程序开发人员显性地对空间进行释放，一旦疏忽将导致内存泄漏情况的发生，因此使用系统级接口进行内存管理需要谨慎对存储空间的使用情况进行分析，使用相关检测工具对内存泄漏情况进行有效地核查。

此外，当系统中存在对GPU设备上的显存空间进行管理的时候，申请、释放操作所产生的时间代价相对普通内存来说更大。不同于内存空间的申请，在申请或释放显存的时候需要对CPU正在执行的操作进行中断，交由GPU设备进行显存的操作，因此这部分产生的时间消耗远比内存申请来说大得多，最终导致频繁地对显存空间进行操作会更严重地拖慢系统整体的执行效率。

针对以上问题，本系统支持使用内存池（Memory Pool）来对系统中的存储空间（包括内存和显存）进行管理。内存池的概念主要是在对存储空间进行使用之前，预先从系统中申请一整块的空间，由程序自身（内存池）对这部分的空间进行管理。这样做的好处在于对存储空间的申请、释放等操作不需要对系统的相应接口进行频繁调用，降低了其中中断、搜寻最优块等操作的耗时，同时也不易产生内存碎片。此外，由于内存池的申请是一次性的操作，因此不会在系统全局产生大规模内存|泄漏的情况，对系统的稳定性会有所助益。

具体来说，想要在NiuTrans.Tensor的工具包中使用内存池（XMem）进行操作，只需要三个步骤：内存池的定义，使用以及释放。

* 内存池的定义

最简单的定义一个内存池只需指定一个设备ID即可，下面是一段示例代码。
```
// 定义一个内存池mem，它的类型是XMem
XMem * mem = new XMem(devID);
```
若需要更具体地指定内存池的信息，可以定义内存池的时候通过myMode、myBlockSize、myBlockNum、myBufSize等参数设置内存池的使用模型、内存块大小、内存块数量以及缓存区大小。

* 内存池的使用

在定义好内存池之后，我们即可在该空间上进行变量的定义及使用了，这里以张量的定义为例，下面是一段示例代码。
```
// 声明一个变量tensor，它的类型是XTensor
XTensor tensor;                         

// 在内存池上初始化这个变量为50列*100行的矩阵(2阶张量)      
InitTensor2D(&tensor, 50, 100, X_FLOAT, -1, mem);
```
我们可以看到，上述代码相对之前之前未使用内存池时的定义方式而言，仅需在定义的时候指定所使用的内存池即可，无需更复杂的操作。

* 内存池的释放
   
当希望将完全对内存池进行释放的时候，我们仅需直接对内存池进行删除即可，下面是一段示例代码。
```
// 删除内存池mem
delete mem;
```

## 实例1：矩阵乘法

这里我们给出一个矩阵乘法的例子，首先定义张量维度的大小，然后初始化两个维度分别为2*3和3*2的矩阵，使用SetData()方法对矩阵进行赋值，最后计算两个矩阵相乘。

关于矩阵乘法的详细代码请见NiuTrans.Tensor/Tensor/sample/mul/。

```
#include "mul.h"

namespace nts
{
void sampleMUL1()
{
    DTYPE aData[2][3] = { { 1.0F, 2.0F, 3.0F },
                          { -4.0F, 5.0F, 6.0F } };
    DTYPE bData[3][2] = { { 0.0F, -1.0F },
                          { 1.0F, 2.0F },
                          { 2.0F, 1.0F } };
    DTYPE answer[2][2] = { { 8.0F, 6.0F },
                           { 17.0F, 20.0F } };

    /* a source tensor of size (2, 3) */
    int aOrder = 2;
    int * aDimSize = new int[aOrder];
    aDimSize[0] = 2;
    aDimSize[1] = 3;

    int aUnitNum = 1;
    for (int i = 0; i < aOrder; i++)
        aUnitNum *= aDimSize[i];

    /* a source tensor of size (3, 2) */
    int bOrder = 2;
    int * bDimSize = new int[bOrder];
    bDimSize[0] = 3;
    bDimSize[1] = 2;

    int bUnitNum = 1;
    for (int i = 0; i < bOrder; i++)
        bUnitNum *= bDimSize[i];

    /* a target tensor of size (2, 2) */
    int resultOrder = 2;
    int * resultDimSize = new int[resultOrder];
    resultDimSize[0] = 2;
    resultDimSize[1] = 2;

    int resultUnitNum = 1;
    for (int i = 0; i < resultOrder; i++)
        resultUnitNum *= resultDimSize[i];

	/* create tensors */
    XTensor * a = NewTensor(aOrder, aDimSize);
    XTensor * b = NewTensor(bOrder, bDimSize);
    XTensor * result = NewTensor(resultOrder, resultDimSize);

	/* initialize variables */
    a->SetData(aData, aUnitNum);
    b->SetData(bData, bUnitNum);
    result->SetZeroAll();

	/* call MatrixMul function */
    _MatrixMul(a, X_NOTRANS, b, X_NOTRANS, result);

    result->Dump(stderr, "result:");

	/* destroy variables */
    delete[] aDimSize;
    delete[] bDimSize;
    delete[] resultDimSize;
    delete a;
    delete b;
    delete result;
}
```

## 实例2：前馈神经网络

下面我们来实现一个简单的前馈神经网络语言模型。

语言建模任务是通过某种方式对语言建立数学模型的过程。在神经网络出现之前，一般使用统计的方法来设计语言模型。比较常见的为n-gram模型，它对文本中若干词语共现的频率进行统计，并使用平滑算法对未见词语搭配进行修正，最终得到该语言中不同词语连续出现的概率值。神经语言模型相对传统基于统计的模型而言，能够在学习词语搭配的同时学习到词汇之间的相似性，相对平滑算法而言有效提高了对已知单词的未见搭配的预测效果，获得了更好的性能。

神经语言模型最早由Bengio等人系统化提出并进行了深入研究，其整体结构上和普通的前馈神经网络类似，由输入层、隐藏层和输出层组成，层和层之间存在连接，每一层将本层接收到的向量映射到另一维空间上作为该层的输出。

前馈神经网络语言模型的主要流程如下所示:

```
int FNNLMMain(int argc, const char ** argv)
{
    if(argc == 0)
        return 1;

    FNNModel model;

    /* load arguments */
    LoadArgs(argc, argv, model);

    /* check the setting */
    Check(model);

    /* initialize model parameters */
    Init(model);

    /* learn model parameters */
    if(strcmp(trainFN, ""))
        Train(trainFN, shuffled, model);

    /* save the final model */
    if(strcmp(modelFN, "") && strcmp(trainFN, ""))
        Dump(modelFN, model);

    /* load the model if neccessary */
    if(strcmp(modelFN, ""))
        Read(modelFN, model);

    /* test the model on the new data */
    if(strcmp(testFN, "") && strcmp(outputFN, ""))
        Test(testFN, outputFN, model);

    return 0;
}
```

对模型中的参数进行初始化：

```
/* initialize the model */
void Init(FNNModel &model)
{
    /* create embedding parameter matrix: vSize * eSize */
    InitModelTensor2D(model.embeddingW, model.vSize, model.eSize, model);
    
    /* create hidden layer parameter matrics */
    for(int i = 0; i < model.hDepth; i++){
        /* hidden layer parameter matrix: (n-1)eSize * hsize if it is the first layer
                                           hsize * hsize otherwise */
        if(i == 0)
            InitModelTensor2D(model.hiddenW[i], (model.n - 1) * model.eSize, model.hSize, model);
        else
            InitModelTensor2D(model.hiddenW[i], model.hSize, model.hSize, model);
        
        /* bias term: a row vector of hSize entries */
        InitModelTensor1D(model.hiddenB[i], model.hSize, model);
    }
    
    /* create the output layer parameter matrix and bias term */
    int iSize = model.hDepth == 0 ? (model.n - 1) * model.eSize : model.hSize;
    InitModelTensor2D(model.outputW, iSize, model.vSize, model);
    InitModelTensor1D(model.outputB, model.vSize, model);
    
    /* then, we initialize model parameters using a uniform distribution in range
       of [-minmax, minmax] */
    model.embeddingW.SetDataRand(-minmax, minmax);
    model.outputW.SetDataRand(-minmax, minmax);
    for(int i = 0; i < model.hDepth; i++)
        model.hiddenW[i].SetDataRand(-minmax, minmax);
    
    /* all bias terms are set to zero */
    model.outputB.SetZeroAll();
    for(int i = 0; i < model.hDepth; i++)
        model.hiddenB[i].SetZeroAll();
}
```

训练过程：

```
void Train(const char * train, bool isShuffled, FNNModel &model)
{
    char name[MAX_NAME_LENGTH];
    
    /* shuffle the data */
    if(isShuffled){
        sprintf(name, "%s-tmp", train);
        Shuffle(train, name);
    }
    else
        strcpy(name, train);
    
    int epoch = 0;
    int step = 0;
    int wordCount = 0;
    int wordCountTotal = 0;
    int ngramNum = 1;
    float loss = 0;
    bool isEnd = false;
    
    NGram * ngrams = new NGram[MAX_LINE_LENGTH_HERE];

    /* make a model to keep gradients */
    FNNModel grad;
    Copy(grad, model);

    /* XNet for automatic differentiation */
    XNet autoDiffer;

    double startT = GetClockSec();
    
    /* iterate for a number of epochs */
    for(epoch = 0; epoch < nEpoch; epoch++){

        /* data file */
        FILE * file = fopen(name, "rb");
        CheckErrors(file, "Cannot open the training file");

        wordCount = 0;
        loss = 0;
        ngramNum = 1;

        while(ngramNum > 0){
            
            /* load a minibatch of ngrams */
            ngramNum = LoadNGrams(file, model.n, ngrams, sentBatch, wordBatch);

            if (ngramNum <= 0)
                break;

            /* previous n - 1 words */
            XTensor inputs[MAX_N_GRAM];

            /* the predicted word */
            XTensor output;

            /* the gold standard */
            XTensor gold;

            /* make the input tensor for position i */
            for(int i = 0; i < model.n - 1; i++)
                MakeWordBatch(inputs[i], ngrams, ngramNum, i, model.vSize, model.devID, model.mem);

            /* make the gold tensor */
            MakeWordBatch(gold, ngrams, ngramNum, model.n - 1, model.vSize, model.devID, model.mem);

            if(!autoDiff){
                /* prepare an empty network for building the fnn */
                FNNNet net;

                /* gradident = 0 */
                Clear(grad);

                /* forward computation */
                Forward(inputs, output, model, net);

                /* backward computation to obtain gradients */
                Backward(inputs, output, gold, CROSSENTROPY, model, grad, net);

                /* update model parameters */
                Update(model, grad, learningRate, false);
            }
            else{
                /* forward + backward process */
                ForwardAutoDiff(inputs, output, model);

                /* automatic differentiation */
                autoDiffer.Backward(output, gold, CROSSENTROPY);

                /* update model parameters */
                Update(model, grad, learningRate, true);
            }
                
            /* get probabilities */
            float prob = GetProb(output, gold);
                
            loss += -prob;
            wordCount += ngramNum;
            wordCountTotal += ngramNum;
            
            if(++step >= nStep){
                isEnd = true;
                break;
            }

            if (step % 100 == 0) {
                double elapsed = GetClockSec() - startT;
                XPRINT5(0, stderr, "[INFO] elapsed=%.1fs, step=%d, epoch=%d, ngram=%d, ppl=%.3f\n",
                           elapsed, step, epoch + 1, wordCountTotal, exp(loss / wordCount));
            }
        }

        fclose(file);
        
        if(isEnd)
            break;
    }

    double elapsed = GetClockSec() - startT;
    
    XPRINT5(0, stderr, "[INFO] elapsed=%.1fs, step=%d, epoch=%d, ngram=%d, ppl=%.3f\n", 
               elapsed, step, epoch, wordCountTotal, exp(loss / wordCount));
    XPRINT3(0, stderr, "[INFO] training finished (took %.1fs, step=%d and epoch=%d)\n", 
               elapsed, step, epoch);
    
    delete[] ngrams;
}
```

在这里只介绍部分主要代码，详细代码请参见NiuTrans.Tensor/source/sample/FNNLM.cpp

前馈神经网络前向部分：经过数据处理之后我们得到了语言模型的输入（n-1个词），我们把输入input和输入层的权重w1（词向量）相乘得到每个输入单词的向量表示，公式如下：

>embedding = input * w1

最后将n-1个词的向量连接起来作为输入层最终的输出。

同理，我们将输入层的输出分别经过隐藏层和输出层得到最终的结果，公式如下：

>h = tanh(h_pre*w2+b)

>y = softmax(h_last*w3)

前向过程代码如下：

```
/*
forward procedure
>> inputs - input word representations
>> output - output probability
>> model - the fnn model
>> net - the network that keeps the internal tensors generated in the process
*/
void Forward(XTensor inputs[], XTensor &output, FNNModel &model, FNNNet &net)
{
    int batchSize = -1;
    int n = model.n;
    int depth = model.hDepth;
    XList eList(n - 1);

    /* previoius n - 1 words */
    for(int i = 0; i < n - 1; i++){
        XTensor &input = inputs[i];
        XTensor &w = model.embeddingW;
        XTensor &embedding = net.embeddings[i];

        if(batchSize == -1)
            batchSize = input.dimSize[0];
        else{
            CheckErrors(batchSize == input.dimSize[0], "Wrong input word representations!");
        }

        /* embedding output tensor of position i */
        InitModelTensor2D(embedding, batchSize, model.eSize, model);

        /* generate word embedding of position i:
           embedding = input * w   */
        _MatrixMul(&input, X_NOTRANS, &w, X_NOTRANS, &embedding);

        eList.Add(&net.embeddings[i]);
    }

    /* concatenate word embeddings
       embeddingcat = cat(embedding_0...embedding_{n-1}) */
    InitModelTensor2D(net.embeddingCat, batchSize, (n - 1) * model.eSize, model);
    _Concatenate(&eList, &net.embeddingCat, 1);

    /* go over each hidden layer */
    for(int i = 0; i < depth; i++){
        XTensor &h_pre = i == 0 ? net.embeddingCat : net.hiddens[i - 1];
        XTensor &w = model.hiddenW[i];
        XTensor &b = model.hiddenB[i];
        XTensor &h = net.hiddens[i];
        XTensor &s = net.hiddenStates[i];

        InitModelTensor2D(h, batchSize, model.hSize, model);
        InitModelTensor2D(s, batchSize, model.hSize, model);

        /* generate hidden states of layer i: 
           s = h_pre * w    */
        _MatrixMul(&h_pre, X_NOTRANS, &w, X_NOTRANS, &s);

        /* make a 2d tensor for the bias term */
        XTensor b2D;
        InitTensor(&b2D, &s);
        _Unsqueeze(&b, &b2D, 0, batchSize);

        /* introduce bias term:
           s = s + b
           NOTE: the trick here is to extend b to a 2d tensor
                 to fit into the 2d representation in tensor summation */
        _Sum(&s, &b2D, &s);

        /* pass the state through the hard tanh function:
           h = tanh(s) */
        _HardTanH(&s, &h);
    }

    /* generate the output Pr(w_{n-1}|w_0...w_{n-2}):
       y = softmax(h_last * w) 
       Note that this is the implementation as that in Bengio et al.' paper.
       TODO: we add bias term here */
    {
        XTensor &h_last = depth > 0 ? net.hiddens[depth - 1] : net.embeddingCat;
        XTensor &w = model.outputW;
        XTensor &b = model.outputB;
        XTensor &s = net.stateLast;
        XTensor &y = output;

        InitModelTensor2D(s, batchSize, model.vSize, model);
        InitModelTensor2D(y, batchSize, model.vSize, model);

        /* s = h_last * w  */
        _MatrixMul(&h_last, X_NOTRANS, &w, X_NOTRANS, &s);

        XTensor b2D;
        InitTensor(&b2D, &s);
        _Unsqueeze(&b, &b2D, 0, batchSize);

        _Sum(&s, &b2D, &s);

        /* y = softmax(s) */
        _LogSoftmax(&s, &y, 1);
    }   
}
```

反向部分：首先利用前向得到的最终结果和标准答案计算总的损失函数L，然后采用梯度下降的方法通过反向传播计算得到损失函数L对每层的参数w的导数∂L/∂w，之后我们根据

>w_(k+1)= w_k-η*  ∂L/(∂w_k )	

对参数W进行更新，其中η是学习率。

反向以及反向传播后的更新代码如下：

```
/*
backward procedure
>> inputs - input word representations
>> output - output probability
>> gold - gold standard
>> loss - loss function name
>> model - the fnn model
>> grad - the model that keeps the gradient information
>> net - the network that keeps the internal tensors generated in the process
*/
void Backward(XTensor inputs[], XTensor &output, XTensor &gold, LOSS_FUNCTION_NAME loss, 
              FNNModel &model,  FNNModel &grad, FNNNet &net)
{
    int batchSize = output.GetDim(0);
    int n = model.n;
    int depth = model.hDepth;

    /* back-propagation for the output layer */
    XTensor &y = output;
    XTensor &s = net.stateLast;
    XTensor &x = depth > 0 ? net.hiddens[depth - 1] : net.embeddingCat;
    XTensor &w = model.outputW;
    XTensor &dedw = grad.outputW;
    XTensor &dedb = grad.outputB;
    XTensor deds(&y);
    XTensor dedx(&x);

    /* for y = softmax(s), we get dE/ds
        where E is the error function (define by loss) */
    _LogSoftmaxBackward(&gold, &y, &s, NULL, &deds, 1, loss);

    /* for s = x * w, we get 
       dE/w_{i,j} = dE/ds_j * ds/dw_{i,j} 
                  = dE/ds_j * x_{i}
       (where i and j are the row and column indices, and
        x is the top most hidden layer)
       so we know 
       dE/dw = x^T * dE/ds */
    _MatrixMul(&x, X_TRANS, &deds, X_NOTRANS, &dedw);

    /* gradient of the bias: dE/db = dE/ds * 1 = dE/ds
    specifically dE/db_{j} = \sum_{i} dE/ds_{i,j} */
    _ReduceSum(&deds, &dedb, 0);

    /* then, we compute 
       dE/dx_{j} = \sum_j' (dE/ds_{j'} * ds_{j'}/dx_j) 
                 = \sum_j' (dE/ds_{j'} * w_{j, j'})
       i.e., 
       dE/dx = dE/ds * w^T */
    _MatrixMul(&deds, X_NOTRANS, &w, X_TRANS, &dedx);

    XTensor &gradPassed = dedx;
    XTensor dedsHidden;
    XTensor dedxBottom;
    if (depth > 0)
        InitTensor(&dedsHidden, &dedx);
    InitTensor(&dedxBottom, &net.embeddingCat);

    /* back-propagation from top to bottom in the stack of hidden layers
       for each layer, h = f(s)
                       s = x * w + b */
    for (int i = depth - 1; i >= 0; i--) {
        XTensor &h = net.hiddens[i];
        XTensor &s = net.hiddenStates[i];
        XTensor &x = i == 0 ? net.embeddingCat : net.hiddenStates[i - 1];
        XTensor &w = model.hiddenW[i];
        XTensor &dedh = gradPassed;  // gradient passed though the previous layer
        XTensor &dedx = i == 0 ? dedxBottom : dedh;
        XTensor &deds = dedsHidden;
        XTensor &dedw = grad.hiddenW[i];
        XTensor &dedb = grad.hiddenB[i];
        
        /* backpropagation through the activation fucntion: 
           dE/ds = dE/dh * dh/ds */
        _HardTanHBackward(NULL, &h, &s, &dedh, &deds, NOLOSS);

        /* gradient of the weight: dE/dw = x^T * dE/ds   */
        _MatrixMul(&x, X_TRANS, &deds, X_NOTRANS, &dedw);

        /* gradient of the bias: dE/db = dE/ds * 1 = dE/ds
           specifically dE/db_{j} = \sum_{i} dE/ds_{i,j} */
        _ReduceSum(&deds, &dedb, 0);

        /* gradient of the input: dE/dx = dE/ds * w^T    */
        _MatrixMul(&deds, X_NOTRANS, &w, X_TRANS, &dedx);

        if (i > 0)
            _CopyValues(&dedx, &gradPassed);
    }

    XList eList(n - 1);

    /* back-propagation for the embedding layer */
    for (int i = 0; i < n - 1; i++) {
        XTensor * dedy = NewTensor2D(batchSize, model.eSize, X_FLOAT, model.devID, model.mem);
        eList.Add(dedy);
    }

    /* gradient of the concatenation of the embedding layers */
    XTensor &dedyCat = depth > 0 ? dedxBottom : dedx;

    /* split the concatenation of gradients of the embeddings */
    _Split(&dedyCat, &eList, 1, n - 1);

    /* go over for each word */
    for (int i = 0; i < n - 1; i++) {
        XTensor * dedy = (XTensor*)eList.GetItem(i);
        XTensor &x = inputs[i];
        XTensor &dedw = grad.embeddingW;

        /* gradient of the embedding weight: dE/dw += x^T * dE/dy 
           NOTE that we accumulate dE/dw here because the matrix w
           is shared by several layers (or words) */
        _MatrixMul(&x, X_TRANS, dedy, X_NOTRANS, &dedw, 1.0F, 1.0F);

        delete dedy;
    }
}
```

## 实例3：循环神经网络

## NiuTrans.Tensor团队

* 肖桐
* 李垠桥
* 许晨
* 姜雨帆
* 林野
* 张裕浩
* 胡驰

## 附录

在XTensor.h头文件中定义的成员变量说明：

| 成员变量 | 功能 |
| - | - |
| int id | 张量标识 |
| XMem * mem | 张量所使用的内存池 |
| void * data | 保存元素的数据数组 |
| void * dataHost | 主机内存上的数据副本，只在GPU上运行时被激活 |
| void ** dataP | 指向数据地址的指针 |
| int devID | 设备ID，指张量所申请的空间所在CPU或者GPU设备的编号，-1表示CPU |
| int order | 张量的维度，例如：一个矩阵（维度为2）是一个二维张量 |
| int dimSize[ ] | 张量中每一维度的大小，索引0表示第1维 |
| int dimSizeRDI[ ] | 转置模式下张量中每一维度的大小，索引0表示第1维 |
| TENSOR_DATA_TYPE dataType | 每个数据单元的数据类型 |
| int unitSize | 数据单元的大小，类似于sizeof() |
| int unitNum | 数据单元的数量 |
| bool isSparse | 是否稠密，一个n * m稠密矩阵的数据量大小为n * m,而稀疏（非稠密）矩阵的数据量大小则取决于矩阵中非零元素个数。|
| int unitNumNonZero | 稀疏矩阵中非零元素个数 |
| float denseRatio | 稠密度，指非零单元的比例，是介于0和1之间的一个实数，0表示所有单元全为零，1表示全为非零单元。|
| bool isShared | 标志数据数组是否被其他张量所共享 |
| bool isDefaultDType | 矩阵中使用的数据类型是否是属于默认数据类型 |
| bool isInGlobalMem | 标志数据是否在全局内存而不是内存池中 |
| bool isAllValued[ ] | 标志稀疏矩阵中是否每个维度都具有非零元素 |
| bool isInit | 张量是否被初始化 |
| bool isTmp | 张量是否为临时创建 |
| bool isGrad | 当使用模型参数时张量是否保持梯度 |
| unsigned int visitMark | 节点访问标志 |
| XTensor * grad | 反向传播的梯度 |
| XLink income | 超边的入边 |
| XLink outgo | 超边的出边 |

在XTensor.h头文件中定义的方法说明：

| 功能 | 函数  | 参数 |
| - | - | - |
| 构造函数 | XTensor() | N/A |
| 析构函数 | ~XTensor() | N/A |
| 初始化成员变量 | void Init() | N/A |
| 销毁数据 | void DestroyData() | N/A |
| 张量的浅层复制 | void ShallowCopy(<br>const XTensor &tensor) | tensor - 进行复制的张量 |
| 重载等于符号 | XTensor& operator= (<br>const XTensor &tensor) | tensor - 重载的张量 |
| 重载加法符号 | XTensor  operator+ (<br>const XTensor &tensor) | tensor - 重载的张量 |
| 重载乘法符号 | XTensor  operator* (<br>const XTensor &tensor) | tensor - 重载的张量 |
| 线性变换 | XTensor Lin(<br>DTYPE scale, DTYPE shift = 0) | scale - 缩放参数 <br> shift - 偏移参数 |
| 判断两个张量数据类型<br>和大小是否相同 | static bool IsIdentical(<br> XTensor * a, XTensor * b) | a - 进行比较的第一个张量 <br> b - 进行比较的第二个张量 |
| 判断三个张量数据类型<br>和大小是否相同 | static bool IsIdentical(<br> XTensor * a, XTensor * b, XTensor * c) | a - 进行比较的第一个张量 <br> b - 进行比较的第二个张量 <br> c - 进行比较的第三个张量 |
| 设置张量每一维度的大小 | void SetDim(<br>int * myDimSize) |myDimSize - 张量每一维度的大小 |
| 得到张量中给定的维度大小 | int GetDim(<br>const int dim) | dim - 张量的维度 |
| 重新调整矩阵维度 | void Reshape(<br> const int order, const int * myDimSize) | order - 张量的维度 <br> myDimSize - 张量每一维的大小 |
| 得到张量中元素数量 | int GetSize() | N/A |
| 得到内存使用大小 | int GetDataSizeInChar() | N/A |
| 得到所给数据类型的数据<br> 单元大小 | int GetUnitSize(<br> TENSOR_DATA_TYPE myDataType) | myDataType - 所给数据类型 |
| 张量中所有元素设置为0 | void SetZeroAll(XStream * stream = NULL) | stream - 多线程流|
| 用数组赋值张量 | void SetData(<br> const void * d, int num, int beg = 0) | d - 赋值数组  <br> num - 数组大小 <br> beg - 赋值时从张量的第几位开始 |
| 设置张量服从均匀分布 | void SetDataRand(<br> DTYPE lower, DTYPE upper) | lower - 最小值 <br> upper - 最大值 |
| 设置张量服从正态分布 | void SetDataRandn(<br> DTYPE mean, DTYPE standardDeviation) | mean - 均值 <br> standardDeviation - 标准差 |
| 检查张量中元素是否相同 | bool CheckData(<br> const void * answer, int num, int beg = 0) | answer - 给定数组 <br> num - 数组大小 <br> beg - 赋值时从张量的第几位开始 |
| 设置数据指针 | void SetDataPointer() | N/A |
| 将给定维度中元素<br> 设置为升序 | void SetAscendingOrder(<br>int dim) | dim - 给定维度 |
| 得到索引指向的单元的值 | DTYPE Get(int index[], int size = -1) | index - 给定索引 <br> size-矩阵大小 |
| 获取张量中元素指针 | void * GetCell(<br>int * index, int size)    | index - 元素位置 <br> size-矩阵大小 |
| 获取一维张量中元素的<br>默认类型值 | DTYPE Get1D(<br>int i) | i - 第一维 |
| 获取二维张量中元素的<br>默认类型值 | DTYPE Get2D(<br>int ni, int mi) const | ni - 第一维 <br> mi - 第二维 |
| 获取三维张量中元素的<br>默认类型值 | DTYPE Get3D(<br>int d0, int d1, int d2) | d0 - 第一维 <br> d1 - 第二维 <br> d2 - 第三维 |
| 获取一维张量中元素的<br>整形值 |int Get1DInt(<br>int i) | i - 第一维 |
| 获取二维张量中元素的<br>整形值 | int Get2DInt(<br>int ni, int mi) | ni - 第一维 <br> mi - 第二维 |
| 获取三维张量中元素的整形值 | int Get3DInt(<br>int d0, int d1, int d2) | d0 - 第一维 <br> d1 - 第二维 <br> d2 - 第三维 |
| 获取稀疏张量的值 | DTYPE GetInSparse(int i) | i - 稀疏矩阵中非0元素位置 |
| 获取稀疏张量中<br> 元组的键值 | int GetKeyInSparse(int i) | i - 稀疏矩阵中非0元素位置 |
| 设置单元中的值 | bool Set(<br>DTYPE value, int index[], int size = -1) | value - 值 <br> index - 元素位置 <br> size-矩阵大小 |
| 设置一维张量中的单元值 | bool Set1D(<br>DTYPE value, int i) | value - 值 <br> i - 第一维 |
| 设置二维张量中的单元值 | bool Set2D(<br>DTYPE value, int ni, int mi) | value - 值 <br> ni - 第一维 <br> mi - 第二维 |
| 设置三维张量中的单元值 | bool Set3D(<br>DTYPE value, int d0, int d1, int d2) | value - 值 <br> d0 - 第一维 <br> d1 - 第二维 <br> d2 - 第三维 |
| 增加二维张量中<br> 的单元值 | bool Add2D(<br>DTYPE value, int ni, int mi = 0) | value - 单元值 <br> ni - 行值 <br> mi - 列值 |
| 获取稀疏矩阵中<br> 非零元素数量 | int GetNonzeroSize() | N/A |
| 设置张量为临时变量 | void SetTMP(<br>bool myIsTmp = true) | myIsTmp - 是否为临时变量 |
| 张量是否保持梯度 | void SetGrad(<br>bool myIsGrad = true) | myIsTmp - 是否保持梯度 |
| 将矩阵重置为特定大小 | bool Resize(<br> const int myOrder, <br> const int * myDimSize, <br> const TENSOR_DATA_TYPE myDataType = DEFAULT_DTYPE, <br> const float myDenseRatio = 1.0F) | myOrder - 张量的维度 <br> myDimSize - 张量每一维的大小,索引0表示第一维 <br> myDataType - 张量的数据类型 <br> myDenseRatio - 张量的稠密度，1表示稠密张量 |
| 将矩阵重置为特定大小<br>并不申请新空间 | bool ResizeWithNoData(<br> const int myOrder, <br> const int * myDimSize, <br> const TENSOR_DATA_TYPE myDataType = DEFAULT_DTYPE, <br> const float myDenseRatio = 1.0F) | myOrder - 张量的维度 <br> myDimSize - 张量每一维的大小,索引0表示第一维 <br> myDataType - 张量的数据类型 <br> myDenseRatio - 张量的稠密度，1表示稠密张量 |
| 将矩阵重置为<br> 另一矩阵大小 | bool Resize(<br> const XTensor * myTensor) | myTensor - 重置矩阵大小的参考矩阵 |
| 用二值搜索方法<br> 找到稀疏矩阵中元素 | bool BinarySearch(<br> int key, DTYPE &value, void * &position) | key - 稀疏矩阵中元素位置 <br> value - 元素值 <br> position - 元素坐标位置 |
| 将数据刷新到<br> 目标设备中 | void FlushToMem(XMem * targetMem) | targetMem - 目标设备 |
| 在全局内存中<br> 申请矩阵的内存空间 | static void AllocateData(<br> XTensor * matrix, <br> XMem * myMem = NULL, <br> bool useBuf = false) | matrix - 申请内存空间的矩阵 <br> myMem - 是否在内存池中申请空间 <br> useBuf - 是否使用缓冲区 |
| 在全局内存中<br> 释放矩阵的内存空间 | static void FreeData(<br> XTensor * matrix, <br> XMem * myMem = NULL, <br> bool useBuf = false) | matrix - 申请内存空间的矩阵 <br> myMem - 是否在内存池中申请空间 <br> useBuf - 是否使用缓冲区 |
| 在缓冲区创建张量 | XTensor * NewTensorBuf( <br> const int myOrder,  <br> const int * myDimSize, XMem * myMem, <br> const TENSOR_DATA_TYPE myDataType = <br> X_FLOAT, const float myDenseRatio = 1.0F) | myOrder - 张量的维度 <br> myDimSize - 张量每一维的大小,索引0表示第一维 <br> myMem - 张量所使用的内存池 <br>  myDataType - 张量的数据类型 <br> myDenseRatio - 张量的稠密度，1表示稠密张量 |
| 依据给定张量<br>复制一个新的张量 | XTensor * NewTensor(<br>XTensor * a, bool isFilledData = true) | a - 给定张量 <br>  isFilledData - 是否申请张量中的数据空间 |
| 依据给定张量<br>释放数据空间 | void DelTensor(<br>const XTensor * tensor) | tensor - 给定张量 |
| 依据给定张量<br>在缓存中释放数据空间 | void DelTensorBuf(<br>const XTensor * tensor) | tensor - 给定张量 |