# NiuTrans.Tensor张量计算库

## NiuTrans.Tensor

NiuTrans.Tensor是小牛开源项目所开发的一个工具包，提供了完整的张量定义及计算功能，可以被用于深度学习相关研究及工业系统的开发。NiuTrans.Tensor具有以下特点：

* 简单小巧，易于修改
* c语言编写，代码高度优化
* 同时支持CPU和GPU设备
* 丰富的张量计算接口
* 支持C/C++调用方式

## 安装方法

在开始创建您的项目并使用NiuTrans.Tensor工具包时，需要注意的是：

* 所创建项目如在CPU上运行，我们的系统支持高性能的数学运算库，推荐安装[MKL](https://software.intel.com/en-us/mkl)或[OpenBLAS](http://www.openblas.net/)。
* 所创建项目如需在GPU上运行，需安装 [CUDA](https://developer.nvidia.com/cuda-downloads)，CUDA版本需求为9.0及以上，CUDA工具为创建高性能GPU加速应用程序提供了开发环境。

小牛开源项目所开发的NiuTrans.Tensor工具包采用源程序编译方法，在Windows和Linux环境下的安装方法如下所示。

### Windows

若在Windows上使用NiuTrans.Tensor工具包：

* 首先需要将NiuTrans.Tensor代码包含在所创建的项目中
* 在所创建项目中需要引用XTensor.h、core里的CHeader.h和function里的FHeader.h这三个头文件：
    * 通过XTensor.h可以获取我们需要操作的XTensor类
    * 通过core里的CHeader.h可以对Tensor进行一些张量运算
    * 通过function里的FHeader.h可以调用一些激活函数
* 在所创建项目中使用命名空间nts

此外，一些必须的环境配置方法请参考 [NiuTrans.Tensor环境配置](http://47.105.50.196/NiuTrans/NiuTrans.Tensor/blob/master/doc/Configuration.md)。

### Linux

若在Linux上使用NiuTrans.Tensor工具包，直接执行make.sh即可在同级目录下生成tensorCPU和tensorGPU，分别对应于NiuTrans.Tensor的CPU以及GPU的可执行文件。以前馈神经网络语言模型为例，输入以下命令即可在GPU上执行提供的测试用例：
>./tensorGPU -test

更多详细使用方法请见[NiuTrans.Tensor开发文档](http://47.104.97.237/niutrans/site/niutensor/index.html)


## 开发团队

NiuTrans.Tensor张量计算库由东北大学自然语言处理实验室小牛开源团队开发，致力于为深度学习相关研究及工业系统的开发提供完整的张量定义及计算功能。

## 更新版本

NiuTrans.Tensor version 0.1.0 - 2018年8月3日
