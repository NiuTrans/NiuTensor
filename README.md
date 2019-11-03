# NiuTensor张量计算库

## NiuTensor

NiuTensor是小牛开源项目所开发的一个轻量级工具包，提供了完整的张量定义及计算功能，可以被用于深度学习相关研究及工业系统的开发。NiuTensor具有以下特点：

* 简单小巧，易于修改
* c语言编写，代码高度优化
* 同时支持CPU和GPU设备
* 丰富的张量计算接口
* 支持C/C++调用方式

## 安装方法

在开始创建您的项目并使用NiuTensor工具包时，需要注意的是：

* 所创建项目如在CPU上运行，我们的系统支持高性能的数学运算库，推荐安装[MKL](https://software.intel.com/en-us/mkl)或[OpenBLAS](http://www.openblas.net/)。
* 所创建项目如需在GPU上运行，需安装 [CUDA](https://developer.nvidia.com/cuda-downloads)，CUDA版本需求为9.0及以上，CUDA工具为创建高性能GPU加速应用程序提供了开发环境。

小牛开源项目所开发的NiuTensor工具包采用源程序编译方法，在Windows和Linux环境下的安装方法如下所示。

### Windows

若在Windows上使用NiuTensor工具包：

* 首先需要将NiuTensor代码包含在所创建的项目中
* 在所创建项目中需要引用XTensor.h、core里的CHeader.h和function里的FHeader.h这三个头文件：
    * 通过XTensor.h可以获取我们需要操作的XTensor类
    * 通过core里的CHeader.h可以对Tensor进行一些张量运算
    * 通过function里的FHeader.h可以调用一些激活函数
* 在所创建项目中使用命名空间nts

此外，一些必须的环境配置方法请参考 [NiuTensor环境配置](https://github.com/NiuTrans/NiuTensor/blob/master/Configuration.md)。

### Linux

若在Linux上使用NiuTensor工具包，直接执行make即可在bin目录下生成NiuTensor.CPU或NiuTensor.GPU，分别对应于NiuTensor的CPU以及GPU的可执行文件，同时在lib目录下生成相应的动态链接库。以测试为例，输入以下命令即可在GPU上执行提供的测试用例：
>./bin/NiuTensor.GPU -test

注意：若先生成CPU的可执行文件，之后如需生成GPU可执行文件，需要先执行make clean命令，删除生成CPU时产生的中间结果

更多详细使用方法请见[NiuTensor开发文档](http://niutrans.com/openSource/niutensor/index.html)


## 开发团队

NiuTensor张量计算库由东北大学自然语言处理实验室小牛开源团队开发，致力于为深度学习相关研究及工业系统的开发提供轻量级的张量定义及计算功能。

## 更新版本

NiuTensor version 0.1.1 - 2019年11月3日
