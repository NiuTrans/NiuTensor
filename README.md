# NiuTensor张量计算库

## NiuTensor

NiuTensor是小牛开源项目所开发的一个轻量级工具包，提供了完整的张量定义及计算功能，可以被用于深度学习相关研究及工业系统的开发。NiuTensor具有以下特点：

* 简单小巧，易于修改
* C语言编写，代码高度优化
* 同时支持CPU和GPU设备
* 丰富的张量计算接口
* 支持C/C++调用方式

## 开发文档

更多详细使用方法请见[NiuTensor开发文档](http://opensource.niutrans.com/niutensor/index.html)，包括：
* 张量计算的调用接口及说明
* 示例程序（如语言模型、机器翻译等）

## 安装方法

NiuTensor工具包的安装方法目前支持CMake（跨平台：支持Windows、Linux以及macOS）、Visual Studio项目（Windows平台）以及Makefile（Linux以及macOS平台）三种编译方式，这里推荐使用CMake对工具包进行安装。

在开始创建您的项目并使用NiuTensor工具包时，需要**注意**的是：

* 所创建项目如在CPU上运行，我们的系统支持高性能的数学运算库，推荐安装[MKL](https://software.intel.com/en-us/mkl)或[OpenBLAS](http://www.openblas.net/)（目前CMake方式不支持MKL和OpenBLAS，如希望使用上述运算库，建议使用Visual Studio或Makefile的方式进行编译，后续CMake将提供对其的完整支持）。
* 所创建项目如需在GPU上运行，需安装 [CUDA](https://developer.nvidia.com/cuda-downloads)，CUDA版本需求为9.0及以上，CUDA工具为创建高性能GPU加速应用程序提供了开发环境。

### 编译文件的修改

小牛开源项目所开发的NiuTensor工具包采用源程序编译方法，除在Windows平台手动配置Visual Studio项目外，CMake以及Makefile两种方式均需要针对不同平台对编译文件进行简单的修改，操作方式如下：

#### CMake

在Windows平台使用CMake生成Visual Studio的项目、在Linux或macOS平台使用CMake配置CLion项目以及使用CMake工具通过命令行的方式进行编译安装均需要对NiuTensor工具包根目录的CMakeLists.txt文件进行少量修改，具体步骤如下：

- 打开CMakeLists.txt文件对其进行编辑。
- 操作系统设置：若NiuTensor编译环境为Windows，则在`set(ON_WINDOWS 0)`中将`ON_WINDOWS`的值置为1；若编译环境为Linux或macOS，则将`ON_WINDOWS`的值置为0。
- 编译设备设置：若希望在CPU环境下编译使用NiuTensor工具包，则将`set(USE_CUDA 0)`中的`USE_CUDA`置为0即可；若希望在GPU环境下使用，则需将`USE_CUDA`置为1，同时在`set(CUDA_VERSION 9.0)`中设置CUDA版本号，在`set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")`中设置CUDA工具安装的根目录。

#### Makefile

在Linux或macOS平台可以使用Makefile工具通过命令行的方式进行编译安装，这种方式需要对NiuTensor工具包根目录的Makefile文件进行少量修改，具体步骤如下：

- 打开Makefile文件对其进行编辑。
- 操作系统设置：若NiuTensor编译环境为Windows或Linux，则在`OnMac = 0`中将`OnMac`的值置为0；若编译环境为macOS，则将`OnMac`的值置为1。
- 编译设备设置：若希望在CPU环境下编译使用NiuTensor工具包，则将`USE_CUDA = 0`中的`USE_CUDA`置为0即可；若希望在GPU环境下使用，则需将`USE_CUDA`置为1，同时在`CUDA_ROOT = /usr/local/cuda-9.0`中设置CUDA工具安装的根目录。
- 编译内容设置：若希望生成NiuTensor工具包的动态链接库，则将`dll = 0`中的`dll`置为1即可；若无需编译动态链接库，则将其置为0。

### 工具包的安装

在Windows、Linux以及macOS环境下的NiuTensor工具包的安装方法如下。

#### Windows

对于WIndows平台下NiuTensor工具包的使用，这里推荐通过Visual Studio集成开发环境对项目进行管理，项目配置提供CMake自动生成以及手动配置两种方式，开发人员可任选其一。

##### CMake方式（Visual Studio）

对于WIndows平台的NiuTensor工具包安装，这里可以使用CMake工具自动生成Visual Studio项目（需要用户提前安装CMake工具以及Visual Studio集成开发环境），操作步骤如下：

- 参考前述”编译文件的修改“章节，根据系统自身情况对工具包根目录中的CMakeLists.txt文件进行修改。
- 在工具包根目录新建目录以保存生成的Visual Studio项目文件（如建立build目录）。
- 在项目根目录打开Windows平台的命令行工具（如PowerShell），执行`cd build`命令进入新建的build目录。
- 执行`cmake ..`命令对Visual Studio项目进行生成（如果 visual studio 版本低于 2019，则使用命令`cmake -A x64 ..`），执行成功将显示`Build files have been written to:...`。
- 打开build目录中的NiuTensor.sln文件即可通过Visual Studio打开NiuTensor项目。
- 打开后在解决方案管理器中选中NiuTensor.CPU或NiuTensor.GPU，右键将其设为启动项目即可开始使用。

##### 手动配置方式

同时我们也支持用户根据自身项目需求对Visual Studio项目进行手动配置，在手动配置Visual Studio项目的过程中，一些必须的环境配置方法请参考 [通过Visual Studio手动配置NiuTensor项目](https://github.com/NiuTrans/NiuTensor/blob/master/doc/Configuration.md)。

#### Linux和macOS

对于Linux和macOS平台下NiuTensor工具包的使用，这里提供三种使用方式进行项目管理，分别为基于CMake的CLion集成开发环境、CMake工具（命令行）以及Makefile（命令行）的方式，开发人员可任选其一。

##### CMake方式（CLion）

对于Linux或macOS平台的NiuTensor工具包安装，CLion集成开发环境可以通过对CMakeLists.txt文件进行解析自动获取项目信息（需要用户提前安装CMake工具以及CLion集成开发环境），操作步骤如下：

- 参考前述”编译文件的修改“章节，根据系统自身情况对工具包根目录中的CMakeLists.txt文件进行修改。
- 使用CLion打开NiuTensor项目所在目录即可正常使用（确保CMakeLists.txt在其根目录位置），CLion将根据CMakeLists.txt文件自动读取项目信息。

##### CMake方式（命令行）

若仅需通过命令行方式对项目进行管理，开发者同样可以使用CMake快速对NiuTensor项目进行编译安装（需要用户提前安装CMake工具），操作步骤如下：

- 参考前述”编译文件的修改“章节，根据系统自身情况对工具包根目录中的CMakeLists.txt文件进行修改。
- 在项目根目录打开Linux或macOS平台的命令行工具（如Terminal），在工具包内新建目录以保存生成的中间文件（如执行`mkdir build`建立build目录）。
- 执行`cd build`命令进入新建的build目录。
- 执行`cmake ..`命令对项目进行生成，执行成功将显示`Build files have been written to:...`并在该目录下生成Makefile文件。
- 执行`make`命令对NiuTensor项目进行编译，执行成功将显示`Built target NiuTensor.CPU`或`Built target NiuTensor.GPU`，安装完毕。

完成安装后将在bin目录下生成NiuTensor.CPU或NiuTensor.GPU。

##### Makefile方式（命令行）

除CMake方式外，NiuTensor开源工具包同样提供直接通过Makefile对项目进行编译安装的方式，操作步骤如下：

- 参考前述”编译文件的修改“章节，根据系统自身情况对工具包根目录中的Makefile文件进行修改。
- 在项目根目录打开Linux或macOS平台的命令行工具（如Terminal）。
- 执行`make`命令对NiuTensor项目进行编译，执行成功将显示`Building executable file: ./bin/NiuTensor.CPU`或`Building executable file: ./bin/NiuTensor.GPU`，安装完毕。

完成安装后将在bin目录下生成NiuTensor.CPU或NiuTensor.GPU，或在lib目录下生成相应的动态链接库。

环境配置完成后，以CPU为例，可以使用`NiuTensor.CPU -test`命令运行本项目的测试用例，如果最后输出

```shell
OK! Everything is good!
```

则说明本项目配置成功。

>注意：
>
>1. 若先生成CPU的可执行文件，之后如需生成GPU可执行文件，需要先执行make clean命令，删除生成CPU时产生的中间结果，反之亦然。
>2. 在执行make命令时，可以使用-j参数来指定编译过程中使用的线程数量，该操作将显著加快编译速度，如：`make -j8`。

### 将工具包嵌入到自己的项目中

若希望在自己的项目中使用NiuTensor工具包，这里建议将工具包编译为动态链接库进行使用，编译库的方式可以参考前文所述（目前版本仅支持通过Makefile的方式生成Linux下的动态链接库，CMake方式以及更多的平台将在后续进行支持），而对于库文件的使用规则如下：

* 在所创建项目中需要引用XTensor.h、core里的CHeader.h和function里的FHeader.h这三个头文件：
  * 通过XTensor.h可以获取我们需要操作的XTensor类。
  * 通过core里的CHeader.h可以对Tensor进行一些张量运算。
  * 通过function里的FHeader.h可以调用一些激活函数。
* 在所创建项目中使用命名空间nts。

编译过程以编译好的CPU的动态链接库为例，假设动态链接库所在的路径为当前目录的lib目录，则通过以下命令可以编译生成可执行文件：

```shell
g++ -o MyTest Main.cpp -L./lib -lNiuTensor.CPU -std=c++11
```

然后将lib目录加入到环境变量中，可以通过以下命令进行临时修改（退出shell后则失效）：

```shell
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
```

或者将上述命令加入到~/.bashrc配置文件中进行永久修改（需要将相对路径./lib转换为绝对路径），并执行以下这条命令使修改立即生效：

```shell
source ~/.bashrc
```

之后就可以运行MyTest可执行文件。

```shell
./MyTest
```


## 开发团队

NiuTensor张量计算库由东北大学自然语言处理实验室小牛开源团队开发，致力于为深度学习相关研究及工业系统的开发提供轻量级的张量定义及计算功能。

## 更新版本

NiuTensor version 0.2.2 - 2020年4月14日
