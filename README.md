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

NiuTensor工具包的安装方式为CMake（跨平台：支持Windows、Linux以及macOS），集成开发环境支持Visual Studio（Windows平台）以及CLion（Linux和macOS平台）。

NiuTensor工具包的第三方运算加速库支持：

* 所创建项目如在CPU上运行，我们的系统支持高性能的数学运算库，推荐安装[MKL](https://software.intel.com/en-us/mkl)或[OpenBLAS](http://www.openblas.net/)。
* 所创建项目如需在GPU上运行，需安装 [CUDA](https://developer.nvidia.com/cuda-downloads)，CUDA版本需求为9.2及以上，CUDA工具为创建高性能GPU加速应用程序提供了开发环境。

> 注意：
>
> 对于第三方运算库的支持中，由于MKL和OpenBLAS二者所实现的API互有重叠，因此不支持同时使用。

### 工具包的安装

NiuTensor工具包可以在Windows、Linux以及macOS环境下进行安装，支持生成可执行程序以及动态链接库两种目标，具体安装方法如下。

#### Windows

对于Windows平台下NiuTensor工具包的使用，这里推荐通过Visual Studio集成开发环境对项目进行管理，项目配置使用CMake自动生成。

##### CMake方式（Visual Studio）

对于WIndows平台的NiuTensor工具包安装，这里可以使用CMake工具自动生成Visual Studio项目（需要用户提前安装CMake工具以及Visual Studio集成开发环境），操作步骤如下：

- 在工具包根目录新建目录以保存生成的Visual Studio项目文件（如建立build目录）。
- 在项目根目录打开Windows平台的命令行工具（如PowerShell），执行`cd build`命令进入新建的build目录。
- 执行CMake命令对Visual Studio项目进行生成（如果 visual studio 版本低于 2019，则在使用下列命令的时候需额外加上`-A x64`的CMake参数），如计划生成动态链接库，则仅需在命令中额外加上`-DGEN_DLL=ON`的CMake参数即可，否则默认生成可执行程序。
  - 如项目计划启用MKL数学运算库（需用户自行安装），则仅需在CMake命令中使用`-DUSE_MKL=ON`参数，并通过`-DINTEL_ROOT='/intel/root/path'`指定MKL库（Intel工具包）的安装路径。如`cmake -DUSE_MKL=ON -DINTEL_ROOT='C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2020.2.254/windows' ..`。
  - 如项目计划启用OpenBLAS数学运算库（需用户自行安装），则仅需在CMake命令中使用`-DUSE_OPENBLAS=ON`参数，并通过`-DOPENBLAS_ROOT='/openblas/root/path'`指定OpenBLAS库的安装路径。如`cmake -DUSE_OPENBLAS=ON -DOPENBLAS_ROOT='C:/Program Files/OpenBLAS' ..`。
  - 如项目计划启用CUDA数学运算库（需用户自行安装），则仅需在CMake命令中使用`-DUSE_CUDA=ON`参数，并通过`-DCUDA_ROOT='/cuda/root/path'`指定CUDA库的安装路径。如`cmake -DUSE_CUDA=ON -DCUDA_ROOT='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.2' ..`。如需在GPU设备上使用半精度浮点数进行运算，需在启用`-DUSE_CUDA=ON`参数的同时启用`-USE_HALF_PRECISION=ON`参数（需要注意的是半精度但需要注意的是，半精度操作仅在使用Pascal及更新架构的NVIDIA GPU中提供支持，该项可参考[NVIDIA GPU设备信息](https://developer.nvidia.com/cuda-gpus)进行查询）。
- 执行成功将显示`Build files have been written to:...`。
- 打开build目录中的NiuTensor.sln文件即可通过Visual Studio打开NiuTensor项目。
- 打开后在解决方案管理器中选中NiuTensor，右键将其设为启动项目即可开始使用。

> 注意：
>
> 动态链接库在编译结束后并不能直接运行，生成的库文件将保存在项目根目录的lib路径下，用户可根据自身需求将其嵌入到自己的项目中进行使用。

#### Linux和macOS

对于Linux和macOS平台下NiuTensor工具包的使用，这里提供两种使用方式进行项目管理，分别为基于CMake的CLion集成开发环境以及CMake工具（命令行）的方式，开发人员可任选其一。

##### CMake方式（CLion）

对于Linux或macOS平台的NiuTensor工具包安装，CLion集成开发环境可以通过对CMakeLists.txt文件进行解析自动获取项目信息（需要用户提前安装CMake工具以及CLion集成开发环境），操作步骤如下：

- 使用CLion打开NiuTensor项目所在目录（确保CMakeLists.txt在其根目录位置），CLion将根据CMakeLists.txt文件自动读取项目信息。
- 打开CLion首选项，点击“构建，执行，部署”选项卡中的CMake，在“CMake选项”中进行设置，设置完成后CLion将自动使用CMake对项目进行构建，如计划生成动态链接库，则仅需在在“CMake选项”中额外加上`-DGEN_DLL=ON`的CMake参数即可，否则默认生成可执行程序。
  - 如项目计划启用MKL数学运算库（需用户自行安装），则仅需在“CMake选项”中填入`-DUSE_MKL=ON`，并通过`-DINTEL_ROOT='/intel/root/path'`指定MKL库（Intel工具包）的安装路径。如`-DUSE_MKL=ON -DINTEL_ROOT='/opt/intel/compilers_and_libraries_2020.2.254/linux'`。
  - 如项目计划启用OpenBLAS数学运算库（需用户自行安装），则仅需在“CMake选项”中填入`-DUSE_OPENBLAS=ON`，并通过`-DOPENBLAS_ROOT='/openblas/root/path'`指定OpenBLAS库的安装路径。如`-DUSE_OPENBLAS=ON -DOPENBLAS_ROOT='/opt/OpenBLAS'`。
  - 如项目计划启用CUDA数学运算库（需用户自行安装），则仅需在“CMake选项”中填入`-DUSE_CUDA=ON`，并通过`-DCUDA_ROOT='/cuda/root/path'`指定CUDA库的安装路径。如`-DUSE_CUDA=ON -DCUDA_ROOT='/usr/local/cuda-9.2'`。如需在GPU设备上使用半精度浮点数进行运算，需在启用`-DUSE_CUDA=ON`参数的同时启用`-USE_HALF_PRECISION=ON`参数（需要注意的是半精度但需要注意的是，半精度操作仅在使用Pascal及更新架构的NVIDIA GPU中提供支持，该项可参考[NVIDIA GPU设备信息](https://developer.nvidia.com/cuda-gpus)进行查询）。

##### CMake方式（命令行）

若仅需通过命令行方式对项目进行管理，开发者同样可以使用CMake快速对NiuTensor项目进行编译安装（需要用户提前安装CMake工具），操作步骤如下：

- 在项目根目录打开Linux或macOS平台的命令行工具（如Terminal），在工具包内新建目录以保存生成的中间文件（如执行`mkdir build`建立build目录）。
- 执行`cd build`命令进入新建的build目录。
- 执行CMake命令对项目进行生成，如计划生成动态链接库，则仅需在命令中额外加上`-DGEN_DLL=ON`的CMake参数即可，否则默认生成可执行程序。
  - 如项目计划启用MKL数学运算库（需用户自行安装），则仅需在CMake命令中使用`-DUSE_MKL=ON`参数，并通过`-DINTEL_ROOT='/intel/root/path'`指定MKL库（Intel工具包）的安装路径。如`cmake -DUSE_MKL=ON -DINTEL_ROOT='/opt/intel/compilers_and_libraries_2020.2.254/linux' ..`。
  - 如项目计划启用OpenBLAS数学运算库（需用户自行安装），则仅需在CMake命令中使用`-DUSE_OPENBLAS=ON`参数，并通过`-DOPENBLAS_ROOT='/openblas/root/path'`指定OpenBLAS库的安装路径。如`cmake -DUSE_OPENBLAS=ON -DOPENBLAS_ROOT='/opt/OpenBLAS' ..`。
  - 如项目计划启用CUDA数学运算库（需用户自行安装），则仅需在CMake命令中使用`-DUSE_CUDA=ON`参数，并通过`-DCUDA_ROOT='/cuda/root/path'`指定CUDA库的安装路径。如`cmake -DUSE_CUDA=ON -DCUDA_ROOT='/usr/local/cuda-9.2' ..`。如需在GPU设备上使用半精度浮点数进行运算，需在启用`-DUSE_CUDA=ON`参数的同时启用`-USE_HALF_PRECISION=ON`参数（需要注意的是半精度但需要注意的是，半精度操作仅在使用Pascal及更新架构的NVIDIA GPU中提供支持，该项可参考[NVIDIA GPU设备信息](https://developer.nvidia.com/cuda-gpus)进行查询）。
- 执行成功将显示`Build files have been written to:...`并在该目录下生成Makefile文件。
- 执行`make -j`命令对NiuTensor项目进行编译，执行成功将显示`Built target NiuTensor`，安装完毕。

如编译目标为可执行程序，则在完成安装后将在bin目录下生成NiuTensor，如目标为动态链接库，则所生成的库文件将保存在项目根目录的lib路径下。

对于生成可执行程序的安装方式，在环境配置完成后，可以使用`NiuTensor -test`命令运行本项目的测试用例，如果最后输出

```shell
OK! Everything is good!
```

则说明本项目配置成功。

>注意：
>
>若先生成CPU上运行的可执行文件，之后如需生成GPU的可执行文件，需要先执行make clean命令，删除生成CPU时产生的中间结果，反之亦然。

### 将工具包嵌入到自己的项目中

若希望在自己的项目中使用NiuTensor工具包，这里建议将工具包编译为动态链接库进行使用，编译库的方式可以参考前文所述，而对于库文件的使用规则如下：

* 在所创建项目中需要引用XTensor.h、core里的CHeader.h和function里的FHeader.h这三个头文件：
  * 通过XTensor.h可以获取我们需要操作的XTensor类。
  * 通过core里的CHeader.h可以对Tensor进行一些张量运算。
  * 通过function里的FHeader.h可以调用一些激活函数。
* 在所创建项目中使用命名空间nts。

编译过程以单元测试以及样例程序为例（即编译NiuTensor库中的Main.cpp），假设动态链接库所在的路径为当前目录（项目根目录）的lib目录，则通过以下命令可以编译生成可执行文件：

```shell
g++ -o App ./source/Main.cpp -L./lib -lNiuTensor -std=c++11
```

然后将lib目录加入到环境变量中，可以通过以下命令进行临时修改（退出shell后则失效）：

```shell
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
```

或者将上述命令加入到~/.bashrc配置文件中进行永久修改（需要将相对路径./lib转换为绝对路径），并执行以下这条命令使修改立即生效：

```shell
source ~/.bashrc
```

之后就可以运行App可执行文件。

```shell
./App
```


## 开发团队

NiuTensor张量计算库由东北大学自然语言处理实验室小牛开源团队开发，致力于为深度学习相关研究及工业系统的开发提供轻量级的张量定义及计算功能。

## 更新版本

NiuTensor version 0.3.3 - 2020年9月14日
