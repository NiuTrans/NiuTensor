# NiuTensor

## Windows系统通过Visual Studio配置NiuTensor项目

### 注意事项

* 我们仅仅测试了VS2015和CUDA9.0之后的版本，对于之前的版本并不清楚是否存在问题。
* VS2015版本可以直接使用，使用较新版本的VS（如VS2017）时，需要**安装组件“适用于桌面的 VC++ 2015.3 v14.00 (v140) 工具集”**。
* 建议先安装Visual Studio再安装CUDA。安装CUDA时，建议不要勾选Visual Studio Integration，有时候可能会出错。CUDA安装完成后，解压CUDA安装文件（exe文件可以解压），在CUDAVisualStudioIntegration\extras\visual_studio_integration\MSBuildExtensions路径下有四个文件，拷贝到下述路径中。

  * VS2015
  > C:\Program Files (x86)\MSBuild\Microsoft.Cpp\v4.0\v140\BuildCustomizations

  * VS2015以上版本(以下两个路径分别对应v140工具集和VS默认工具集的路径)
  > C:\Program Files (x86)\MSBuild\Microsoft.Cpp\v4.0\v140\BuildCustomizations
  > C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\IDE\VC\VCTargets\BuildCustomizations

### 新建项目

* 新建一个VC++空项目。
* 将菜单栏中的**解决方案平台**设置为×64（默认是X86）。
* 在**菜单栏->项目->属性**中，将平台设置为X64。
* 将源代码(source文件夹)拷贝到项目的根目录，然后选择**菜单栏->项目->显示所有文件**，解决方案中即可以看到source文件夹，右键点击source，选择包含在项目中，即可将所有的*.h和*.cpp加入到本项目中。

### CUDA配置（无GPU设备可以跳过此步骤）

在VS项目中使用CUDA，需要设置项目的相关属性。
以下配置选项在 **菜单栏->项目 -> 属性** 中可以找到。

* **C/C++->预处理器->预处理器定义** 中，添加

> USE_CUDA;

* **VC++目录->包含目录** 中加入

> $(CUDA_PATH)\include

* **VC++目录->库目录** 中加入 

> $(CUDA_PATH)\lib\Win32

* **链接器->输入->附加依赖项**中加入以下库

> cuda.lib;cudadevrt.lib;cudart.lib;cudart_static.lib;nvcuvid.lib;OpenCL.lib;cublas.lib;curand.lib;

* 上述配置完成后，在**菜单栏->项目->生成自定义**中，勾选CUDA*（根据自己安装的CUDA版本自行选择）。
* 在所有的*.cu和*.cuh文件上右键，包含在项目中。

### 其他配置

注：以下选项也是 **菜单栏-项目 -> 属性** 中可以找到。

* **常规->平台工具集**，设置为Visual Studio 2015（v140）。

* **C/C++->常规->SDL检查**，设为否。

* **C/C++->预处理器->预处理器定义** 中，添加

> WIN32;_DEBUG;_CRT_SECURE_NO_WARNINGS;_CRT_SECURE_NO_WARNINGS_CONSOLE;

* **C/C++->预编译头->预编译头**，设置为不使用预编译头。

* **链接器->系统->子系统**，设置为控制台。

* **常规->字符集**，使用Unicode字符集。

* **调试->命令参数**，设置可执行文件所需要的参数（初始可以设置为-test，用来执行测试用例）。
