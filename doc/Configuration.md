# NiuTrans.Tensor环境配置

## 注意事项

CUDA最新版本9.2尚且不支持VS2017最新版本，因此建议使用CUDA版本为9.0或9.1，建议使用VS版本为VS2015，或使用VS2017时安装v140工具集，解决方案平台设置为×64。

## CUDA配置

在已安装好VS、CUDA并配置好环境变量后，一些关键的CUDA配置选项如下所示，以下配置选项在 **项目 -> 属性** 中可以找到。

>$(CUDA_PATH)\include

加入到 **VC++目录 -> 包含** 中。

>$(CUDA_PATH)\lib\Win32

加入到 **VC++目录 -> 库** 中。

>cuda.lib;cudadevrt.lib;cudart.lib;cudart_static.lib;nvcuvid.lib;OpenCL.lib;cublas.lib;curand.lib;

加入到 **链接器->输入->附加依赖项** 中。

配置完成后，右键 **工程->项目依赖性** ，选择CUDA9。
在.cu文件上右键属性，在项类型中选择"CUDA C/C++"（最好搜索.cu文件，然后全选设置）。

## 其他配置

**C/C++->常规->SDL检查**，设为否。

在 **C/C++->预处理器->预处理器定义** 中，添加

>USE_CUDA;USE_BLAS;WIN32;MKL;_DEBUG;_CRT_SECURE_NO_WARNINGS;_CRT_SECURE_NO_WARNINGS_
CONSOLE;

**链接器->系统->子系统**，设置为控制台。

**常规->字符集**，使用Unicode字符集。

**调试->命令参数**中设置可执行文件所需要的参数。


