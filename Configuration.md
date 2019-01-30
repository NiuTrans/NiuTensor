# NiuTrans.Tensor

# Windows下配置Visual Studio环境

## 注意事项

* 我们仅仅测试了较新版本的VS以及CUDA，对于之前的版本并不清楚是否存在问题。
* 建议使用VS版本为VS2015，或使用VS2017时**安装v140工具集**，之前的VS版本暂不清楚。
* 建议先安装VS再安装CUDA。
* VS2017安装CUDA时，不要勾选Visual Studio Integration，否则安装会出错。CUDA安装完成后，解压CUDA安装文件，在CUDAVisualStudioIntegration\extras\visual_studio_integration\MSBuildExtensions路径下有四个文件，拷贝到VS对应的目录，C:\Program Files (x86)\MSBuild\Microsoft.Cpp\v4.0\v140\BuildCustomizations
* （忽略此条）C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\IDE\VC\VCTargets\BuildCustomizations。
* CUDA9.2及以上尚且不支持VS2017最新版本，因此建议使用CUDA版本为9.0或9.1。

## CUDA配置

* 新建一个VC++空项目。
* 在菜单栏下方的编译选项中将**解决方案平台**设置为×64（默认是X86）。
* 将源代码(source文件夹)拷贝到项目中。

在已安装好VS、CUDA并配置好环境变量后，一些关键的CUDA配置选项如下所示，以下配置选项在 **菜单栏-项目 -> 属性** 中可以找到。

**VC++目录 -> 包含目录** 中加入	$(CUDA_PATH)\include

**VC++目录 -> 库目录** 中加入		$(CUDA_PATH)\lib\Win32

**链接器->输入->附加依赖项**中加入cuda.lib;cudadevrt.lib;cudart.lib;cudart_static.lib;nvcuvid.lib;OpenCL.lib;cublas.lib;curand.lib;

上述配置完成后，在解决方案资源管理器中，右键点击**项目**，选择**生成依赖性-生成自定义** ，选择CUDA9。
在.cu文件上右键属性，在项类型中选择"CUDA C/C++"（最好搜索.cu文件，然后全选设置）。

## 其他配置

注：以下选项也是 **菜单栏-项目 -> 属性** 中可以找到。C/C++选项需要在项目中加入C/C++文件之后才会显示出来。

**C/C++->常规->SDL检查**，设为否。

**C/C++->预处理器->预处理器定义** 中，添加

>USE_CUDA;USE_BLAS;WIN32;MKL;_DEBUG;_CRT_SECURE_NO_WARNINGS;_CRT_SECURE_NO_WARNINGS_CONSOLE;

**C/C++->预编译头->预编译头**，设置为不使用预编译头。

**链接器->系统->子系统**，设置为控制台。

**常规->字符集**，使用Unicode字符集。

**调试->命令参数**中设置可执行文件所需要的参数。


