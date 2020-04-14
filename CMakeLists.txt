# if your visual studio's version is before 2019
# use commond "cmake -A x64 .." to build this project
# or use cmake gui to build VS (remember to select x64)
# if there's some warnings, don't worry about that.

# cmake minimum version
cmake_minimum_required(VERSION 2.8)

# project's name
project(NiuTensor)

# the prefix of the generated executable file
set(NIUTRANS_EXE "NiuTensor")

# generated file path
set(EXECUTABLE_OUTPUT_PATH ../bin)

# 0 - Linux or macOS
# 1 - Windows
set(ON_WINDOWS 0)

# 0 - Not use
# 1 - Use, if set 1, please modify the CUDA_VERSION and CUDA_TOOLKIT_ROOT_DIR below
set(USE_CUDA 0)
if (USE_CUDA)
    set(CUDA_VERSION 9.0)
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda") 
endif()

# use CMAKE_MACOSX_RPATH for macOS 
set(CMAKE_MACOSX_RPATH 1)

# open floder manage
set_property(GLOBAL PROPERTY USE_FOLDERS On)
add_definitions(-std=c++11)

# find all the .cpp .h .cu .chu files in source folder
file(GLOB_RECURSE CPP_FILES source/*.cpp)
file(GLOB_RECURSE H_FILES source/*.h)
file(GLOB_RECURSE CU_FILES source/*.cu)
file(GLOB_RECURSE CUH_FILES source/*.cuh)

function(assign_source_group)
    foreach(_source IN ITEMS ${ARGN})
        if (IS_ABSOLUTE "${_source}")
            file(RELATIVE_PATH _source_rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_source}")
        else()
            set(_source_rel "${_source}")
        endif()
        get_filename_component(_source_path "${_source_rel}" PATH)
        string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
        source_group("${_source_path_msvc}" FILES "${_source}")
    endforeach()
endfunction(assign_source_group)

function(my_add_executable)
    foreach(_source IN ITEMS ${ARGN})
        assign_source_group(${_source})
    endforeach()
    if(USE_CUDA)
        cuda_add_executable(${ARGV})
    else()
        add_executable(${ARGV})
    endif()
endfunction(my_add_executable)

if(USE_CUDA)
    set(NIUTRANS_EXE "${NIUTRANS_EXE}.GPU")
    add_definitions(-DUSE_CUDA)
    if(ON_WINDOWS)
        find_package(CUDA ${CUDA_VERSION} REQUIRED)
        add_compile_options(-Wno-dev)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4819")
        set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-maxrregcount=0 -m64 --disable-warnings -use_fast_math -DUSE_CUDA")
        set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} -arch=sm_30 
            -gencode=arch=compute_30,code=sm_30 
            -gencode=arch=compute_50,code=sm_50 
            -gencode=arch=compute_52,code=sm_52 
            -gencode=arch=compute_60,code=sm_60 
            -gencode=arch=compute_61,code=sm_61 
            -gencode=arch=compute_62,code=sm_62 
            -gencode=arch=compute_70,code=sm_70 
            -gencode=arch=compute_70,code=compute_70 
        )
        set(CMAKE_POLICY_DEFAULT_CMP0028 NEW)
        link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
        include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
        set(CUDA_LIB_PATH_ME "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/")
        set(CUDA_LIBS_ME ${CUDA_LIBS_ME} "${CUDA_LIB_PATH_ME}cublas.lib")
        set(CUDA_LIBS_ME ${CUDA_LIBS_ME} "${CUDA_LIB_PATH_ME}npps.lib")
        set(CUDA_LIBS_ME ${CUDA_LIBS_ME} "${CUDA_LIB_PATH_ME}nppc.lib")
        set(CUDA_LIBS_ME ${CUDA_LIBS_ME} "${CUDA_LIB_PATH_ME}cudadevrt.lib")
        set(CUDA_LIBS_ME ${CUDA_LIBS_ME} "${CUDA_LIB_PATH_ME}curand.lib")
        my_add_executable(${NIUTRANS_EXE} ${CPP_FILES} ${H_FILES} ${CU_FILES} ${CUH_FILES})
    else()
        find_package(CUDA ${CUDA_VERSION} REQUIRED)
        set(CMAKE_CXX_FLAGS "-fPIC -msse4.2 -w -march=native -Wno-enum-compare -Wno-sign-compare -Wno-format -Wno-dev -O3 -DNDEBUG -rdynamic")
        set(CUDA_NVCC_FLAGS "-Xcompiler -fPIC -maxrregcount=0 --disable-warnings -use_fast_math -DUSE_CUDA -Wno-deprecated-gpu-targets -std=c++11")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_30 
            -gencode=arch=compute_30,code=sm_30 
            -gencode=arch=compute_50,code=sm_50 
            -gencode=arch=compute_52,code=sm_52 
            -gencode=arch=compute_60,code=sm_60 
            -gencode=arch=compute_61,code=sm_61 
            -gencode=arch=compute_62,code=sm_62 
            -gencode=arch=compute_70,code=sm_70 
            -gencode=arch=compute_70,code=compute_70 
        )
        link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
        include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
        set(CUDA_LIB_PATH_ME "${CUDA_TOOLKIT_ROOT_DIR}/lib64/")
        set(CUDA_LIBS_ME ${CUDA_LIBS_ME} "${CUDA_LIB_PATH_ME}libcublas_static.a")
        set(CUDA_LIBS_ME ${CUDA_LIBS_ME} "${CUDA_LIB_PATH_ME}libculibos.a")
        set(CUDA_LIBS_ME ${CUDA_LIBS_ME} "${CUDA_LIB_PATH_ME}libnpps_static.a")
        set(CUDA_LIBS_ME ${CUDA_LIBS_ME} "${CUDA_LIB_PATH_ME}libnppc_static.a")
        set(CUDA_LIBS_ME ${CUDA_LIBS_ME} "${CUDA_LIB_PATH_ME}libcudadevrt.a")
        set(CUDA_LIBS_ME ${CUDA_LIBS_ME} "${CUDA_LIB_PATH_ME}libcurand_static.a")
        set(CUDA_LIBS_ME ${CUDA_LIBS_ME} "/usr/lib64/libdl.so.2")
        my_add_executable(${NIUTRANS_EXE} ${CPP_FILES} ${H_FILES} ${CU_FILES} ${CUH_FILES})
    endif()
else()
    set(NIUTRANS_EXE "${NIUTRANS_EXE}.CPU")
    my_add_executable(${NIUTRANS_EXE} ${CPP_FILES} ${H_FILES})
endif()

set(CUDA_LIB ${CUDA_LIBS_ME})

if(ON_WINDOWS)
    add_definitions(-DWIN32) # for windows specially
    if(USE_CUDA)
        message(STATUS "On Windows and use cuda")
        target_link_libraries(${NIUTRANS_EXE} ${CUDA_LIB})
    else()
        message(STATUS "On Windows and not use cuda")
        target_link_libraries(${NIUTRANS_EXE} ${CUDA_LIB})
    endif()
else()
    if(USE_CUDA)
        message(STATUS "On Linux and use cuda")
        target_link_libraries(${NIUTRANS_EXE} ${CUDA_LIB} -lpthread -lcudart -lnvidia-ml)
    else()
        message(STATUS "On Linux and not use cuda")
        target_link_libraries(${NIUTRANS_EXE} ${CUDA_LIB} -lpthread)
    endif()
endif()

message(STATUS "name of executable file:" ${NIUTRANS_EXE})