# the prefix of the generated executable file
NIUTRANS_EXE := NiuTensor

# code path and generated file path
ROOT = .
SRC = $(ROOT)/source
LIB_DIR = $(ROOT)/lib
EXE_DIR = $(ROOT)/bin

# whether to generate dll
dll = 0

# 0 - use CPU 
# 1 - use GPU
USE_CUDA = 1
# modify this path if neccessary
CUDA_ROOT = /usr/local/cuda-9.0
CUDA_LIB_DIR = $(CUDA_ROOT)/lib64
CUDA_INCLUDE = $(CUDA_ROOT)/include

# use MKL 
USE_MKL = 0
INTEL_ROOT = /opt/intel
MKL_ROOT = /opt/intel/mkl
MKL_LIB_DIR = $(MKL_ROOT)/lib/intel64/
MKL_INCLUDE = $(MKL_ROOT)/include

# use OpenBLAS
USE_OPENBLAS = 0
OPENBLAS_ROOT = /opt/OpenBLAS
OPENBLAS_LIB_DIR = $(OPENBLAS_ROOT)/lib
OPENBLAS_INCLUDE = $(OPENBLAS_ROOT)/include

SRC_DIR = $(shell find $(SRC) -type d)

# included header files directory
# depended outside library files directory
INC_DIR = $(SRC_DIR)
DEPLIB_DIR = 
ifeq ($(USE_CUDA), 1)
	INC_DIR += $(CUDA_INCLUDE)
	DEPLIB_DIR += $(CUDA_LIB_DIR)
endif
ifeq ($(USE_MKL), 1)
	INC_DIR += $(MKL_INCLUDE)
	DEPLIB_DIR += $(MKL_LIB_DIR)
endif
ifeq ($(USE_OPENBLAS), 1)
	INC_DIR += $(OPENBLAS_INCLUDE)
	DEPLIB_DIR += $(OPENBLAS_LIB_DIR)
endif

# macro
MACRO = 
ifeq ($(USE_CUDA), 1)
	MACRO += -DUSE_CUDA
endif
ifeq ($(USE_MKL), 1)
	MACRO += -DUSE_BLAS -DMKL
endif
ifeq ($(USE_OPENBLAS), 1)
	MACRO += -DUSE_BLAS -DOPENBLAS
endif

# dependency
STATIC_DEPLIB = 
DYNAMIC_DEPLIB = -lpthread
ifeq ($(USE_MKL), 1)
    STATIC_DEPLIB += $(MKL_LIB_DIR)/libmkl_intel_lp64.a \
	                 $(MKL_LIB_DIR)/libmkl_core.a \
					 $(MKL_LIB_DIR)/libmkl_intel_thread.a \
					 $(INTEL_ROOT)/lib/intel64/libiomp5.a                                              
    DYNAMIC_DEPLIB += -liomp5 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core
endif
ifeq ($(USE_OPENBLAS), 1)
    STATIC_DEPLIB += $(OPENBLAS_LIB_DIR)/libopenblas.a
    DYNAMIC_DEPLIB += -lopenblas
endif
ifeq ($(USE_CUDA), 1)
    STATIC_DEPLIB += $(CUDA_LIB_DIR)/libcublas_static.a \
                     $(CUDA_LIB_DIR)/libculibos.a \
                     $(CUDA_LIB_DIR)/libnpps_static.a \
                     $(CUDA_LIB_DIR)/libnppc_static.a \
                     $(CUDA_LIB_DIR)/libcudadevrt.a \
                     $(CUDA_LIB_DIR)/libcurand_static.a \
					 /lib64/libdl.so.2
    DYNAMIC_DEPLIB += -lcudart -lnvidia-ml
endif 
DEPLIBS = -Wl,--start-group $(STATIC_DEPLIB) -Wl,--end-group -lm -ldl $(DYNAMIC_DEPLIB)

# specify the compilers here
CC = gcc
CXX = g++
NVCC = $(CUDA_ROOT)/bin/nvcc
ifeq ($(USE_INTEL_COMPILER), 1)
	CC = icc
	CXX = icc
endif

# main file
MAIN_FILE = $(SRC)/network/Main.cpp
Tensor_Main := $(SRC)/tensor/Main.cpp
Network_Main := $(SRC)/network/Main.cpp

ifeq ($(USE_CUDA), 1)
	NIUTRANS_EXE := $(NIUTRANS_EXE).GPU
else
	NIUTRANS_EXE := $(NIUTRANS_EXE).CPU
endif

NIUTRANS_DLL := $(LIB_DIR)/lib$(NIUTRANS_EXE).so

NIUTRANS_EXE := $(EXE_DIR)/$(NIUTRANS_EXE)

# specify the compiling arguments here
CFLAGS = -std=c++11 -msse4.2 -w -march=native -Wno-enum-compare -Wno-sign-compare -Wno-reorder -Wno-format

# gtx 1080 arch=compute_61,code=sm_61
# k80 arch=compute_37,code=sm_37
# if we set wrong, the result can be `-inf`
CUDA_FLAG = -arch=sm_30 \
			-gencode=arch=compute_30,code=sm_30 \
			-gencode=arch=compute_50,code=sm_50 \
			-gencode=arch=compute_52,code=sm_52 \
			-gencode=arch=compute_60,code=sm_60 \
			-gencode=arch=compute_61,code=sm_61 \
			-gencode=arch=compute_62,code=sm_62 \
			-gencode=arch=compute_70,code=sm_70 \
			-gencode=arch=compute_70,code=compute_70 \
			-maxrregcount=0  --machine 64 -DUSE_CUDA --use_fast_math -std=c++11

CFLAGS += -O3 -flto -DNDEBUG -rdynamic -fkeep-inline-functions

# include dir
CFLAGS += -fPIC $(addprefix -I, $(INC_DIR))
# CUDA_FLAG += $(addprefix -I, $(INC_DIR))
CXXFLAGS = $(CFLAGS)

# lib dir
LDFLAGS = $(addprefix -L, $(DEPLIB_DIR))

# decoder source file
ifeq ($(USE_CUDA), 1)
	SOURCES := $(foreach dir,$(SRC_DIR),$(wildcard $(dir)/*.c) $(wildcard $(dir)/*.cpp) $(wildcard $(dir)/*.cc) $(wildcard $(dir)/*.cu))
else
	SOURCES := $(foreach dir,$(SRC_DIR),$(wildcard $(dir)/*.c) $(wildcard $(dir)/*.cpp) $(wildcard $(dir)/*.cc) )
endif

SOURCES := $(subst $(Tensor_Main), ,$(SOURCES))
SOURCES := $(subst $(Network_Main), ,$(SOURCES))

# object file
OBJS := $(patsubst %.c,%.o,$(SOURCES))
OBJS := $(patsubst %.cpp,%.o,$(OBJS))
ifeq ($(USE_CUDA), 1)
	OBJS := $(patsubst %.cu,%.cuo,$(OBJS))
endif

all: start lib exe finish

start:
	@echo ""
	@echo "Start building ..."

lib: start_lib niutrans_dll finish_lib

start_lib:
	@mkdir -p $(LIB_DIR)
	@echo ""
	@echo "Start building library"

niutrans_dll: $(NIUTRANS_DLL)

$(NIUTRANS_DLL): $(OBJS)
ifeq ($(dll), 1)
	@echo "Building dynamic link library: $(NIUTRANS_DLL)"
	@$(CXX) -shared -Wall $(CXXFLAGS) $(MACRO) $(LDFLAGS) $(OBJS) $(DEPLIBS) -o $@
else
	@echo "Skip building dynamic link library"
endif
	
finish_lib:
	@echo "Finish building library"
	@echo ""

exe: start_exe niutrans_exe finish_exe

start_exe:
	@mkdir -p $(EXE_DIR)
	@echo ""
	@echo "Start building executable file"

niutrans_exe: $(NIUTRANS_EXE)

$(NIUTRANS_EXE): $(OBJS) $(MAIN_FILE)
	@echo "Building executable file: $(NIUTRANS_EXE)"
	@$(CXX) $(MAIN_FILE) $(CXXFLAGS) $(MACRO) $(LDFLAGS) $(OBJS) $(DEPLIBS) -o $@

finish_exe:
	@echo "Finish building executable file"
	@echo ""

finish:
	@echo "Finish building ..."
	@echo ""

%.o: %.c
	@$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	@$(CXX) $(CXXFLAGS) $(MACRO) -c $< -o $@

%.cuo: %.cu
ifeq ($(dll), 1)
	@$(NVCC) --shared --compiler-options '-fPIC' $(CUDA_FLAG) -c $< -o $@
else
	@$(NVCC) $(CUDA_FLAG) -c $< -o $@
endif

.PHONY: clean
clean:
	@echo "Cleaning object files"
	@-rm -f $(OBJS)
	
