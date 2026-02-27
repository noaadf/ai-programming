cmake_minimum_required(VERSION 3.18)  # 3.18 起原生支持 CUDA 作为语言

# 检测 CUDA 是否可用，可用则启用
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    project(mytensor LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER}")
else()
    project(mytensor LANGUAGES CXX)
    message(STATUS "CUDA not found, building CPU-only version")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 找 pybind11
find_package(pybind11 REQUIRED)

# 根据是否有 CUDA 选择源文件
set(COMMON_SOURCES
    bindings.cpp
)

if(CMAKE_CUDA_COMPILER)
    set(SOURCES ${COMMON_SOURCES} tensor.cu activation.cu layers.cu gemm.cu)
else()
    set(SOURCES ${COMMON_SOURCES} tensor.cpp activation.cpp layers.cpp gemm.cpp)
endif()

# 构建 pybind11 模块
pybind11_add_module(mytensor ${SOURCES})

# CUDA 编译选项
if(CMAKE_CUDA_COMPILER)
    target_compile_definitions(mytensor PRIVATE USE_CUDA)
    find_library(CUBLAS_LIB cublas HINTS /usr/local/cuda/lib64)
    target_link_libraries(mytensor PRIVATE ${CUBLAS_LIB})
endif()
