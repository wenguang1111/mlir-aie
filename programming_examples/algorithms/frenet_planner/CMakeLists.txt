# frenet_planner/CMakeLists.txt
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
#
# CMake build for the NPU Frenet/FISS+ Planner programming example.
# Generates the MLIR module via frenet_planner.py (IRON design) and
# compiles the AIE kernel (frenet_planner_kernel.cc) before linking
# everything into a runnable test binary.
#
# Parameters:
#   -DXRT_INC_DIR : Full path to XRT src/runtime_src/core/include
#   -DXRT_LIB_DIR : Path to xrt_coreutil library
#   -DTARGET_NAME : Output executable name (default: test)

cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

include(../../common.cmake)

find_program(WSL NAMES powershell.exe)

if(NOT WSL)
    if(NOT DEFINED CMAKE_C_COMPILER)
        set(CMAKE_C_COMPILER gcc-13)
    endif()
    if(NOT DEFINED CMAKE_CXX_COMPILER)
        set(CMAKE_CXX_COMPILER g++-13)
    endif()
endif()

set(TARGET_NAME test CACHE STRING "Target executable name")

set(ProjectName  ${TARGET_NAME})
set(currentTarget ${TARGET_NAME})

if(WSL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
    add_compile_options(/Zc:__cplusplus)
endif()

project(${ProjectName})

# ── Host test executable ──────────────────────────────────────────────────────
add_executable(${currentTarget} test.cpp)

set_property(TARGET ${currentTarget} PROPERTY CXX_STANDARD 23)

target_include_directories(${currentTarget} PUBLIC ${XRT_INC_DIR})

target_link_directories(${currentTarget} PUBLIC ${XRT_LIB_DIR})

target_link_libraries(${currentTarget} PUBLIC xrt_coreutil)

target_link_test_utils(${currentTarget})

# ── MLIR / NPU binary generation ─────────────────────────────────────────────
# Requires the IRON toolchain (aiecc.py / Python flow) to be on PATH.
# The generated frenet_planner.mlir is embedded into the test at compile time
# via the standard mlir_aie_compile_design() macro defined in common.cmake.
# Uncomment and adapt the line below once the IRON Python flow is integrated:
#
# mlir_aie_compile_design(frenet_planner.py --dev npu
#     OUTPUT frenet_planner.mlir
#     KERNELS kernels/frenet_planner_kernel.cc)
