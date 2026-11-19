# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Aggregate project for the runnable C and C++ examples.

cmake_minimum_required(VERSION 3.18)
project(ovsensors_examples LANGUAGES C CXX)

# Preserve debug symbols in Release builds (for stack traces and debugging).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -g")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g")
elseif(MSVC)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /Zi")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG")
endif()

# Filtered packages may omit individual examples.
foreach(_ovsensors_example
    minimal
    point-cloud
    custom-provider)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_ovsensors_example}/CMakeLists.txt")
        add_subdirectory(${_ovsensors_example})
    else()
        message(STATUS "skipping example ${_ovsensors_example}: not in this tree")
    endif()
endforeach()
