# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

cmake_minimum_required(VERSION 3.20)
project(ovsensors_minimal CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../cmake")
include(ovsensors_example)
ovsensors_example_find_packages()

add_executable(ovsensors_minimal main.cpp image.cpp)

# Links ovsensors core, the shipped ovrtx-backed provider, ovstage, and ovrtx.
# ovrtx supplies ovrtx_register_schema_paths(); the provider owns rendering and
# this application does not use ovrtx renderer types. See ../cmake/ovsensors_example.cmake.
ovsensors_example_configure(ovsensors_minimal)

# Stage the scene beside the executable so it runs with no argument from any
# working directory.
ovsensors_example_stage_asset(ovsensors_minimal scene.usda)
