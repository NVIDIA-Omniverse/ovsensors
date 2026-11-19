# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

cmake_minimum_required(VERSION 3.20)
project(ovsensors_point_cloud LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/ovsensors_example.cmake")
ovsensors_example_find_packages()
add_executable(ovsensors_point_cloud main.c)
ovsensors_example_configure(ovsensors_point_cloud)
ovsensors_example_stage_asset(ovsensors_point_cloud point_cloud_scene.usda)
