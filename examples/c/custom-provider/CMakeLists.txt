# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

cmake_minimum_required(VERSION 3.20)
project(ovsensors_custom_provider LANGUAGES CXX)

find_package(ovsensors CONFIG REQUIRED COMPONENTS core)
find_package(ovstage CONFIG REQUIRED)

add_library(sample_pose_provider SHARED pose_provider.cpp)
target_compile_features(sample_pose_provider PRIVATE cxx_std_17)
target_compile_definitions(sample_pose_provider PRIVATE POSE_PROVIDER_BUILD)
target_link_libraries(sample_pose_provider PRIVATE ovsensors::ovsensors_headers)
set_target_properties(sample_pose_provider PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)

add_executable(custom_provider main.cpp)
target_compile_features(custom_provider PRIVATE cxx_std_17)
target_link_libraries(custom_provider PRIVATE sample_pose_provider ovsensors::ovsensors ovstage::ovstage)

if(WIN32)
    if(NOT MSVC)
        message(FATAL_ERROR "This example's Windows runtime selection requires MSVC.")
    endif()
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/selected_runtime.cpp"
        INPUT "${CMAKE_CURRENT_SOURCE_DIR}/selected_runtime.cpp.in")
    target_sources(custom_provider PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/selected_runtime.cpp")
    target_link_libraries(custom_provider PRIVATE delayimp)
    foreach(runtime ovsensors::ovsensors ovstage::ovstage)
        target_link_options(custom_provider PRIVATE "/DELAYLOAD:$<TARGET_FILE_NAME:${runtime}>")
    endforeach()
elseif(UNIX AND NOT APPLE)
    set_property(TARGET custom_provider PROPERTY BUILD_RPATH
        "$<TARGET_FILE_DIR:sample_pose_provider>;$<TARGET_FILE_DIR:ovsensors::ovsensors>;$<TARGET_FILE_DIR:ovstage::ovstage>")
    target_link_options(custom_provider PRIVATE "LINKER:--disable-new-dtags")
else()
    message(FATAL_ERROR "This example supports Linux and Windows SDKs.")
endif()

# Run on every build so an asset-only edit is staged without forcing a relink.
add_custom_target(custom_provider_scene ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/scene.usda" "$<TARGET_FILE_DIR:custom_provider>/scene.usda")
add_dependencies(custom_provider custom_provider_scene)
message(STATUS "Custom provider uses core at ${ovsensors_DIR} and ovstage at ${ovstage_DIR}; ovrtx is not required.")
