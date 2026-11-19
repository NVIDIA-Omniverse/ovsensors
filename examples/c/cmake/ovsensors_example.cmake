# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Use explicit SDK prefixes, or dependency wheels from the interpreter used by uv.
set(_OVSENSORS_EXAMPLE_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")
if(WIN32)
    # The C application also uses the C++ Windows delay-load hook below.
    enable_language(CXX)
endif()

macro(ovsensors_example_find_packages)
    if(NOT ovsensors_ROOT)
        get_filename_component(ovsensors_ROOT "${_OVSENSORS_EXAMPLE_CMAKE_DIR}/../../.." ABSOLUTE)
    endif()
    set(_OVSENSORS_DISCOVERED_WHEELS FALSE)
    if(ovrtx_ROOT AND ovstage_ROOT)
        get_filename_component(OVSENSORS_EXAMPLE_SDK "${ovsensors_ROOT}" ABSOLUTE)
        foreach(_package ovrtx ovstage)
            string(TOUPPER "${_package}" _upper)
            get_filename_component(${_upper}_WHEEL_PREFIX "${${_package}_ROOT}" REALPATH)
        endforeach()
    else()
        set(_OVSENSORS_DISCOVERED_WHEELS TRUE)
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" "${_OVSENSORS_EXAMPLE_CMAKE_DIR}/discover_wheels.py"
                "${ovsensors_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/example_wheels.cmake"
            RESULT_VARIABLE _wheel_status ERROR_VARIABLE _wheel_error)
        if(NOT _wheel_status EQUAL 0)
            message(FATAL_ERROR "Wheel discovery failed: ${_wheel_error}\nInstall compatible dependency wheels in the selected Python3_EXECUTABLE environment, or set ovrtx_ROOT and ovstage_ROOT explicitly.")
        endif()
        include("${CMAKE_CURRENT_BINARY_DIR}/example_wheels.cmake")
    endif()
    # Regeneration runs from the build directory, not the original command's cwd.
    # Preserve the SDK selected by a relative command-line root across later runs.
    set(ovsensors_ROOT "${OVSENSORS_EXAMPLE_SDK}")
    set(ovsensors_ROOT "${ovsensors_ROOT}" CACHE PATH "Absolute ovsensors SDK root" FORCE)
    # ovstage first: ovrtxConfig.cmake then reuses this ovstage_DIR for its dependency.
    foreach(_package ovstage ovrtx)
        string(TOUPPER "${_package}" _upper)
        # Use the selected config's exports, including development build locations
        # and platform-specific import libraries.
        # Ignore stale cache entries and ambient CMAKE_PREFIX_PATH installations.
        set(${_package}_DIR "${${_upper}_WHEEL_PREFIX}/lib/cmake/${_package}")
        find_package(${_package} REQUIRED CONFIG PATHS "${${_package}_DIR}" NO_DEFAULT_PATH)
        if(_OVSENSORS_DISCOVERED_WHEELS)
            get_target_property(_selected_library ${_package}::${_package} IMPORTED_LOCATION)
            get_filename_component(_selected_library "${_selected_library}" REALPATH)
            if(NOT _selected_library STREQUAL "${${_upper}_WHEEL_LIBRARY}")
                message(FATAL_ERROR "${_package} target selects another runtime: ${_selected_library}")
            endif()
        endif()
        message(STATUS "${_package}: ${${_upper}_WHEEL_PREFIX}")
    endforeach()
    set(ovsensors_DIR "${OVSENSORS_EXAMPLE_SDK}/lib/cmake/ovsensors")
    find_package(ovsensors REQUIRED CONFIG COMPONENTS core provider_ovrtx
        PATHS "${ovsensors_DIR}" NO_DEFAULT_PATH)
    message(STATUS "ovsensors SDK: ${OVSENSORS_EXAMPLE_SDK}")
endmacro()

function(ovsensors_example_configure TARGET_NAME)
    if(WIN32 AND NOT MSVC)
        message(FATAL_ERROR "ovsensors Windows examples require the Visual Studio C++ toolchain (MSVC).")
    endif()
    target_link_libraries(${TARGET_NAME} PRIVATE
        ovsensors::ovsensors ovsensors::provider_ovrtx ovstage::ovstage ovrtx::ovrtx)
    configure_file("${_OVSENSORS_EXAMPLE_CMAKE_DIR}/example_paths.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/example_paths.h.gen" @ONLY)
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/example_paths.h"
        INPUT "${CMAKE_CURRENT_BINARY_DIR}/example_paths.h.gen")
    target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    if(MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE /W4)
    else()
        target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    if(WIN32)
        # Load the selected SDK/wheel DLLs in place without changing the process PATH.
        target_link_libraries(${TARGET_NAME} PRIVATE delayimp)
        foreach(_runtime ovsensors::ovsensors ovsensors::provider_ovrtx ovrtx::ovrtx ovstage::ovstage)
            target_link_options(${TARGET_NAME} PRIVATE "/DELAYLOAD:$<TARGET_FILE_NAME:${_runtime}>")
        endforeach()
        target_compile_options(${TARGET_NAME} PRIVATE /utf-8)
        set(_windows_runtime "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_windows_runtime_$<CONFIG>.cpp")
        file(GENERATE OUTPUT "${_windows_runtime}"
            INPUT "${_OVSENSORS_EXAMPLE_CMAKE_DIR}/windows_runtime.cpp.in")
        target_sources(${TARGET_NAME} PRIVATE "${_windows_runtime}")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # DT_RPATH also pins transitive loader resolution to the selected packages.
        target_link_options(${TARGET_NAME} PRIVATE "-Wl,--disable-new-dtags")
        set_target_properties(${TARGET_NAME} PROPERTIES
            BUILD_RPATH "${OVSENSORS_BINARY_DIR};${OVRTX_BINARY_DIR};${OVSTAGE_BINARY_DIR}")
    endif()
endfunction()

function(ovsensors_example_stage_asset TARGET_NAME ASSET)
    string(MAKE_C_IDENTIFIER "${ASSET}" _asset_name)
    # Asset-only edits must be staged even when the executable needs no relink.
    add_custom_target(${TARGET_NAME}_${_asset_name} ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${TARGET_NAME}>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_LIST_DIR}/${ASSET}"
            "$<TARGET_FILE_DIR:${TARGET_NAME}>/${ASSET}"
        COMMENT "Staging ${ASSET} beside ${TARGET_NAME}"
        VERBATIM)
    add_dependencies(${TARGET_NAME} ${TARGET_NAME}_${_asset_name})
endfunction()
