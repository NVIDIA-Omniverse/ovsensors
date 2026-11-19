// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary

#ifndef EXAMPLE_POSE_PROVIDER_H
#define EXAMPLE_POSE_PROVIDER_H

#include <ovsensors/ovsensors_provider.h>

#if defined(_WIN32)
#if defined(POSE_PROVIDER_BUILD)
#define POSE_API __declspec(dllexport)
#else
#define POSE_API __declspec(dllimport)
#endif
#else
#define POSE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// The caller owns the bundle, serializes calls, and keeps this DSO loaded.
// Failure zeroes out_provider. This sample factory is not an ovsensors API.
POSE_API ovsensors_api_status_t pose_provider_create(ovsensors_provider_t* out_provider);

// Call only after every borrowing ovsensors instance was successfully destroyed.
// BUSY preserves ownership if still attached or holding sensors, operations, or maps.
// SUCCESS consumes and zeroes the bundle; an already-zero bundle is a no-op.
POSE_API ovsensors_api_status_t pose_provider_destroy(ovsensors_provider_t* provider);

#ifdef __cplusplus
}
#endif
#endif
