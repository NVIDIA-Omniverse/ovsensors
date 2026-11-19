# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""The optional Warp consumer: sum RGB bytes without changing borrowed pixels."""

import warp as wp


# [snippet:warp-reduction]
@wp.kernel
def sum_rgb(pixels: wp.array3d(dtype=wp.uint8), total: wp.array(dtype=wp.uint64)):
    y, x = wp.tid()
    value = wp.uint64(pixels[y, x, 0]) + wp.uint64(pixels[y, x, 1]) + wp.uint64(pixels[y, x, 2])
    wp.atomic_add(total, 0, value)
# [/snippet:warp-reduction]
