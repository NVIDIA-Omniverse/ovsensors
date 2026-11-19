# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Wrappers for provider factories shipped as independent shared libraries.

Provider modules are imported explicitly and load their shared libraries lazily when
used. Importing ``ovsensors`` or this package does not eagerly load a provider DSO.
"""

__all__ = []
