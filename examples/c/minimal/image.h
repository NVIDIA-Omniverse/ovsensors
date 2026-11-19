// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//
#ifndef OVSENSORS_EXAMPLE_IMAGE_H
#define OVSENSORS_EXAMPLE_IMAGE_H

#include <ovx/dlpack/dlpack.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace example
{
struct Image
{
    int64_t width = 0;
    int64_t height = 0;
    std::vector<unsigned char> rgb;
};

// LdrColor is dense HWC RGB(A). Accept uint8 or normalized float32; reject
// unsupported layouts rather than interpreting padding or device data as pixels.
inline Image copyImage(const DLTensor& tensor)
{
    if (!tensor.data || !tensor.shape || tensor.ndim != 3 || tensor.device.device_type != kDLCPU ||
        tensor.dtype.lanes != 1 || tensor.shape[0] <= 0 || tensor.shape[1] <= 0 ||
        (tensor.shape[2] != 3 && tensor.shape[2] != 4))
        throw std::runtime_error("LdrColor must be a nonempty CPU tensor with shape [height, width, 3 or 4]");
    const bool bytes = tensor.dtype.code == kDLUInt && tensor.dtype.bits == 8;
    const bool floats = tensor.dtype.code == kDLFloat && tensor.dtype.bits == 32;
    if (!bytes && !floats)
        throw std::runtime_error("LdrColor must contain uint8 or float32 components");
    const int64_t height = tensor.shape[0], width = tensor.shape[1], channels = tensor.shape[2];
    const size_t itemSize = bytes ? 1 : sizeof(float);
    const uint64_t limit = std::min<uint64_t>(std::numeric_limits<int64_t>::max(),
                                            std::numeric_limits<size_t>::max());
    if (static_cast<uint64_t>(width) > limit / channels / itemSize / height)
        throw std::runtime_error("LdrColor shape overflows addressable storage");
    const size_t elementCount = static_cast<size_t>(height * width * channels);
    if (tensor.byte_offset > limit - elementCount * itemSize)
        throw std::runtime_error("LdrColor byte offset overflows addressable storage");
    if (tensor.strides && (tensor.strides[2] != 1 || tensor.strides[1] != channels ||
                           tensor.strides[0] != width * channels))
        throw std::runtime_error("LdrColor must use dense HWC strides");

    Image image{ width, height, std::vector<unsigned char>(static_cast<size_t>(height * width * 3)) };
    const auto* data = static_cast<const unsigned char*>(tensor.data) + tensor.byte_offset;
    for (size_t i = 0; i < elementCount; ++i)
    {
        float value = 0;
        if (bytes)
            value = data[i] / 255.0f;
        else
            std::memcpy(&value, data + i * sizeof(float), sizeof(float));
        // Alpha must be finite too, although this example saves RGB only.
        if (!std::isfinite(value))
            throw std::runtime_error("LdrColor contains a nonfinite component");
        const size_t channel = i % channels;
        if (channel < 3)
            image.rgb[(i / channels) * 3 + channel] =
                static_cast<unsigned char>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
    }
    return image;
}

// Encode RGB pixels as PNG and fail if encoding, writing, or closing fails.
void writePng(const Image& image, const std::filesystem::path& path);
} // namespace example
#endif
