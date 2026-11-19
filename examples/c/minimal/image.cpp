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
#include "image.h"

#include <cstdlib>
#include <fstream>
#include <memory>

// Keep the vendored header identical to the ovrtx minimal example. File I/O is
// ours: stb's stdio writer does not report failed fwrite/fclose operations.
#define STBI_WRITE_NO_STDIO
#define STB_IMAGE_WRITE_IMPLEMENTATION
#if defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#include "stb_image_write.h"
#if defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif

namespace example
{
void writePng(const Image& image, const std::filesystem::path& path)
{
    // stb uses signed int sizes internally. Bound both the raw image and its
    // compression workspace before narrowing dimensions or calling the encoder.
    constexpr int64_t limit = std::numeric_limits<int>::max() / 2;
    if (image.width <= 0 || image.height <= 0 || image.width > (limit - 1) / 3 ||
        image.height > limit / (image.width * 3 + 1) ||
        image.rgb.size() != static_cast<size_t>(image.width * image.height * 3))
        throw std::runtime_error("invalid RGB image dimensions or buffer size for PNG encoding");

    int size = 0;
    const std::unique_ptr<unsigned char, decltype(&std::free)> encoded(
        stbi_write_png_to_mem(image.rgb.data(), static_cast<int>(image.width * 3),
                              static_cast<int>(image.width), static_cast<int>(image.height), 3, &size),
        &std::free);
    if (!encoded || size <= 0)
        throw std::runtime_error("PNG encoding failed");
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream file;
    file.exceptions(std::ios::failbit | std::ios::badbit);
    file.open(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(encoded.get()), size);
    file.close();
}
} // namespace example
