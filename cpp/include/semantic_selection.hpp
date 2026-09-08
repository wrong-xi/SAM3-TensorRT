#pragma once

#include <cstdint>

inline constexpr std::uint8_t sam3_handle_label = 128;
inline constexpr std::uint8_t sam3_background_label = 255;

struct SemanticClassSelection
{
    std::uint8_t class_label;
    float mask_probability;
};

#if defined(__CUDACC__)
#define SAM3_HOST_DEVICE __host__ __device__
#else
#define SAM3_HOST_DEVICE
#endif

SAM3_HOST_DEVICE inline SemanticClassSelection select_semantic_class(
    const float handle_presence_probability,
    const float handle_mask_probability,
    const float handle_presence_threshold,
    const float handle_mask_threshold)
{
    const bool handle_detected =
        handle_presence_probability >= handle_presence_threshold &&
        handle_mask_probability >= handle_mask_threshold;
    if (handle_detected)
    {
        return {sam3_handle_label, handle_mask_probability};
    }

    return {sam3_background_label, 0.0F};
}

#undef SAM3_HOST_DEVICE
