#pragma once

#include <cstdint>

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
    const float door_presence_probability,
    const float door_mask_probability,
    const float handle_presence_probability,
    const float handle_mask_probability,
    const float door_presence_threshold,
    const float door_mask_threshold,
    const float handle_presence_threshold,
    const float handle_mask_threshold)
{
    const bool handle_detected =
        handle_presence_probability >= handle_presence_threshold &&
        handle_mask_probability >= handle_mask_threshold;
    if (handle_detected)
    {
        return {1, handle_mask_probability};
    }

    const bool door_detected =
        door_presence_probability >= door_presence_threshold &&
        door_mask_probability >= door_mask_threshold;
    if (door_detected)
    {
        return {0, door_mask_probability};
    }

    return {255, 0.0F};
}

#undef SAM3_HOST_DEVICE
