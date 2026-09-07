#pragma once

#include "semantic_selection.hpp"
#include <filesystem>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>
#include <string>

inline void save_segmentation_outputs(
    const cv::Mat& image,
    const cv::Mat& mask,
    const std::filesystem::path& output_dir,
    const std::filesystem::path& input_name,
    bool save_vis = false,
    float vis_alpha = 0.5F)
{
    if (image.empty() || image.type() != CV_8UC3 ||
        mask.type() != CV_8UC1 || mask.size() != image.size())
    {
        throw std::runtime_error(
            "Saving segmentation requires a BGR image and an equally sized CV_8UC1 mask");
    }

    // Keep the source extension to distinguish e.g. frame.jpg and frame.png.
    const std::string output_name = input_name.filename().string() + ".png";
    const auto mask_dir = output_dir / "masks";
    std::filesystem::create_directories(mask_dir);
    const auto mask_path = mask_dir / output_name;
    if (!cv::imwrite(mask_path.string(), mask))
    {
        throw std::runtime_error("Failed to save mask: " + mask_path.string());
    }

    if (!save_vis)
    {
        return;
    }

    // Visualize the final labels; background pixels retain the original image.
    cv::Mat colors = image.clone();
    colors.setTo(cv::Scalar(0, 185, 118), mask == sam3_door_label);
    colors.setTo(cv::Scalar(230, 159, 0), mask == sam3_handle_label);
    cv::Mat vis;
    cv::addWeighted(image, 1.0F - vis_alpha, colors, vis_alpha, 0.0, vis);

    const auto vis_dir = output_dir / "vis";
    std::filesystem::create_directories(vis_dir);
    const auto vis_path = vis_dir / output_name;
    if (!cv::imwrite(vis_path.string(), vis))
    {
        throw std::runtime_error("Failed to save visualization: " + vis_path.string());
    }
}
