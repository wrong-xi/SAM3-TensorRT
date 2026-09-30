#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>

struct Sam3Instance
{
    int query_index;
    float score;
    cv::Vec4f box; // 原图 xyxy
    cv::Mat mask; // 二值 0/255，与语义标签 128/255 不同
};

struct Sam3Selection
{
    int index = -1;
    const char* reason = "no_target";
};

inline std::vector<Sam3Instance> decode_instances(const float* masks,
    const float* logits, float presence, const float* boxes, int count,
    int width, int height, cv::Size size, float confidence = 0.6F)
{
    if (!masks || !logits || !boxes || count <= 0 || width <= 0 || height <= 0 ||
        size.width <= 0 || size.height <= 0 || !(confidence > 0 && confidence <= 1) ||
        !std::isfinite(presence)) throw std::runtime_error("Invalid instance inputs");
    const auto sigmoid = [](float x) { return 1.0F / (1.0F + std::exp(-x)); };
    const auto area = static_cast<std::size_t>(width) * height;
    std::vector<Sam3Instance> result;
    for (int q = 0; q < count; ++q)
    {
        if (!std::isfinite(logits[q])) throw std::runtime_error("Non-finite score");
        const float score = sigmoid(logits[q]) * sigmoid(presence);
        if (score <= confidence) continue;
        cv::Mat low(height, width, CV_32FC1), resized, binary;
        const auto* begin = masks + static_cast<std::size_t>(q) * area;
        std::copy(begin, begin + area, low.ptr<float>());
        if (!cv::checkRange(low)) throw std::runtime_error("Non-finite mask logits");
        // 沿用官方 PC 的顺序：先插值 logits；sigmoid(x)>0.5 等价于 x>0。
        cv::resize(low, resized, size, 0, 0, cv::INTER_LINEAR);
        cv::compare(resized, 0, binary, cv::CMP_GT);
        cv::Vec4f box;
        for (int j = 0; j < 4; ++j)
        {
            if (!std::isfinite(boxes[q * 4 + j])) throw std::runtime_error("Non-finite box");
            box[j] = boxes[q * 4 + j] * ((j % 2 == 0) ? size.width : size.height);
        }
        result.push_back({q, score, box, binary});
    }
    return result;
}

// reference 必须来自当前帧；没有参考时不按面积或分数擅自选择多个候选。
inline Sam3Selection select_instance(const std::vector<Sam3Instance>& instances,
    const cv::Mat& reference = {}, double min_iou = 0.2)
{
    if (!(min_iou >= 0 && min_iou <= 1)) throw std::runtime_error("Invalid IoU threshold");
    if (instances.empty()) return {};
    if (reference.empty()) return instances.size() == 1 ? Sam3Selection{0, "selected"} :
        Sam3Selection{-1, "ambiguous_targets"};
    if (reference.type() != CV_8UC1) throw std::runtime_error("Expected gray8 reference");
    const cv::Mat ref = reference != 0;
    int best = -1;
    double best_iou = -1;
    for (std::size_t i = 0; i < instances.size(); ++i)
    {
        const auto& mask = instances[i].mask;
        if (mask.size() != ref.size() || mask.type() != CV_8UC1)
            throw std::runtime_error("Candidate/reference dimensions differ");
        cv::Mat overlap, combined;
        cv::bitwise_and(mask, ref, overlap);
        cv::bitwise_or(mask, ref, combined);
        const double iou = static_cast<double>(cv::countNonZero(overlap)) /
            std::max(cv::countNonZero(combined), 1);
        if (iou > best_iou) { best_iou = iou; best = static_cast<int>(i); }
    }
    return best_iou >= min_iou ? Sam3Selection{best, "selected"} :
        Sam3Selection{-1, "target_association_failed"};
}
