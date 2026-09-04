#pragma once

#include "sam3.hpp"
#include <filesystem>
#include <fstream>
#include "cuda_runtime.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "prepost.cuh"

static void cuda_check(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::stringstream ss;
        ss << "Error in " << msg << ": " << cudaGetErrorString(err);
        throw std::runtime_error(ss.str());
    }
}

class TRTLogger : public nvinfer1::ILogger
{
    public:
        void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        if (severity <= nvinfer1::ILogger::Severity::kVERBOSE)
        {
            std::cout << "\033[1;35m [TRT] " << msg << "\033[0m\n";
        }
    }
};

inline TRTLogger trt_logger{};

class SAM3_PCS
{
public:
    SAM3_PCS(
        const std::string& engine_path,
        float vis_alpha,
        SAM3_CLASS_THRESHOLDS door_thresholds,
        SAM3_CLASS_THRESHOLDS handle_thresholds);
    ~SAM3_PCS();
    bool infer_on_image(const cv::Mat& input, cv::Mat& result, SAM3_VISUALIZATION vis_type);
    bool run_blind_inference();
    void pin_opencv_matrices(cv::Mat& input_mat, cv::Mat& result_mat);
    const float* semantic_logits_host() const noexcept;
    const float* presence_logits_host() const noexcept;
    int semantic_mask_width() const noexcept;
    int semantic_mask_height() const noexcept;
    std::vector<void*> output_cpu;

private:
    bool is_zerocopy; //check if we can use zero-copy on this platform
    cudaStream_t sam3_stream;
    dim3 bsize;
    dim3 gsize;
    int in_width, in_height, opencv_inbytes, opencv_resultbytes;
    int mask_width, mask_height;
    int semantic_output_index = -1;
    int presence_output_index = -1;

    std::vector<void*> input_cpu;
    std::vector<void*> input_gpu;
    int image_index=0;

    std::vector<void*> output_gpu;
    std::vector<size_t>output_sizes;

    void* opencv_input; // used only if dGPU
    uint8_t* gpu_result; // used for both
    uint8_t* zc_input; // used only if iGPU

    uint8_t* input_ptr; // placeholder for dGPU/iGPU ptr to pass into kernel
    void check_zero_copy();
    void allocate_io_buffers();
    void load_engine();
    bool infer_on_dGPU(const cv::Mat& input, cv::Mat& result, SAM3_VISUALIZATION vis_type);
    bool infer_on_iGPU(const cv::Mat& input, cv::Mat& result, SAM3_VISUALIZATION vis_type);

    void visualize_on_dGPU(const cv::Mat& input, cv::Mat& result, SAM3_VISUALIZATION vis_type);
    const float _overlay_alpha;
    const SAM3_CLASS_THRESHOLDS _door_thresholds;
    const SAM3_CLASS_THRESHOLDS _handle_thresholds;

    const std::string _engine_path;

    std::vector<std::string> _input_names;
    std::vector<std::string> _output_names;

    std::unique_ptr<nvinfer1::IRuntime> trt_runtime; // highest level
    std::unique_ptr<nvinfer1::ICudaEngine> trt_engine; // lower than runtime
    std::unique_ptr<nvinfer1::IExecutionContext> trt_ctx; // lower than engine
    
};
