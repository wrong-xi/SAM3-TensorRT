#include "sam3.hpp"
#include "sam3.cuh"
#include "segmentation_output.hpp"
#include <array>
#include <chrono>
#include <thread>
#include <opencv2/imgproc.hpp>

void read_image_into_buffer(const std::string imgpath, char* raw_buffer, cv::Mat& buffer)
{
    size_t file_size = std::filesystem::file_size(imgpath);
    if (file_size==0)
    {
        std::stringstream err;
        err << "Image file is empty";
        throw std::runtime_error(err.str());
    }

    std::ifstream file(imgpath, std::ios::binary);

    if (!file.is_open())
    {
        std::stringstream err;
        err << "File " << imgpath << " could not be opened. Please check permissions\n";
        throw std::runtime_error(err.str());
    }

    file.read(raw_buffer, file_size);
    file.close();
    
    cv::Mat raw_mat(1, static_cast<int>(file_size), CV_8UC1, raw_buffer); 
    // just a wrapper, minimal allocation

    cv::imdecode(raw_mat, cv::IMREAD_COLOR, &buffer);
}

void infer_one_image(SAM3_PCS& pcs, 
    const cv::Mat& img, 
    cv::Mat& mask,
    const std::filesystem::path& input_name,
    bool benchmark_run,
    bool save_vis,
    float vis_alpha)
{
    if (!pcs.infer_on_image(img, mask, SAM3_VISUALIZATION::VIS_CLASS_MAP))
    {
        throw std::runtime_error("SAM3 inference failed for " + input_name.string());
    }

    if (benchmark_run)
    {
        return;
    }

    save_segmentation_outputs(img, mask, "results", input_name, save_vis, vis_alpha);
}

void report_timing(int count, double cpu_wall_ms,
    const std::array<double, 6>& gpu_totals, bool final_report)
{
    const char* prefix = final_report ? "Summary" : "Processed";
    if (count == 0)
    {
        printf("%s 0 images (no timing samples)\n", prefix);
        return;
    }
    printf("%s %d images | CPU wall (infer+save): %.3f ms/image\n",
        prefix, count, cpu_wall_ms / count);
    printf("  GPU cudaEvent avg (ms/image): preprocess=%.3f TensorRT=%.3f "
        "postprocess=%.3f H2D=%.3f D2H=%.3f total=%.3f\n",
        gpu_totals[0] / count, gpu_totals[1] / count, gpu_totals[2] / count,
        gpu_totals[3] / count, gpu_totals[4] / count, gpu_totals[5] / count);
}

int main(int argc, char* argv[])
{
    if (argc < 3 || argc > 5)
    {
        std::cout << "Usage: ./sam3_pcs_app indir engine_path.engine [benchmark=0] [save_vis=0]" << std::endl;
        return 1;
    }

    const std::string in_dir = argv[1];
    std::string epath = argv[2];
    bool benchmark=false; // in benchmarking mode we dont save output images
    bool save_vis=false;

    for (int arg_index = 3; arg_index < argc; ++arg_index)
    {
        const std::string value = argv[arg_index];
        if (value != "0" && value != "1")
        {
            std::cerr << "benchmark and save_vis must be 0 or 1" << std::endl;
            return 1;
        }
    }
    if (argc >= 4)
    {
        benchmark = (std::string(argv[3]) == "1");
    }
    if (argc == 5)
    {
        save_vis = (std::string(argv[4]) == "1");
    }
    std::cout << "Benchmarking: " << benchmark << std::endl;
    std::cout << "Save visualization: " << (save_vis && !benchmark) << std::endl;

    double cpu_wall_ms = 0.0;
    std::array<double, 6> gpu_totals{};
    constexpr int warmup_iterations = 5;

    const float vis_alpha = 0.6;
    const float handle_mask_threshold = 0.5F;

    SAM3_PCS pcs(
        epath,
        vis_alpha,
        handle_mask_threshold);

    cv::Mat img, result;
    char* raw_bytes;

    int num_images_read=0;

    for (const auto& fname : std::filesystem::recursive_directory_iterator(in_dir))
    {
        if (std::filesystem::is_regular_file(fname.path())) 
        {
            const std::string image_path = fname.path().string();
            const auto input_name = fname.path().lexically_relative(in_dir);
            
            if (num_images_read==0)
            {
                cv::Mat tmp = cv::imread(image_path, cv::IMREAD_COLOR);
                raw_bytes = (char *)malloc(tmp.total()*tmp.elemSize());
                read_image_into_buffer(image_path, raw_bytes, img);
                result = cv::Mat(img.size(), CV_8UC1, cv::Scalar(sam3_background_label));
                pcs.pin_opencv_matrices(img, result);
                for (int iteration = 0; iteration < warmup_iterations; ++iteration)
                {
                    infer_one_image(pcs, img, result, input_name,
                        true, false, vis_alpha);
                }
                printf("Warmup complete: %d iterations (excluded; no output saved)\n",
                    warmup_iterations);
            }
            else
            {
                read_image_into_buffer(image_path, raw_bytes, img);
            }
            const auto start = std::chrono::steady_clock::now();
            infer_one_image(pcs, img, result, input_name, benchmark, save_vis, vis_alpha);
            const auto end = std::chrono::steady_clock::now();
            cpu_wall_ms += std::chrono::duration<double, std::milli>(end - start).count();
            const auto timings = pcs.last_gpu_timings();
            const std::array<double, 6> frame_times = {timings.preprocess_ms,
                timings.tensorrt_ms, timings.postprocess_ms, timings.h2d_ms,
                timings.d2h_ms, timings.total_ms};
            for (size_t stage = 0; stage < gpu_totals.size(); ++stage)
            {
                gpu_totals[stage] += frame_times[stage];
            }
            num_images_read++;

            if (num_images_read>0 && num_images_read%10==0)
            {
                report_timing(num_images_read, cpu_wall_ms, gpu_totals, false);
            }
        }
    }
    report_timing(num_images_read, cpu_wall_ms, gpu_totals, true);
}
