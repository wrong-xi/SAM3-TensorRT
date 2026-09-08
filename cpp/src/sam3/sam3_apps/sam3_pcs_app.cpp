#include "sam3.hpp"
#include "sam3.cuh"
#include "segmentation_output.hpp"
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

    auto start = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> diff;
    float millis_elapsed = 0.0; // int will overflow after ~650 hours

    const float vis_alpha = 0.6;
    const SAM3_CLASS_THRESHOLDS handle_thresholds = {0.5F, 0.5F};

    SAM3_PCS pcs(
        epath,
        vis_alpha,
        handle_thresholds);

    cv::Mat img, result;
    char* raw_bytes;

    int num_images_read=0;

    for (const auto& fname : std::filesystem::directory_iterator(in_dir))
    {
        if (std::filesystem::is_regular_file(fname.path())) 
        {
            const std::string image_path = fname.path().string();
            
            if (num_images_read==0)
            {
                cv::Mat tmp = cv::imread(image_path, cv::IMREAD_COLOR);
                raw_bytes = (char *)malloc(tmp.total()*tmp.elemSize());
                read_image_into_buffer(image_path, raw_bytes, img);
                result = cv::Mat(img.size(), CV_8UC1, cv::Scalar(sam3_background_label));
                pcs.pin_opencv_matrices(img, result);
            }
            else
            {
                read_image_into_buffer(image_path, raw_bytes, img);
            }
            start = std::chrono::system_clock::now();
            infer_one_image(pcs, img, result, fname.path().filename(), benchmark, save_vis, vis_alpha);
            num_images_read++;
            end = std::chrono::system_clock::now();
            diff = end - start;
            millis_elapsed += (diff.count() * 1000);

            if (num_images_read>0 && num_images_read%10==0)
            {
                float msec_per_image = millis_elapsed/num_images_read;
                printf("Processed %d images at %f msec/image\n", num_images_read, msec_per_image);
            }
        }
    }
}
