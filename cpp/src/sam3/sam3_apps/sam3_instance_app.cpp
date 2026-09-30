#include "sam3.cuh"
#include "instance_selection.hpp"
#include <iomanip>

int main(int argc, char** argv)
{
    if (argc < 4 || argc > 5)
    {
        std::cerr << "usage: sam3_instance_app IMAGE ENGINE NEW_OUTPUT_DIR [CURRENT_FRAME_BINARY_MASK]\n";
        return 2;
    }
    try
    {
        const std::filesystem::path output = argv[3];
        if (std::filesystem::exists(output))
            throw std::runtime_error("Output directory must not exist");
        cv::Mat input = cv::imread(argv[1], cv::IMREAD_COLOR);
        if (input.empty()) throw std::runtime_error("Cannot read image");
        cv::Mat scratch(input.size(), CV_8UC1, cv::Scalar(0));
        cv::Mat reference;
        if (argc == 5)
        {
            reference = cv::imread(argv[4], cv::IMREAD_UNCHANGED);
            if (reference.empty() || reference.type() != CV_8UC1 || reference.size() != input.size())
                throw std::runtime_error("Reference must be original-size gray8 binary mask");
            // 只接受 0/1 或 0/255，禁止误读 128/255 语义类别图。
            cv::Mat invalid = (reference != 0) & (reference != 1) & (reference != 255);
            if (cv::countNonZero(invalid)) throw std::runtime_error("Reference is not binary");
        }
        SAM3_PCS model(argv[2], 0.6F, 0.5F);
        if (!model.has_instances()) throw std::runtime_error("Four-output instance engine required");
        model.pin_opencv_matrices(input, scratch);
        if (!model.infer_on_image(input, scratch, SAM3_VISUALIZATION::VIS_NONE))
            throw std::runtime_error("SAM3 inference failed");
        const auto instances = decode_instances(model.output_host("pred_masks"),
            model.output_host("pred_logits"), model.output_host("presence_logits")[0],
            model.output_host("pred_boxes"), model.instance_count(),
            model.semantic_mask_width(), model.semantic_mask_height(), input.size());
        const auto selected = select_instance(instances, reference);
        std::filesystem::create_directories(output);
        std::ofstream report(output / "instances.tsv");
        if (!report) throw std::runtime_error("Cannot create report");
        report << "query\tscore\tx1\ty1\tx2\ty2\tselected\n" << std::setprecision(9);
        for (std::size_t i = 0; i < instances.size(); ++i)
        {
            const auto& item = instances[i];
            const auto filename = output / ("query_" + std::to_string(item.query_index) + ".png");
            if (!cv::imwrite(filename.string(), item.mask)) throw std::runtime_error("Cannot save mask");
            report << item.query_index << '\t' << item.score;
            for (int j = 0; j < 4; ++j) report << '\t' << item.box[j];
            report << '\t' << (selected.index == static_cast<int>(i)) << '\n';
        }
        if (selected.index >= 0 && !cv::imwrite((output / "selected.png").string(),
                instances[selected.index].mask)) throw std::runtime_error("Cannot save selected mask");
        report.flush();
        if (!report) throw std::runtime_error("Cannot write report");
        std::cout << "candidates=" << instances.size() << " reason=" << selected.reason
                  << " selected_index=" << selected.index << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "sam3_instance_app: " << error.what() << '\n';
        return 1;
    }
}
