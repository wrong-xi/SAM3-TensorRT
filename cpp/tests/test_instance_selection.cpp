#include "instance_selection.hpp"
#include <iostream>

void check(bool condition) { if (!condition) throw std::runtime_error("Test failed"); }

int main()
{
    try
    {
        const float masks[] = {-2, 2, 2, -2, 2, -2, -2, 2};
        const float scores[] = {10, 10};
        const float boxes[] = {0, 0, 1, 1, 0, 0, 1, 1};
        auto items = decode_instances(masks, scores, 10, boxes, 2, 2, 2, {2, 2});
        check(items.size() == 2 && cv::countNonZero(items[0].mask) == 2);
        check(items[0].mask.at<unsigned char>(0, 0) == 0);
        check(items[0].box[2] == 2 && items[0].box[3] == 2);
        check(std::string(select_instance(items).reason) == "ambiguous_targets");
        check(select_instance(items, items[1].mask).index == 1);
        check(std::string(select_instance(items, cv::Mat::zeros(2,2,CV_8UC1)).reason)
              == "target_association_failed");
        items.resize(1);
        check(select_instance(items).index == 0);
        check(std::string(select_instance({}).reason) == "no_target");
        check(decode_instances(masks, scores, -10, boxes, 2, 2, 2, {2, 2}).empty());
        const float zero[] = {0};
        check(decode_instances(zero, zero, 0, boxes, 1, 1, 1, {1, 1}, 0.25F).empty());
        // 零 logit 不应成为前景；插值后再判定，输出为原图大小。
        auto resized = decode_instances(zero, scores, 10, boxes, 1, 1, 1, {5, 3});
        check(resized[0].mask.size() == cv::Size(5,3));
        check(cv::countNonZero(resized[0].mask) == 0);
        cv::Mat five = cv::Mat::ones(1, 5, CV_8UC1) * 255;
        cv::Mat one = cv::Mat::zeros(1, 5, CV_8UC1);
        one.at<unsigned char>(0, 0) = 255;
        std::vector<Sam3Instance> edge{{0, 1.0F, {}, one}};
        check(select_instance(edge, five, 0.2).index == 0);
        check(select_instance(edge, five, 0.21).index == -1);
        edge.push_back(edge[0]);
        check(select_instance(edge, five).index == 0); // IoU 相同保留首个候选
        // 两类使用不同 mask/分数/presence/box，任何一处偏移错位都应被发现。
        const float class_masks[] = {2,-2,-2,-2, -2,-2,-2,2};
        const float class_scores[] = {2,10}, class_presence[] = {10,2};
        const float class_boxes[] = {0,0,.5F,.5F, .5F,.5F,1,1};
        auto door = decode_instance_class(class_masks,class_scores,class_presence,class_boxes,
            2,0,1,2,2,{2,2});
        auto handle = decode_instance_class(class_masks,class_scores,class_presence,class_boxes,
            2,1,1,2,2,{2,2});
        check(door.size() == 1 && handle.size() == 1);
        check(door[0].mask.at<unsigned char>(0,0) == 255);
        check(handle[0].mask.at<unsigned char>(1,1) == 255);
        check(door[0].box[2] == 1 && handle[0].box[0] == 1);
        const float presence_off[] = {-10,10};
        check(decode_instance_class(class_masks,class_scores,presence_off,class_boxes,
            2,0,1,2,2,{2,2}).empty());
        check(decode_instance_class(class_masks,class_scores,presence_off,class_boxes,
            2,1,1,2,2,{2,2}).size() == 1);
        bool rejected = false;
        try { decode_instance_class(class_masks,class_scores,class_presence,class_boxes,
            2,2,1,2,2,{2,2}); } catch (const std::runtime_error&) { rejected = true; }
        check(rejected);
        std::cout << "instance selection tests passed\n";
        return 0;
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
