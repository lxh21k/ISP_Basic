#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;

class AdaptiveUnsharp {
private:
    int width = 4624;
    int height = 3472;
    float alpha = 0.6;

    cv::Mat ori_img;
    cv::Mat gray_img;
    cv::Mat sharp_img;
    cv::Mat unsharp_mask;
    cv::Mat unsharp_res;
    cv::Mat result;

private:
    void SharpImg();
    void UnsharpMask(cv::Mat &src, cv::Mat &dst, float low_thd, float high_thd);

public:
    AdaptiveUnsharp(cv::Mat &img);
    ~AdaptiveUnsharp();
    void Execute();
};