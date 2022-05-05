#include "adaptive_unsharp.h"

using namespace std;

AdaptiveUnsharp::AdaptiveUnsharp(cv::Mat &img) {
    this->ori_img = img;
}

AdaptiveUnsharp::~AdaptiveUnsharp() {

}

void AdaptiveUnsharp::SharpImg() {
    cv::Mat lap_kernel = (cv::Mat_<double>(3,3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);
    cv::filter2D(gray_img, sharp_img, gray_img.depth(), lap_kernel);
    

    // cv::imwrite("../result/sharp_img.png", sharp_img);
}

void AdaptiveUnsharp::UnsharpMask(cv::Mat &src, cv::Mat &dst, float low_thd, float high_thd) {

    int kernel_size = 3;
    int pad_w = (kernel_size - 1) / 2;
    int pad_h = (kernel_size - 1) / 2;

    dst = cv::Mat::zeros(src.rows, src.cols, src.type());

    cv::Mat mid_map = cv::Mat::zeros(src.rows, src.cols, src.type());
    cv::Mat high_map = cv::Mat::zeros(src.rows, src.cols, src.type());

    cv::Mat mean_map;
    cv::Mat mean_kernel = cv::Mat::ones(kernel_size, kernel_size, src.type());
    mean_kernel /= 9;
    cv::filter2D(src, mean_map, src.depth(), mean_kernel);

    cv::Mat pad_src;
    cv::copyMakeBorder(src, pad_src, pad_h, pad_h, pad_w, pad_w, cv::BORDER_REPLICATE);

    for (int y=0; y < src.rows; y++) {
        float *mean_ptr = mean_map.ptr<float>(y);
        // float *dst_ptr = dst.ptr<float>(y);
        float *mid_ptr = mid_map.ptr<float>(y);
        float *high_ptr = high_map.ptr<float>(y);


        for (int x=0; x < src.cols; x++) {
            float var = 0.;
            for (int i=-pad_h; i <= pad_h; i++) {

                float *src_ptr = pad_src.ptr<float>(y+i+pad_h);
                for (int j=-pad_w; j <= pad_w; j++) {
                    float cur = src_ptr[x+j+pad_w] - mean_ptr[x];
                    var += (cur * cur);
                }
            }
            var /= (kernel_size * kernel_size);
            var = var * 255 * 255;

            if(var < low_thd)
                continue;
            else if(var > high_thd)
                high_ptr[x] = 1.0;
            else
                mid_ptr[x] = 1.0;

        }
    }

    dst += (mid_map + 0.5 * high_map);

    // dst.convertTo(dst, CV_8UC1, 255.);
    // cv::imwrite("../result/unsharp_mask.png", dst);

    // mid_map.convertTo(mid_map, CV_8UC1, 255.);
    // high_map.convertTo(high_map, CV_8UC1, 255.);
    // cv::imwrite("../result/mid_map.png", mid_map);
    // cv::imwrite("../result/high_map.png", high_map);


}

void AdaptiveUnsharp::Execute() {

    cv::cvtColor(ori_img, gray_img, CV_BGR2GRAY);

    
    SharpImg();
    UnsharpMask(gray_img, unsharp_mask, 6.0, 25.0);
    cv::GaussianBlur(unsharp_mask, unsharp_mask, cv::Size(3, 3), 0, 0);

    unsharp_res = unsharp_mask.mul(sharp_img);

    result = gray_img + alpha * unsharp_res;

    result.convertTo(result, CV_8UC1, 255.);
    cv::imwrite("../result/result_gaus_mask.png", result);
}