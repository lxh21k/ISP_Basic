#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "adaptive_unsharp.h"

using namespace std;

int main() {
    cv::String filename="../data/in.png";
    cv::Mat img = cv::imread(filename);
    img.convertTo(img, CV_32FC3);
    img = img / 255.;

    AdaptiveUnsharp au(img);
    au.Execute();

}