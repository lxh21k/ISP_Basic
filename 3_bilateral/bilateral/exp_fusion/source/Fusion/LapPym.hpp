//
//  LapPym.hpp
//  exposureFusiion
//
//  Created by Jia Lanpeng on 2021/7/8.
//  Copyright © 2021 Jia Lanpeng. All rights reserved.
//

#ifndef LapPym_hpp
#define LapPym_hpp

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <vector>

void gaussian_pyramid(cv::Mat &I, std::vector<cv::Mat> &pyra);

void downsample32FC1(cv::Mat &I, cv::Mat &R, int levelnumber);

void downsample32FC3(cv::Mat &I, cv::Mat &R, int levelnumber);
void upsample(cv::Mat &I, cv::Mat &R);
void laplacian_pyramid(cv::Mat &I);
void laplacian_pyramid(cv::Mat &I ,std::vector<cv::Mat > &lap_pym,std::vector<cv::Mat > &lap_pym_aux,int pym_num );
std::vector<cv::Mat>  allocatePyramid(int r,int c,int type);
cv::Mat reconstruct_laplacian_pyramid(std::vector<cv::Mat > &lap_pym,std::vector<cv::Mat > &lap_pym_aux,int pym_num);
cv::Mat repmatchannelx3(cv::Mat &single);
#endif /* LapPym_hpp */
