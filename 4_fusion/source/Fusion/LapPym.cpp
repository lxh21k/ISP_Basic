//
//  LapPym.cpp
//  exposureFusiion
//
//  Created by Jia Lanpeng on 2021/7/8.
//  Copyright © 2021 Jia Lanpeng. All rights reserved.
//

#include "LapPym.hpp"


void gaussian_pyramid(cv::Mat &I, std::vector<cv::Mat> &pyra){
  
}


std::vector<cv::Mat>  allocatePyramid(int r,int c,int type){
    std::vector<cv::Mat> pyr;
 
    
    return pyr;
}

void downsample32FC1(cv::Mat &I, cv::Mat &R, int levelnumber){
   
}

void downsample32FC3(cv::Mat &I, cv::Mat &R, int levelnumber){
   
}

void upsample(cv::Mat &I, cv::Mat &R){
   
}

void laplacian_pyramid(cv::Mat &I ,std::vector<cv::Mat > &lap_pym,std::vector<cv::Mat > &lap_pym_aux,int pym_num ){
  
}
cv::Mat reconstruct_laplacian_pyramid(std::vector<cv::Mat > &lap_pym,std::vector<cv::Mat > &lap_pym_aux,int pym_num){
    cv::Mat result ;
   
    return result;
}
cv::Mat repmatchannelx3(cv::Mat &single){
    cv::Mat result;
    
    
    return result;
}

