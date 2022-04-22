//
//  LapPym.cpp
//  exposureFusiion
//
//  Created by Jia Lanpeng on 2021/7/8.
//  Copyright © 2021 Jia Lanpeng. All rights reserved.
//

#include "LapPym.hpp"

using namespace std;

void gaussian_pyramid(cv::Mat &I, std::vector<cv::Mat> &pyra){

    cv::Mat temp;
    I.copyTo(temp);
    for(int i=0; i < pym_depth; i++){
        pyra.push_back(temp);
        cv::pyrDown(temp, temp);
    }

    // for(int i=0; i < pyra.size(); i++){
    //     cv::imwrite("../results/pyramids/gaussian_pyramid_" + to_string(i) + ".jpg", pyra[i]);
    // }
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

void laplacian_pyramid(cv::Mat &I ,vector<cv::Mat > &lap_pym){
    cv::Mat input_imgs;
    I.copyTo(input_imgs);

    vector<cv::Mat> gaus_pym;
    gaussian_pyramid(input_imgs, gaus_pym);
    
    lap_pym.resize(pym_depth);
    lap_pym[0] = gaus_pym[pym_depth-1];

    for(int i=1; i < pym_depth; i++){
        cv::pyrUp(gaus_pym[pym_depth-i], lap_pym[i], gaus_pym[pym_depth-i-1].size());
        
        cv::subtract(gaus_pym[pym_depth-i-1], lap_pym[i], lap_pym[i]);
        
    }

    // for(int i=0; i < lap_pym.size(); i++){
    //     cv::imwrite("../results/pyramids/laplacian_pyramid_" + to_string(i) + ".jpg", lap_pym[i]);
    // }
}
cv::Mat reconstruct_laplacian_pyramid(vector<cv::Mat > &lap_pym){
    cv::Mat result;
    lap_pym[0].copyTo(result);
    
    for(int i=1; i < pym_depth; i++){
        cv::pyrUp(result, result, lap_pym[i].size());
        cv::add(result, lap_pym[i], result);
    }

    return result;
}
cv::Mat repmatchannelx3(cv::Mat &single){
    cv::Mat result;
    
    
    return result;
}

