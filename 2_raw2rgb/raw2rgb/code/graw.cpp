#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include "graw.h"

// static std::vector<uint16_t> getGammaLUT(std::vector<uint16_t> org_gamma_tb) {
    
//     // gamma表差值
    
// }

// void bilinear_resize(const cv::Mat& small, cv::Mat& big, int width, int height){
//     // 可尝试自己实现双线性插值
// }


// void bilinear_demosaicing(const cv::Mat& src, cv::Mat& dst, int ori_height, int ori_width){

//     // 可尝试自己实现demosaicing

// }


void GRaw::read_file(const std::string& path){

    // 读入二进制数据

}

void GRaw::bayer_cvt(){

    // 解析 bayerpattern ,并保存为 4 通道 (RGGB) 格式

}

void GRaw::linear_scale(){

    // 线性化 / 黑电平校正

}

void GRaw::apply_gain(){

    // 白平衡

}

void GRaw::apply_lsc(){
    
    // lsc

}

void GRaw::apply_demosaic(){
    
    // 去马塞克

}


void GRaw::apply_ccm(){
   
   //ccm

}

void GRaw::apply_gamma(){
    
    //gamma

}


void GRaw::write_bgr(const std::string& path){
    
    // 将中间过程均以rgb的形式输出显示

}