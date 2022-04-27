#include "MultiFusion.h"
#include "LapPym.hpp"

using namespace std;

MultiFusion::MultiFusion(const vector<cv::Mat> &imgs){
    // cout << "MultiFusion class is being created" << endl;
    this->imgs = imgs;
}

// paper chapter 3.1
void MultiFusion::Contrast(vector<cv::Mat> &imgs){
    contrast_weights.resize(imgs.size());

    
    for (int i=0; i < imgs.size(); i++){
        cv::Mat img = imgs[i];
        cv::Mat img_gray, c_weight;

        img.convertTo(img, CV_32FC3);
        img = img / 255;
        cv::cvtColor(img, img_gray, cv::COLOR_BGRA2GRAY);
        cv::Laplacian(img_gray, c_weight, CV_32FC1, 3);
        // cv::convertScaleAbs(c_weight, c_weight);
        c_weight = abs(c_weight);

        c_weight.convertTo(c_weight, CV_32FC1);
        contrast_weights[i] = c_weight;

        // cout << "Saving contrast weight image" << endl;
        // cv::Mat c_weight_save;
        // contrast_weights[i].convertTo(c_weight_save, CV_8UC3, 255.);
        // cv::imwrite("../results/contrast_weight_" + to_string(i) + ".jpg", c_weight_save);
    }
	
	
}
// paper chapter 3.1
void MultiFusion::Saturation(vector<cv::Mat> &imgs){
    saturation_weights.resize(imgs.size());

    for (int i=0; i < imgs.size(); i++){
        cv::Mat img = imgs[i];
        img.convertTo(img, CV_32FC3);
        img = img / 255;

        vector<cv::Mat> rgb;
        cv::split(img, rgb);

        cv::Mat mean = (rgb[0] + rgb[1] + rgb[2]) / 3.;
        vector<cv::Mat> variance;
        variance.resize(rgb.size());

        for (int j=0; j < rgb.size(); j++){
            cv::pow((rgb[j] - mean), 2, variance[j]);
        }

        cv::Mat std;
        cv::sqrt((variance[0] + variance[1] + variance[2]) / 3., std);

        saturation_weights[i] = std;

        // cout << "Saving saturation weight image" << endl;
        // cv::Mat s_weight_save;
        // saturation_weights[i].convertTo(s_weight_save, CV_8UC3, 255.);
        // cv::imwrite("../results/saturation_weight_" + to_string(i) + ".jpg", s_weight_save);

    }
	
}
// paper chapter 3.1
void MultiFusion::WellExposedness(vector<cv::Mat> &imgs){
    exposedness_weights.resize(imgs.size());
    float sigma = 0.2;

    for (int i=0; i < imgs.size(); i++){
        cv::Mat img = imgs[i];
        img.convertTo(img, CV_32FC3);
        img = img / 255;

        vector<cv::Mat> rgb;
        cv::split(img, rgb);


        for (int j=0; j < rgb.size(); j++){
            cv::exp(-0.5*((rgb[j] - 0.5).mul(rgb[j] - 0.5)) / (sigma*sigma), rgb[j]);
        }

        exposedness_weights[i] = rgb[0].mul(rgb[1]).mul(rgb[2]);

        // cout << "Saving well exposedness weight image" << endl;
        // cv::Mat e_weight_save;
        // exposedness_weights[i].convertTo(e_weight_save, CV_8UC3, 255.);
        // cv::imwrite("../results/exposedness_weight_" + to_string(i) + ".jpg", e_weight_save);
    }
	
	
}
// paper chapter 3.2
void MultiFusion::MakeWeights(){
    fusion_weights.resize(contrast_weights.size());
    norm_fusion_weights.resize(contrast_weights.size());
    cv::Mat weights_sum = cv::Mat(contrast_weights[0].rows, contrast_weights[0].cols, CV_32FC1, 0.);

    for (int i=0; i < fusion_weights.size(); i++){
        cv::pow(contrast_weights[i], w_c, contrast_weights[i]);
        cv::pow(saturation_weights[i], w_s, saturation_weights[i]);
        cv::pow(exposedness_weights[i], w_e, exposedness_weights[i]);
        fusion_weights[i] = contrast_weights[i].mul(saturation_weights[i]).mul(exposedness_weights[i]);
        // cout << contrast_weights[i].type() << endl;
        // cout << saturation_weights[i].type() << endl;
        // cout << exposedness_weights[i].type() << endl;
        weights_sum += fusion_weights[i];
    }

    weights_sum += 1e-12;
    for (int i=0; i < fusion_weights.size(); i++){
        norm_fusion_weights[i] = fusion_weights[i] / weights_sum;

        // cout << "Saving fusion weight image" << endl;
        // cv::Mat fusion_weight_save;
        // norm_fusion_weights[i].convertTo(fusion_weight_save, CV_8UC3, 255.);
        // cv::imwrite("../results/fusion_weight_" + to_string(i) + ".jpg", fusion_weight_save);
    }

}


cv::Mat MultiFusion::DirectFusion(vector<cv::Mat> &imgs) {
    cv::Mat fusion_img = cv::Mat(imgs[0].rows, imgs[0].cols, CV_32FC3, 0.);
    // vector<cv::Mat> weights = norm_fusion_weights;
    
    for (int i=0; i < imgs.size(); i++){
        cv::Mat img = imgs[i];
        img.convertTo(img, CV_32FC3);
        img = img / 255;

        cv::Mat img_weight = norm_fusion_weights[i]; //这里weight只有一个通道,跟那两个对不上
        
        // convert 1Channel weight map to 3Channels
        vector<cv::Mat> channels;
        for (int i=0; i < 3; i++){
            channels.push_back(img_weight);
        }
        cv::merge(&channels[0], 3, img_weight);
        
        fusion_img += img_weight.mul(img);
    }

    // fusion_img.convertTo(fusion_img, CV_8UC3, 255.);
    // cv::imwrite("../results/direct_fusion_img.jpg", fusion_img);
    return fusion_img;
}

vector<cv::Mat> MultiFusion::PyramidFusion(vector<cv::Mat> &imgs) {
    cv::Mat fusion_img = cv::Mat(imgs[0].rows, imgs[0].cols, CV_32FC3, 0.);

    weights_gaussian_pyramid.resize(imgs.size());
    imgs_laplacian_pyramid.resize(imgs.size());

    for (int i=0; i < imgs.size(); i++){
        cv::Mat img = imgs[i];
        img.convertTo(img, CV_32FC3);
        img = img / 255;

        gaussian_pyramid(norm_fusion_weights[i], weights_gaussian_pyramid[i]);
        laplacian_pyramid(img, imgs_laplacian_pyramid[i]);
    }

    int depth = weights_gaussian_pyramid[0].size();

	vector<cv::Mat> fused_laplacian_pyramid;
    fused_laplacian_pyramid.resize(depth);

    for (int layer=0; layer < depth; layer++){
        for (int i=0; i < imgs.size(); i++){
            cv::Mat layer_weight = weights_gaussian_pyramid[i][depth - layer - 1];
            cv::Mat layer_img = imgs_laplacian_pyramid[i][layer];

            vector<cv::Mat> channels;
            for (int i=0; i < 3; i++){
                channels.push_back(layer_weight);
            }
            cv::merge(&channels[0], 3, layer_weight);

            fused_laplacian_pyramid[layer] = cv::Mat(layer_img.rows, layer_img.cols, CV_32FC3, 0.);

            // cout << "Layer Img" << layer_img.type() << endl;
            // cout << "Layer Weight" << layer_weight.type() << endl;
            // cout << "Pyramid" << fused_laplacian_pyramid[layer].type() << endl;

            fused_laplacian_pyramid[layer] += layer_weight.mul(layer_img);
        }
        
    }

    // for (int layer=0; layer < weights_gaussian_pyramid[0].size(); layer++){
    //     cv::Mat fusion_img_layer = fused_laplacian_pyramid[layer];
    //     fusion_img_layer.convertTo(fusion_img_layer, CV_8UC3, 255.);
    //     cv::imwrite("../results/pyramids/fusion_img_layer_" + to_string(layer) + ".jpg", fusion_img_layer);
    // }
    return fused_laplacian_pyramid;

}

// using Figure3 which reconstructed the image by laplacian pym.
// return the result in "m_result"
void MultiFusion::Execute(){

	
    cout << "Calc Fusion Weights" << endl;
	Contrast(imgs);  
	Saturation(imgs);   
	WellExposedness(imgs);  
	MakeWeights();

    cout << "Direct Fushion" << endl;
    d_result = DirectFusion(imgs);

    cout << "Construct Laplacian Pyramid" << endl;
    // vector<cv::Mat> laplacian_pyr;
    // cv::Mat img0 = imgs[0];
    // laplacian_pyramid(img0, laplacian_pyr);

    // cout << "Reconstruct Image" << endl;
    // cv::Mat reconst_img = reconstruct_laplacian_pyramid(laplacian_pyr);
    // cv::imwrite("../results/reconst_img.jpg", reconst_img);
    
    cout << "Merging" << endl;
    
    vector<cv::Mat> fused_pym;
    fused_pym = PyramidFusion(imgs);
    m_result = reconstruct_laplacian_pyramid(fused_pym);
    m_result.convertTo(m_result, CV_8UC3, 255.);
    cv::imwrite("../results/multires_fused_img.jpg", m_result);

    
    cout << "Reconstruct" << endl;
}

MultiFusion::~MultiFusion(){

}
cv::Mat MultiFusion::GetResult()
{
    cout << "return result " << endl;
    return m_result;
}

