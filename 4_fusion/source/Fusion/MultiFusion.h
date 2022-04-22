
#include <opencv2/opencv.hpp>
// #include <vector>
#ifndef __MULTIFUSION__
#define __MULTIFUSION__

using namespace std;

class MultiFusion{

private:
	int height = 500;
	int width = 752;
	float w_s = 1.;
	float w_c = 1.;
	float w_e = 1.;

	vector<cv::Mat> imgs;

	vector<cv::Mat> contrast_weights;
	vector<cv::Mat> saturation_weights;
	vector<cv::Mat> exposedness_weights;
	vector<cv::Mat> fusion_weights;
	vector<cv::Mat> norm_fusion_weights;

	vector<vector<cv::Mat>> weights_gaussian_pyramid;
	vector<vector<cv::Mat>> imgs_laplacian_pyramid;
	vector<cv::Mat> fused_laplacian_pyramid;

	cv::Mat fusion_result;

	cv::Mat d_result;
	cv::Mat m_result;

private:
	void Contrast(vector<cv::Mat> &imgs);
	void Saturation(vector<cv::Mat> &imgs);
	void WellExposedness(vector<cv::Mat> &imgs);
	void MakeWeights();
	cv::Mat DirectFusion(vector<cv::Mat> &imgs);
	vector<cv::Mat> PyramidFusion(vector<cv::Mat> &imgs);

public:
    MultiFusion(const vector<cv::Mat> &imgs);
	~MultiFusion();
	void Execute();
    cv::Mat GetResult();
};
#endif
