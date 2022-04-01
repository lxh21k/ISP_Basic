
#include <opencv2/opencv.hpp>
#include <vector>
#ifndef __MULTIFUSION__
#define __MULTIFUSION__

class MultiFusion{

private:
	cv::Mat m_result;
private:
	void Contrast();
	void Saturation();
	void WellExposedness();
	void MakeWeights();

public:
    MultiFusion(std::vector<cv::Mat> &rgb);
	~MultiFusion();
	void Execute();
    cv::Mat GetResult();
};
#endif
