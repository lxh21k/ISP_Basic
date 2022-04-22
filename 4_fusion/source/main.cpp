#include <opencv2/opencv.hpp>
#include "./Fusion/MultiFusion.h"
using namespace std;
// using namespace cv;

vector<cv::Mat> readImages(cv::String &filename);

int main(){

	vector<cv::Mat> images;

	cv::String filename = "./images/*.jpg";
	images = readImages(filename);

	// cv::imwrite("result.jpg", images[0]);

	
	MultiFusion mf(images);
	mf.Execute();
	// //save result
    // mf.GetResult();
    return 0;

}

vector<cv::Mat> readImages(cv::String &filename)
{
	vector<cv::String> fn;
	cv::glob(filename, fn, false);

	vector<cv::Mat> images;

	for (int i = 0; i < fn.size(); i++)
	{
		cout << "Reading image " << fn[i] << endl;
		cv::Mat img = cv::imread(fn[i]);
		if (img.empty())
		{
			std::cout << "Can not read image " << fn[i] << std::endl;
			continue;
		}
		images.push_back(img);
	}

	return images;
}