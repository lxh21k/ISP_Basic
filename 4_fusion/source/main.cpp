#include <opencv2/opencv.hpp>
#include "Fusion/MultiFusion.h"
using namespace std;
using namespace cv;

int main(){
    
	vector<cv::Mat> images;
   
	MultiFusion mf(images);
	mf.Execute();
	//save result
    mf.GetResult();
    return 0;

}
