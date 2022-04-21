#include <opencv2/photo.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>
#include <fstream>
using namespace cv;
using namespace std;

// Read Images
void readImages(vector<Mat> &images)
{
  int numImages = 4;
  static const char* filenames[] = 
  {
    "../images/A.jpg",
    "../images/B.jpg",
    "../images/C.jpg",
    "../images/D.jpg"
  };

  for(int i=0; i < numImages; i++)
  {
    Mat im = imread(filenames[i]);
    images.push_back(im);
  }
}

int main(int argc, char **argv)
{
  cout << "Reading images ..." << endl;
  vector<Mat> images;

  readImages(images);

  cout << "Merging using Exposure Fusion ..." << endl;
  Mat exposureFusion;
  Ptr<MergeMertens> mergeMertens = createMergeMertens();
  mergeMertens->process(images, exposureFusion);

  cout << "Saving output ... exposure-fusion.jpg"<< endl;
  imwrite("exposure-fusion.jpg", exposureFusion * 255);

  return EXIT_SUCCESS;
}
