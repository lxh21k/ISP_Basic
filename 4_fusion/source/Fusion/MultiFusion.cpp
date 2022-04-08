#include "MultiFusion.h"
//#include "LapPym.hpp"

MultiFusion::MultiFusion(std::vector<cv::Mat> &rgb){

}

// paper chapter 3.1
void MultiFusion::Contrast(){
	
	
}
// paper chapter 3.1
void MultiFusion::Saturation(){
    
	
}
// paper chapter 3.1
void MultiFusion::WellExposedness(){
	
	
}
// paper chapter 3.2
void MultiFusion::MakeWeights(){
	
}

// using Figure3 which reconstructed the image by laplacian pym.
// return the result in "m_result"
void MultiFusion::Execute(){

	
    std::cout << "Calc Fusion Weights" << std::endl;
	Contrast();  
	Saturation();   
	WellExposedness();  
	MakeWeights();
    
    std::cout << "Merging" << std::endl;
    
    std::cout << "Reconstruct" << std::endl;
}

MultiFusion::~MultiFusion(){

}
cv::Mat MultiFusion::GetResult()
{
    std::cout << "return result " << std::endl;
    return m_result;
}
