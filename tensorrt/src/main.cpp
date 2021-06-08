#include <vector>
#include <iostream>

#include "lanedetector.hpp"

using namespace cv;
using namespace std;

int main() 
{
    cv::Mat input_img = imread("../../download/lishui_tl.png",1);
    LaneDetector lane_detector;

    lane_detector.detect(input_img);

    // net_ptr_->doInference(input_data.data(), output_data.get());

    return 0;
}