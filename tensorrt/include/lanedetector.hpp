#ifndef LANE_DETECTOR_H_
#define LANE_DETECTOR_H_

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "trtnet.h"

class LaneDetector
{
public:
  LaneDetector();
  ~LaneDetector();

  std::vector<float> prepareImage(cv::Mat & in_img);

  void post_process(float* output_data,std::vector<float> & positions);

  void calcSoftmax(std::vector<float> data, std::vector<float> & probs);

  void draw(cv::Mat & in_img,std::vector<float> positions);
  
  std::vector<double> linspace(double start_in, double end_in, int num_in);

  void detect(cv::Mat & in_img);

  std::shared_ptr<Tn::TrtNet> net_ptr_;

  std::vector<int> autocore_row_anchor = {820,843,873,903,930,961,985,1008,1043};
  std::vector<float> result_; 

  std::vector<bool> empty_lanerow_;

  int grid_num_;
  int anchor_row_num_;
  int max_lane_num_;
};

#endif