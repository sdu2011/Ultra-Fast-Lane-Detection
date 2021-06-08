#include "lanedetector.hpp"
#include "utils.h"

#include "opencv2/core.hpp"
// #include "opencv2/imgproc.hpp"
// #include "opencv2/highgui.hpp"
using namespace cv;

#include <vector>
#include <iomanip>
#include <algorithm>
using namespace std;


LaneDetector::LaneDetector()
{
    net_ptr_.reset(new Tn::TrtNet("../data/ep153.onnx",Tn::RUN_MODE::FLOAT32));

    int outputCount = net_ptr_->getOutputSize() / sizeof(float);
    // cout<<"outputCount="<<outputCount<<endl;
    result_.reserve(outputCount);

    nvinfer1::Dims dims = net_ptr_->get_output_dims();
    int grid_num = dims.d[1];
    int anchor_row_num = dims.d[2];
    int max_lane_num = dims.d[3];
    empty_lanerow_.reserve(anchor_row_num * max_lane_num);
    for(int i = 0;i<empty_lanerow_.size();i++)
    {
        empty_lanerow_[i] = false;
    }

    grid_num_ = grid_num;
    anchor_row_num_ = anchor_row_num;
    max_lane_num_ = max_lane_num;
}

LaneDetector::~LaneDetector()
{

}

//hwc bgr --> chw rgb
std::vector<float> LaneDetector::prepareImage(cv::Mat & in_img)
{
  // using namespace cv;

  int c = 3;
  int h = 1080;
  int w = 1440;

  float scale = std::min(float(w) / in_img.cols, float(h) / in_img.rows);
  auto scaleSize = cv::Size(in_img.cols * scale, in_img.rows * scale);

  cv::Mat rgb;
  cv::cvtColor(in_img, rgb, CV_BGR2RGB);
  cv::Mat resized;
  cv::resize(rgb, resized, scaleSize, 0, 0, cv::INTER_CUBIC);

  cv::Mat cropped(h, w, CV_8UC3, 127);
  cv::Rect rect(
    (w - scaleSize.width) / 2, (h - scaleSize.height) / 2, scaleSize.width, scaleSize.height);
  resized.copyTo(cropped(rect));

  cv::Mat img_float;
  cropped.convertTo(img_float, CV_32FC3, 1 / 255.0);

  // HWC TO CHW
  std::vector<cv::Mat> input_channels(c);
  cv::split(img_float, input_channels);

  std::vector<float> result(h * w * c);
  auto data = result.data();
  int channelLength = h * w;
  for (int i = 0; i < c; ++i) {
    memcpy(data, input_channels[i].data, channelLength * sizeof(float));
    data += channelLength;
  }

  return result;
}

// void LaneDetector::calcSoftmax(float * data, std::vector<float> & probs, int num_output)
// {
//     float exp_sum = 0.0;
//     for (int i = 0; i < num_output; ++i) 
//     {
//         exp_sum += exp(data[i]);
//     }

//     for (int i = 0; i < num_output; ++i) 
//     {
//         probs.push_back(exp(data[i]) / exp_sum);
//     }
// }

void LaneDetector::calcSoftmax(vector<float> data, std::vector<float> & probs)
{
    float exp_sum = 0.0;
    for (int i = 0; i < data.size(); ++i) 
    {
        exp_sum += exp(data[i]);
    }

    for (int i = 0; i < data.size(); ++i) 
    {
        probs.push_back(exp(data[i]) / exp_sum);
    }
}

//torch.Size([1, 101, 9, 2]) [batch,grid+1,anchor_row,lane] out[,a,b,c]为例,其含义为第b行的第a个grid为第c条车道线点的概率.
void LaneDetector::post_process(float* output_data,vector<float> & positions)
{
    int outputCount = net_ptr_->getOutputSize() / sizeof(float);
    vector<float> result(outputCount);
    
    nvinfer1::Dims dims = net_ptr_->get_output_dims();
    int grid_num = dims.d[1];
    int anchor_row_num = dims.d[2];
    int max_lane_num = dims.d[3];

    //softmax
    float* cur_data = output_data;
    std::vector<float> probs; //将输出转换为概率  存储顺序[lane,row,grid]

    for(int lane = 0; lane < max_lane_num; lane++)
    {
        for(int row = 0;row < anchor_row_num;row++)
        {
            vector<float> all_grid_data;
            vector<float> grid_data; 
            vector<float> grid_probs;
            
            for(int grid= 0;grid<grid_num;grid++) 
            {         
                int index = grid * (max_lane_num * anchor_row_num) + row * max_lane_num + lane;  //data[w,h,c]
                // int index2 = grid * (max_lane_num * anchor_row_num) + ((anchor_row_num -1) - row) * max_lane_num + lane; //data[w,anchor_row_num-1-h,c]           
                float value = output_data[index];  
                if(grid != grid_num -1)
                {
                    grid_data.push_back(value);
                }

                all_grid_data.push_back(value);
            }

            //argmax
            size_t index = argmax(all_grid_data.begin(), all_grid_data.end());
            if(index == grid_num - 1)
            {
                int pos = lane * anchor_row_num + row;
                empty_lanerow_[pos] = true;
            }
            
            calcSoftmax(grid_data, grid_probs); //沿着grid这个维度只对前grid_num - 1个grid做softmax

            probs.insert(probs.end(), grid_probs.begin(), grid_probs.end());
        }
    }

    //anchor行的第几个grid为第n条车道线的概率
    
    for(int lane = 0; lane < max_lane_num; lane++)
    {
        for(int row = 0;row < anchor_row_num;row++)
        {
            float grid_exp = 0;//第lane条车道线在第row个参考行的grid的期望位置.
            for(int grid= 0;grid<grid_num - 1;grid++)
            {
                int index = lane * ( (grid_num-1) * anchor_row_num) + row * (grid_num-1) + grid;

                grid_exp += probs[index] * (1 + grid);
            }
            

            int pos = lane * anchor_row_num + row;
            if(empty_lanerow_[pos] == true)
            {
                grid_exp = 0;
            }

            // cout<<"grid_exp="<<grid_exp<<endl;
    
            positions.push_back(grid_exp);
        }            
    }
}

void LaneDetector::draw(cv::Mat & in_img,vector<float> positions)
{
    std::vector<double> linSpaceVector = linspace(0, in_img.cols - 1, grid_num_);
    double linSpace = linSpaceVector[1] - linSpaceVector[0];

    for(int lane = 0; lane < max_lane_num_; lane++)
    {
        for(int row = 0;row < anchor_row_num_;row++)
        {
            int index = lane * anchor_row_num_ + row;
            float grid = positions[index];
            // int h = autocore_row_anchor[autocore_row_anchor.size()-1-row];
            int h = autocore_row_anchor[row];
            int w = linSpace * grid;

            circle( in_img, Point( w, h ), 5, Scalar( 0, 255, 0 ), -1);
        }
    }

    imshow("lane",in_img);
    waitKey(0);
}

std::vector<double> LaneDetector::linspace(double start_in, double end_in, int num_in)
{
    std::vector<double> linspaced;

    double start = static_cast<double>(start_in);
    double end = static_cast<double>(end_in);
    double num = static_cast<double>(num_in);

    if (num == 0) { return linspaced; }
    if (num == 1) 
    {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for(int i=0; i < num-1; ++i)
    {
      linspaced.push_back(start + delta * i);
    }
    linspaced.push_back(end); // I want to ensure that start and end
                            // are exactly the same as the input

    return linspaced;
}

void LaneDetector::detect(cv::Mat & in_img)
{
    result_.clear();
    vector<float> input = prepareImage(in_img);

    net_ptr_->doInference(input.data(),result_.data());

    vector<float> positions;
    post_process(result_.data(),positions);

    draw(in_img,positions);
}