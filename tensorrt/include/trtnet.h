/*
 * MIT License
 * 
 * Copyright (c) 2018 lewes6369
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
*/
#ifndef __TRT_NET_H_
#define __TRT_NET_H_

#include <algorithm>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"
#include "utils.h"

namespace Tn
{

enum class RUN_MODE { FLOAT32 = 0, FLOAT16 = 1, INT8 = 2 };

class TrtNet
{
public:
  TrtNet(const std::string & onnxFile,RUN_MODE mode);
  // Load from engine file
  explicit TrtNet(const std::string & engineFile);

  ~TrtNet()
  {
    // // Release the stream and the buffers
    // cudaStreamSynchronize(mTrtCudaStream);
    // cudaStreamDestroy(mTrtCudaStream);
    // for (auto & item : mTrtCudaBuffer) cudaFree(item);

    // // mTrtPluginFactory.destroyPlugin();

    // if (!mTrtRunTime) mTrtRunTime->destroy();
    // if (!mTrtContext) mTrtContext->destroy();
    // if (!mTrtEngine) mTrtEngine->destroy();
  };


private:
    RUN_MODE run_mode_ = RUN_MODE::FLOAT32;

    //build engine from onnxfile
    void loadModelAndCreateEngine(const char * onnxFile, int maxBatchSize, nvinfer1::IHostMemory *& trtModelStream);

    //save engine file to disk
    void saveEngine(std::string fileName);

    //
    void loadEngineFileAndCreateEngine(const char* engineFile);

    //init engine,malloc memory.
    void initEngine();

    //
    nvinfer1::IExecutionContext * mTrtContext;
    nvinfer1::ICudaEngine * mTrtEngine;
    nvinfer1::IRuntime * mTrtRunTime;
    cudaStream_t mTrtCudaStream;

    std::vector<void *> mTrtCudaBuffer;
    std::vector<int64_t> mTrtBindBufferSize;

    int mTrtInputCount = 0;

public:
    //
    void doInference(const void * inputData, void * outputData);

    //
    size_t getInputSize();

    size_t getOutputSize();

    nvinfer1::Dims get_output_dims(); 
};

}  // namespace Tn

#endif  //__TRT_NET_H_
