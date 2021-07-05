#include "trtnet.h"
// #include <cublas_v2.h>
#include <cudnn.h>
#include <string.h>
#include <time.h>
#include <cassert>
#include <chrono>
#include <iostream>
#include <sstream>
#include <unordered_map>

using namespace nvinfer1;
using namespace nvonnxparser;
using namespace plugin;
using namespace std;

static Tn::Logger gLogger;

#define RETURN_AND_LOG(ret, severity, message)                            \
  do {                                                                    \
    std::string error_message = "lane_detection_error_log: " + std::string(message); \
    gLogger.log(ILogger::Severity::k##severity, error_message.c_str());   \
    return (ret);                                                         \
  } while (0)

inline void * safeCudaMalloc(size_t memSize)
{
  void * deviceMem;
  CUDA_CHECK(cudaMalloc(&deviceMem, memSize));
  if (deviceMem == nullptr) {
    std::cerr << "Out of memory" << std::endl;
    exit(1);
  }
  return deviceMem;
}

inline int64_t volume(const nvinfer1::Dims & d)
{
  return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
  switch (t) {
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kINT8:
      return 1;
  }
  throw std::runtime_error("Invalid DataType.");
  return 0;
}

namespace Tn
{

TrtNet::TrtNet(const std::string & onnxFile,RUN_MODE mode)
{
    IHostMemory * trtModelStream{nullptr};
    const int maxBatchSize = 1;

    std::string engine_path = onnxFile;
    engine_path.replace(engine_path.find("onnx"),sizeof("onnx")-1,"engine");
    cout<<"engine_path="<<engine_path<<endl;

    std::ifstream fs(engine_path);
    if(fs.is_open())
    {
        loadEngineFileAndCreateEngine(engine_path.c_str());

    }else
    {
        //load model to modelstream
        loadModelAndCreateEngine(onnxFile.c_str(), maxBatchSize,trtModelStream);

        //
        saveEngine(engine_path);
    }

    //
    initEngine();
}


void TrtNet::loadModelAndCreateEngine(const char * onnxFile, int maxBatchSize, IHostMemory *& trtModelStream)
{
    // Create the builder
    IBuilder * builder = createInferBuilder(gLogger);

    // Parse the model to populate the network, then set the outputs.
    //trt7的onnx parser要求必须显示指定batch
    const auto explicitBatch = (1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    INetworkDefinition * network = builder->createNetworkV2(explicitBatch);

    auto parser = nvonnxparser::createParser(*network, gLogger);

    std::cout << "Begin parsing model..." << std::endl;
    int verbosity = 1;
    parser->parseFromFile(onnxFile, verbosity);
    std::cout << "End parsing model..." << std::endl;

    // Build the engine.
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 30);  // 1G
    if (run_mode_ == RUN_MODE::INT8) 
    {
        std::cout << "setInt8Mode" << std::endl;
        if (!builder->platformHasFastInt8())
        std::cout << "Notice: the platform do not has fast for int8" << std::endl;
        builder->setInt8Mode(true);
    } 
    else if (run_mode_ == RUN_MODE::FLOAT16) 
    {
        std::cout << "setFp16Mode" << std::endl;
        if (!builder->platformHasFastFp16())
        std::cout << "Notice: the platform do not has fast for fp16" << std::endl;
        builder->setFp16Mode(true);
    }

    std::cout << "Begin building engine..." << std::endl;
    ICudaEngine * engine = builder->buildCudaEngine(*network);
    std::cout << "End building engine..." << std::endl;
    if (!engine)
    {
        std::cout << "ERROR:can not create engine......" << std::endl;        
        return;
    }
    

    // We don't need the network any more, and we can destroy the parser.
    network->destroy();
    parser->destroy();

    // Serialize the engine, then close everything down.
    trtModelStream = engine->serialize();
    assert(trtModelStream != nullptr);
    
    builder->destroy();
            
    //desrialize modelstream to ICudaEngine.  
    engine->destroy();
    mTrtRunTime = createInferRuntime(gLogger);
    assert(mTrtRunTime != nullptr);
    mTrtEngine = mTrtRunTime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size());
    assert(mTrtEngine != nullptr);
    trtModelStream->destroy();
}

void TrtNet::saveEngine(std::string fileName)
{
    if (mTrtEngine) 
    {
        nvinfer1::IHostMemory * data = mTrtEngine->serialize();
        std::ofstream file;
        file.open(fileName, std::ios::binary | std::ios::out);
        if (!file.is_open()) 
        {
            std::cout << "read create engine file" << fileName << " failed" << std::endl;
            return;
        }
        else
        {
            file.write((const char *)data->data(), data->size());
            file.close();

            cout<<"save enginge file to "<<fileName<<endl;
        }
    }
}

void TrtNet::loadEngineFileAndCreateEngine(const char* engineFile)
{
    //read engine file to data
    std::fstream file;
    file.open(engineFile, std::ios::binary | std::ios::in);
    if (!file.is_open()) 
    {
        cout << "read engine file" << engineFile << " failed" << endl;
        return;
    }

    file.seekg(0, ios::end);
    int length = file.tellg();
    file.seekg(0, ios::beg);
    std::unique_ptr<char[]> data(new char[length]);
    file.read(data.get(), length);

    file.close();

    //cretate ICudaEngine
    mTrtRunTime = createInferRuntime(gLogger);
    assert(mTrtRunTime != nullptr);
    mTrtEngine = mTrtRunTime->deserializeCudaEngine(data.get(), length);
    assert(mTrtEngine != nullptr);    
}

void TrtNet::initEngine()
{
    const int maxBatchSize = 1;
    mTrtContext = mTrtEngine->createExecutionContext();
    assert(mTrtContext != nullptr);

    //input and output number
    int nbBindings = mTrtEngine->getNbBindings();
    // cout<<"nbBindings="<<nbBindings<<endl;

    //malloc buffer for inputs and outputs
    mTrtCudaBuffer.resize(nbBindings);
    mTrtBindBufferSize.resize(nbBindings);
    for (int i = 0; i < nbBindings; ++i) 
    {
        Dims dims = mTrtEngine->getBindingDimensions(i);
        DataType dtype = mTrtEngine->getBindingDataType(i);
        int64_t totalSize = volume(dims) * maxBatchSize * getElementSize(dtype);
        mTrtBindBufferSize[i] = totalSize;
        // cout<<"mTrtBindBufferSize["<<i<<"]="<<totalSize<<endl;
        mTrtCudaBuffer[i] = safeCudaMalloc(totalSize);
        if (mTrtEngine->bindingIsInput(i)) 
        {
            mTrtInputCount++;
        }
    }

  CUDA_CHECK(cudaStreamCreate(&mTrtCudaStream));
}

void TrtNet::doInference(const void * inputData, void * outputData)
{
    // std::cout << " begin" << std::endl;

    static const int batchSize = 1;

    // cout<<"line:"<<__LINE__<<endl;
    // cout<<"mTrtBindBufferSize[0]"<<mTrtBindBufferSize[0]<<endl;
    //model has only one input
    CUDA_CHECK(
        cudaMemcpyAsync(mTrtCudaBuffer[0], inputData, mTrtBindBufferSize[0], cudaMemcpyHostToDevice,
        mTrtCudaStream)
        );
    // cout<<"line:"<<__LINE__<<endl;
    auto t_start = std::chrono::high_resolution_clock::now();

    //prepare input/output pointer
    void * bindings[2];
    bindings[0] = mTrtCudaBuffer[0]; //input
    bindings[1] = mTrtCudaBuffer[1]; //output

    // mTrtContext->executeV2(bindings);
    mTrtContext->enqueueV2(bindings, mTrtCudaStream, nullptr);
    auto t_end = std::chrono::high_resolution_clock::now();
    float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();

    std::cout << "Time taken for inference is " << total << " ms." << std::endl;

    // copy output from buffer to outputData
    for (size_t bindingIdx = mTrtInputCount; bindingIdx < mTrtBindBufferSize.size(); ++bindingIdx) 
    {
        auto size = mTrtBindBufferSize[bindingIdx];
        CUDA_CHECK(cudaMemcpyAsync(
        outputData, mTrtCudaBuffer[bindingIdx], size, cudaMemcpyDeviceToHost, mTrtCudaStream));
        outputData = (char *)outputData + size;
    }
}

size_t TrtNet::getInputSize()
{
    return std::accumulate(mTrtBindBufferSize.begin(), mTrtBindBufferSize.begin() + mTrtInputCount, 0);
};

size_t TrtNet::getOutputSize()
{
    size_t len = 0;
    len = std::accumulate(mTrtBindBufferSize.begin() + mTrtInputCount, mTrtBindBufferSize.end(), 0);

    // cout<<"len="<<len<<endl;
    return len;
};

nvinfer1::Dims TrtNet::get_output_dims()
{
    int output_binding_index = 1;
    nvinfer1::Dims dims = mTrtEngine->getBindingDimensions(output_binding_index);

    return dims;
}

}
