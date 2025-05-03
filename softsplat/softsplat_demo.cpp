//
//  pictureRecognition.cpp
//  MNN
//
//  Created by MNN on 2018/05/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#define MNN_OPEN_TIME_TRACE
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>
#include <MNN/AutoTime.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "cnpy/cnpy.h"

using namespace MNN;
using namespace MNN::CV;

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        MNN_PRINT("Usage: %s model.mnn x.npy y.npy\n", argv[0]);
        return -1;
    }

    // 1. Load MNN Model
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]), Interpreter::destroy);
    net->setSessionMode(Interpreter::Session_Backend_Fix);
    net->setSessionHint(Interpreter::MAX_TUNING_NUMBER, 5);
    ScheduleConfig config;
    BackendConfig backendConfig;
    backendConfig.precision = BackendConfig::Precision_Low;
    config.backendConfig = &backendConfig;
    config.type = MNN_FORWARD_OPENCL;
    // config.type = MNN_FORWARD_CPU;
    // config.numThread = 1;
    // config.mode = MNN_GPU_TUNING_NORMAL | MNN_GPU_MEMORY_BUFFER; // MNN_GPU_MEMORY_BUFFER, MNN_GPU_MEMORY_IMAGE?
    auto session = net->createSession(config);

    // 2. Prepare Input Tensors
    auto inputX = net->getSessionInput(session, "x");
    auto inputY = net->getSessionInput(session, "y");
    
    // 3. Load NPY Data
    cnpy::NpyArray x_npy = cnpy::npy_load(argv[2]);
    cnpy::NpyArray y_npy = cnpy::npy_load(argv[3]);
    
    // 4. Validate Input Shapes
    if (x_npy.shape.size() != 4 || y_npy.shape.size() != 4) {
        MNN_ERROR("Invalid input dimensions\n");
        return -1;
    }

    // 5. Create MNN Tensors from NPY data
    std::shared_ptr<Tensor> tensorX(Tensor::create<float>(inputX->shape(), x_npy.data<float>(), Tensor::CAFFE));
    std::shared_ptr<Tensor> tensorY(Tensor::create<float>(inputY->shape(), y_npy.data<float>(), Tensor::CAFFE));
    
    // 7. Set input tensors
    inputX->copyFromHostTensor(tensorX.get());
    inputY->copyFromHostTensor(tensorY.get());

    // 8. Run inference
    net->runSession(session);

    // 9. Get output tensor
    auto output = net->getSessionOutput(session, nullptr);
    std::shared_ptr<Tensor> outputUser(new Tensor(output, output->getDimensionType()));
    output->copyToHostTensor(outputUser.get());

    // 10. Save output to NPY
    std::vector<size_t> output_shape;
    std::vector<size_t> inX_shape;
    std::vector<size_t> inY_shape;
    for (int i=0; i<outputUser->dimensions(); ++i) {
        output_shape.push_back(outputUser->length(i));
    }
    for (int i=0; i<inputX->dimensions(); ++i) {
        inX_shape.push_back(inputX->length(i));
    }
    for (int i=0; i<inputY->dimensions(); ++i) {
        inY_shape.push_back(inputY->length(i)); 
    }
    cnpy::npy_save("out.npy", outputUser->host<float>(), output_shape, "w");
    // cnpy::npy_save("in_x.npy", inputX->host<float>(), inX_shape, "w");
    // cnpy::npy_save("in_y.npy", inputY->host<float>(), inY_shape, "w");

    MNN_PRINT("Inference completed. Result saved to out.npy\n");
    return 0;
}