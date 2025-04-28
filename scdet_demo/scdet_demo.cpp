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

using namespace MNN;
using namespace MNN::CV;

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./scdet_demo.out model.mnn input0.jpg input1.jpg \n");
        return 0;
    }
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]), Interpreter::destroy);
    // net->setCacheFile("scdet.cache");  // auto, no need
    net->setSessionMode(Interpreter::Session_Backend_Fix);
    net->setSessionHint(Interpreter::MAX_TUNING_NUMBER, 5);
    ScheduleConfig config;
    BackendConfig backendConfig;
    backendConfig.precision = BackendConfig::Precision_Low;
    config.backendConfig = &backendConfig;
    config.type  = MNN_FORWARD_VULKAN;
    // BackendConfig bnconfig;
    // bnconfig.precision = BackendConfig::Precision_Low;
    // config.backendConfig = &bnconfig;
    auto session = net->createSession(config);

    auto input0 = net->getSessionInput(session, "input0");
    auto input1 = net->getSessionInput(session, "input1");
    auto shape = input0->shape();

    // net->resizeTensor(input, shape);
    // net->resizeSession(session);
    float memoryUsage = 0.0f;
    net->getSessionInfo(session, MNN::Interpreter::MEMORY, &memoryUsage);
    float flops = 0.0f;
    net->getSessionInfo(session, MNN::Interpreter::FLOPS, &flops);
    int backendType[2];
    net->getSessionInfo(session, MNN::Interpreter::BACKENDS, backendType);
    MNN_PRINT("Session Info: memory use %f MB, flops is %f M, backendType is %d\n", memoryUsage, flops, backendType[0]);

    auto output = net->getSessionOutput(session, NULL);
    if (nullptr == output || output->elementSize() == 0) {
        MNN_ERROR("Resize error, the model can't run batch: %d\n", shape[0]);
        return 0;
    }
    std::shared_ptr<Tensor> inputUser0(new Tensor(input0, Tensor::CAFFE));
    std::shared_ptr<Tensor> inputUser1(new Tensor(input1, Tensor::CAFFE));
    auto bpp          = inputUser0->channel();
    auto size_h       = inputUser0->height();
    auto size_w       = inputUser0->width();
    MNN_PRINT("input: w:%d , h:%d, bpp: %d\n", size_w, size_h, bpp);

    int width, height, channel;
    auto inputPatch0 = argv[2];
    auto inputImage0 = stbi_load(inputPatch0, &width, &height, &channel, 4);
    auto inputPatch1 = argv[3];
    auto inputImage1 = stbi_load(inputPatch1, &width, &height, &channel, 4);
    if (nullptr == inputImage0 || nullptr == inputImage1) {
        MNN_ERROR("Can't open %s %s\n", inputPatch0, inputPatch1);
        return 0;
    }
    MNN_PRINT("origin size: %d, %d\n", width, height);
    
    Matrix trans;
    trans.setScale((float)(width-1) / (size_w-1), (float)(height-1) / (size_h-1));
    ImageProcess::Config ip_config;
    ip_config.filterType = BILINEAR;
    float mean[3]     = {0.0f, 0.0f, 0.0f};
    float normals[3] = {1. / 255.f, 1. / 255.f, 1. / 255.f, };
    ::memcpy(ip_config.mean, mean, sizeof(mean));
    ::memcpy(ip_config.normal, normals, sizeof(normals));
    ip_config.sourceFormat = RGBA;
    ip_config.destFormat   = BGR;

    std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(ip_config), ImageProcess::destroy);
    pretreat->setMatrix(trans);
    pretreat->convert((uint8_t*)inputImage0, width, height, 0, inputUser0->host<uint8_t>(), size_w, size_h, bpp, 0, inputUser0->getType());
    pretreat->convert((uint8_t*)inputImage1, width, height, 0, inputUser1->host<uint8_t>(), size_w, size_h, bpp, 0, inputUser1->getType());

    // stbi_image_free(inputImage0);
    // stbi_image_free(inputImage1);

    input0->copyFromHostTensor(inputUser0.get());
    input1->copyFromHostTensor(inputUser1.get());

    net->runSession(session);

    auto dimType = output->getDimensionType();
    std::shared_ptr<Tensor> outputUser(new Tensor(output, dimType));
    output->copyToHostTensor(outputUser.get());
    auto type = outputUser->getType();

    auto size = outputUser->stride(1);
    auto values = outputUser->host<float>();
    MNN_PRINT("%f, %f\n", *values, *(values + size));
    net->updateCacheFile(session);
    return 0;
}
