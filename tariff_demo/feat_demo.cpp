//
//  feat_demo.cpp
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
#include <string>
#include <MNN/AutoTime.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "cnpy/cnpy.h"

using namespace MNN;
using namespace MNN::CV;

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        MNN_PRINT("Usage: %s feature.mnn feat_img0.npy feat_img1.npy feat_coords.npy\n", argv[0]);
        return -1;
    }
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]), Interpreter::destroy);
    net->setSessionMode(Interpreter::Session_Backend_Fix);
    net->setSessionHint(Interpreter::MAX_TUNING_NUMBER, 5);
    net->setCacheFile("feat.cache");
    ScheduleConfig config;
    BackendConfig backendConfig;
    backendConfig.precision = BackendConfig::Precision_Low;
    backendConfig.memory = BackendConfig::Memory_Low;
    config.backendConfig = &backendConfig;
    config.type  = MNN_FORWARD_CPU;
    config.numThread = 8;

    // BackendConfig bnconfig;
    // bnconfig.precision = BackendConfig::Precision_Low;
    // config.backendConfig = &bnconfig;
    auto session = net->createSession(config);

    // 2. Prepare Input Tensors
    auto img0 = net->getSessionInput(session, "img0");
    auto img1 = net->getSessionInput(session, "img1");
    auto coords = net->getSessionInput(session, "coords");
    
    // 3. Load NPY Data
    cnpy::NpyArray img0_npy = cnpy::npy_load(argv[2]);
    cnpy::NpyArray img1_npy = cnpy::npy_load(argv[3]);
    cnpy::NpyArray coords_npy = cnpy::npy_load(argv[4]);
    
    // 4. Validate Input Shapes
    if (img0_npy.shape.size() != 4 || img1_npy.shape.size() != 4) {
        MNN_ERROR("Invalid input dimensions\n");
        return -1;
    }

    // 5. Create MNN Tensors from NPY data
    std::shared_ptr<Tensor> tensorImg0(Tensor::create<float>(img0->shape(), img0_npy.data<float>(), Tensor::CAFFE));
    std::shared_ptr<Tensor> tensorImg1(Tensor::create<float>(img1->shape(), img1_npy.data<float>(), Tensor::CAFFE));
    std::shared_ptr<Tensor> tensorCoords(Tensor::create<float>(coords->shape(), coords_npy.data<float>(), Tensor::CAFFE));
    
    // 7. Set input tensors
    img0->copyFromHostTensor(tensorImg0.get());
    img1->copyFromHostTensor(tensorImg1.get());
    coords->copyFromHostTensor(tensorCoords.get());

    // 8. Run inference
    net->runSession(session);

    // 9. Get output tensor
    // ["flow01", "flow10", "metric0", "metric1", "feat11", "feat12", "feat13", "feat21", "feat22", "feat23"]
    std::vector<std::string> output_names = {"flow01", "flow10", "metric0", "metric1", "feat11", "feat12", "feat13", "feat21", "feat22", "feat23"};
    for (auto& name : output_names) {
        auto output = net->getSessionOutput(session, name.c_str());
        std::shared_ptr<Tensor> outputUser(new Tensor(output, output->getDimensionType()));
        output->copyToHostTensor(outputUser.get());
        std::string npy_name = static_cast<std::string>(name) + ".npy";
        std::vector<size_t> output_shape;
        for (int dim : outputUser->shape()) {
            output_shape.push_back(static_cast<size_t>(dim));
        }
        cnpy::npy_save(npy_name, outputUser->host<float>(), output_shape, "w");
    }

    MNN_PRINT("Inference completed. Result saved to out.npy\n");
    net->updateCacheFile(session);
    return 0;
}