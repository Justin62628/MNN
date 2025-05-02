//
//  fusion_demo.cpp
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
        MNN_PRINT("Usage: %s fusion.mnn ... \n", argv[0]);
        return -1;
    }
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]), Interpreter::destroy);
    net->setSessionMode(Interpreter::Session_Backend_Auto);
    net->setSessionHint(Interpreter::MAX_TUNING_NUMBER, 5);
    net->setCacheFile("fusion.cache");
    ScheduleConfig config;
    BackendConfig backendConfig;
    backendConfig.precision = BackendConfig::Precision_Low;
    backendConfig.memory = BackendConfig::Memory_Low;
    config.backendConfig = &backendConfig;
    config.type  = MNN_FORWARD_OPENCL;
    config.numThread = 8;

    // BackendConfig bnconfig;
    // bnconfig.precision = BackendConfig::Precision_Low;
    // config.backendConfig = &bnconfig;
    auto session = net->createSession(config);

    // 2. Prepare Input Tensors
    std::vector<std::string> input_names = {"img0", "img1", "flow01", "flow10", "metric0", "metric1", "feat11", "feat12", "feat13", "feat21", "feat22", "feat23", "timestep"};

    int i=0;
    for (auto& name : input_names) {
        auto input = net->getSessionInput(session, name.c_str());
        cnpy::NpyArray input_npy = cnpy::npy_load(argv[2+i]);
        std::shared_ptr<Tensor> tensorInput(Tensor::create<float>(input->shape(), input_npy.data<float>(), Tensor::CAFFE));
        input->copyFromHostTensor(tensorInput.get());
        i++;
    }
    // 8. Run inference
    net->runSession(session);

    // 9. Get output tensor
    std::vector<std::string> output_names = {"out"};
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