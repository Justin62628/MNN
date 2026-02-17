// tariff_py.cpp
// Supports NB202 (coords-based) and PWR (img0, img1 only; scale=1.0 fixed) Tariff models
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <vector>
#include <memory>
#include <string>

namespace py = pybind11;
using namespace MNN;
using namespace MNN::CV;

enum class ModelType { NB202, PWR };

class TariffProcessor {
public:
    TariffProcessor(const std::string& feat_path, const std::string& fusion_path,
                    const std::string& model_type = "nb202") {
        model_type_ = (model_type == "pwr") ? ModelType::PWR : ModelType::NB202;

        BackendConfig backendConfig;
        // backendConfig.precision = BackendConfig::Precision_High;
        // backendConfig.power = BackendConfig::Power_High;
        // backendConfig.memory = BackendConfig::Memory_Low;
        
        ScheduleConfig feat_config;
        feat_config.backendConfig = &backendConfig;
        // feat_config.type  = MNN_FORWARD_CPU;
        feat_config.type  = MNN_FORWARD_OPENCL;
        // feat_config.type  = MNN_FORWARD_VULKAN;
        // feat_config.numThread = 1;
        // feat_config.mode = MNN_GPU_TUNING_NORMAL | MNN_GPU_MEMORY_BUFFER;
        feat_config.mode = MNN_GPU_TUNING_NORMAL | MNN_GPU_MEMORY_BUFFER;

        auto runtimeInfo = Interpreter::createRuntime({feat_config});

        feat_net.reset(Interpreter::createFromFile(feat_path.c_str()), Interpreter::destroy);
        feat_net->setSessionMode(Interpreter::Session_Backend_Fix);
        // feat_net->setSessionHint(Interpreter::MAX_TUNING_NUMBER, 5);
        feat_net->setCacheFile("feat.cache");
        feat_session = feat_net->createSession(feat_config, runtimeInfo);

        ScheduleConfig fusion_config;
        fusion_config.backendConfig = &backendConfig;
        fusion_config.type  = MNN_FORWARD_CPU;
        // feat_config.type  = MNN_FORWARD_OPENCL;
        // feat_config.type  = MNN_FORWARD_VULKAN;
        // fusion_config.numThread = 16;
        // feat_config.mode = MNN_GPU_TUNING_NORMAL | MNN_GPU_MEMORY_BUFFER;
        // fusion_config.mode = MNN_GPU_TUNING_NORMAL | MNN_GPU_MEMORY_BUFFER;

        fusion_net.reset(Interpreter::createFromFile(fusion_path.c_str()), Interpreter::destroy);
        fusion_net->setSessionMode(Interpreter::Session_Backend_Fix);
        // fusion_net->setSessionHint(Interpreter::MAX_TUNING_NUMBER, 5);
        fusion_net->setCacheFile("fusion.cache");
        // fusion_session = fusion_net->createSession(fusion_config, runtimeInfo);
        fusion_session = fusion_net->createSession(fusion_config);

        if (model_type_ == ModelType::NB202) {
            feature_outputs_ = {"flow01", "flow10", "metric0", "metric1",
                               "feat11", "feat12", "feat13", "feat21", "feat22", "feat23"};
        } else {
            feature_outputs_ = {"img0_out", "img1_out", "flow01", "flow10", "metric0", "metric1",
                                "f0", "f1"};
        }
    }

    py::list process(
        py::array_t<float, py::array::c_style | py::array::forcecast> img0,
        py::array_t<float, py::array::c_style | py::array::forcecast> img1,
        py::array_t<float, py::array::c_style | py::array::forcecast> timesteps,
        py::object coords = py::none()) {

        auto img0_buf = img0.request();
        auto img1_buf = img1.request();
        auto ts_buf = timesteps.request();

        if (img0_buf.ndim != 4 || img1_buf.ndim != 4)
            throw std::runtime_error("img0, img1 must be 4D arrays (BCHW)");

        const int B = img0_buf.shape[0], C = img0_buf.shape[1], H = img0_buf.shape[2], W = img0_buf.shape[3];

        if (model_type_ == ModelType::NB202) {
            if (coords.is_none())
                throw std::runtime_error("NB202 model requires coords input");
            auto coords_buf = py::array_t<float, py::array::c_style>(coords).request();
            if (coords_buf.ndim != 4)
                throw std::runtime_error("coords must be 4D (BCHW)");
            run_feature_nb202(img0_buf.ptr, img1_buf.ptr, coords_buf.ptr, B, C, H, W);
            cached_features["img0"] = std::shared_ptr<Tensor>(Tensor::create<float>({B, C, H, W}, img0_buf.ptr, Tensor::CAFFE), Tensor::destroy);
            cached_features["img1"] = std::shared_ptr<Tensor>(Tensor::create<float>({B, C, H, W}, img1_buf.ptr, Tensor::CAFFE), Tensor::destroy);
        } else {
            run_feature_pwr(img0_buf.ptr, img1_buf.ptr, B, C, H, W);
        }

        const int T = ts_buf.shape[1];
        std::vector<py::array_t<float>> outputs;

        for (int t = 0; t < T; ++t) {
            float* ts_ptr = static_cast<float*>(ts_buf.ptr) + t * ts_buf.shape[2] * ts_buf.shape[3];
            auto out_tensor = run_fusion_model(ts_ptr, ts_buf.shape[0], ts_buf.shape[2], ts_buf.shape[3]);
            outputs.emplace_back(create_numpy_array(out_tensor.get()));
        }

        return concatenate_outputs(outputs);
    }

private:
    ModelType model_type_;
    std::shared_ptr<Interpreter> feat_net, fusion_net;
    Session *feat_session, *fusion_session;
    std::vector<std::string> feature_outputs_;
    std::unordered_map<std::string, std::shared_ptr<Tensor>> cached_features;

    void run_feature_nb202(void* img0_ptr, void* img1_ptr, void* coords_ptr,
                          int B, int C, int H, int W) {
        auto feat_img0 = feat_net->getSessionInput(feat_session, "img0");
        auto feat_img1 = feat_net->getSessionInput(feat_session, "img1");
        auto feat_coords = feat_net->getSessionInput(feat_session, "coords");

        copy_data_to_tensor(feat_img0, img0_ptr, B, C, H, W);
        copy_data_to_tensor(feat_img1, img1_ptr, B, C, H, W);
        copy_data_to_tensor(feat_coords, coords_ptr, B, 2, H / 2, W / 2);

        feat_net->runSession(feat_session);

        for (const auto& name : feature_outputs_) {
            auto tensor = feat_net->getSessionOutput(feat_session, name.c_str());
            auto nchwTensor = std::make_shared<Tensor>(tensor, Tensor::CAFFE);
            tensor->copyToHostTensor(nchwTensor.get());
            cached_features[name] = nchwTensor;
        }
    }

    void run_feature_pwr(void* img0_ptr, void* img1_ptr, int B, int C, int H, int W) {
        auto feat_img0 = feat_net->getSessionInput(feat_session, "img0");
        auto feat_img1 = feat_net->getSessionInput(feat_session, "img1");

        copy_data_to_tensor(feat_img0, img0_ptr, B, C, H, W);
        copy_data_to_tensor(feat_img1, img1_ptr, B, C, H, W);

        feat_net->runSession(feat_session);

        for (const auto& name : feature_outputs_) {
            auto tensor = feat_net->getSessionOutput(feat_session, name.c_str());
            auto nchwTensor = std::make_shared<Tensor>(tensor, Tensor::CAFFE);
            tensor->copyToHostTensor(nchwTensor.get());
            cached_features[name] = nchwTensor;
        }
    }

    std::shared_ptr<Tensor> run_fusion_model(float* timestep, int B, int H, int W) {
        std::vector<std::string> input_names;
        if (model_type_ == ModelType::NB202) {
            input_names = {"img0", "img1", "flow01", "flow10", "metric0", "metric1",
                          "feat11", "feat12", "feat13", "feat21", "feat22", "feat23",
                          "timestep"};
        } else {
            input_names = {"img0_out", "img1_out", "flow01", "flow10", "metric0", "metric1",
                          "f0", "f1", "timestep"};
        }

        for (const auto& name : input_names) {
            auto input = fusion_net->getSessionInput(fusion_session, name.c_str());

            if (name == "timestep") {
                std::shared_ptr<Tensor> ts_tensor(
                    Tensor::create<float>({B, 1, H, W}, timestep, Tensor::CAFFE));
                input->copyFromHostTensor(ts_tensor.get());
            } else {
                input->copyFromHostTensor(cached_features[name].get());
            }
        }

        fusion_net->runSession(fusion_session);

        auto output = fusion_net->getSessionOutput(fusion_session, "out");
        std::shared_ptr<Tensor> output_tensor(new Tensor(output, output->getDimensionType()));
        output->copyToHostTensor(output_tensor.get());
        return output_tensor;
    }

    py::array_t<float> create_numpy_array(Tensor* tensor) {
        std::vector<size_t> shape;
        for (int dim : tensor->shape()) shape.push_back(dim);
        return py::array_t<float>(shape, tensor->host<float>());
    }

    py::list concatenate_outputs(const std::vector<py::array_t<float>>& outputs) {
        py::list result_list;
        for (const auto& arr : outputs) {
            result_list.append(arr);
        }
        return result_list;
    }

    void copy_data_to_tensor(Tensor* dest, void* src, int B, int C, int H, int W) {
        auto nchwTensor = Tensor::create<float>({B, C, H, W}, src, Tensor::CAFFE);
        dest->copyFromHostTensor(nchwTensor);
        delete nchwTensor;
    }
};

PYBIND11_MODULE(tariff_mnn, m) {
    py::class_<TariffProcessor>(m, "TariffProcessor")
        .def(py::init<const std::string&, const std::string&, const std::string&>(),
             py::arg("feat_path"), py::arg("fusion_path"), py::arg("model_type") = "nb202")
        .def("process", &TariffProcessor::process,
             py::arg("img0"), py::arg("img1"), py::arg("timesteps"),
             py::arg("coords") = py::none());
}
