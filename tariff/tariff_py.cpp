// tariff_py.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#include <vector>
#include <memory>

namespace py = pybind11;
using namespace MNN;
using namespace MNN::CV;

class TariffProcessor {
public:
    TariffProcessor(const std::string& feat_path, const std::string& fusion_path) {
        // 初始化Feature模型
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

        feat_net.reset(Interpreter::createFromFile(feat_path.c_str()), Interpreter::destroy);  // shared ptr
        feat_net->setSessionMode(Interpreter::Session_Backend_Fix);
        // feat_net->setSessionHint(Interpreter::MAX_TUNING_NUMBER, 5);
        feat_net->setCacheFile("feat.cache");
        feat_session = feat_net->createSession(feat_config, runtimeInfo);

        // 初始化Fusion模型
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

        // 预分配特征模型输出存储
        feature_outputs = {"flow01", "flow10", "metric0", "metric1", 
                           "feat11", "feat12", "feat13", "feat21", "feat22", "feat23"};
    }

    py::array_t<float> process(
        py::array_t<float, py::array::c_style | py::array::forcecast> img0,
        py::array_t<float, py::array::c_style | py::array::forcecast> img1,
        py::array_t<float, py::array::c_style | py::array::forcecast> coords,
        py::array_t<float, py::array::c_style | py::array::forcecast> timesteps) {
        
        // getchar();
        // 获取输入缓冲区和形状信息
        auto img0_buf = img0.request();
        auto img1_buf = img1.request();
        auto coords_buf = coords.request();
        auto ts_buf = timesteps.request();

        // 验证输入形状
        if (img0_buf.ndim != 4 || img1_buf.ndim != 4 || coords_buf.ndim != 4)
            throw std::runtime_error("Inputs must be 4D arrays (BCHW)");
        
        const int B = img0_buf.shape[0], C = img0_buf.shape[1], H = img0_buf.shape[2], W = img0_buf.shape[3];

        // 运行Feature模型推理
        run_feature_model(img0_buf.ptr, img1_buf.ptr, coords_buf.ptr, 
                          B, C, H, W);

        cached_features["img0"] = std::shared_ptr<Tensor>(Tensor::create<float>({B, C, H, W}, img0_buf.ptr, Tensor::CAFFE), Tensor::destroy);
        cached_features["img1"] = std::shared_ptr<Tensor>(Tensor::create<float>({B, C, H, W}, img1_buf.ptr, Tensor::CAFFE), Tensor::destroy);

        // 处理时间步序列
        const int T = ts_buf.shape[1];
        std::vector<py::array_t<float>> outputs;
        
        for (int t = 0; t < T; ++t) {
            // 获取当前时间步数据
            float* ts_ptr = static_cast<float*>(ts_buf.ptr) + t * ts_buf.shape[2] * ts_buf.shape[3];
            
            // 运行Fusion模型
            auto out_tensor = run_fusion_model(ts_ptr, ts_buf.shape[0], 
                                               ts_buf.shape[2], ts_buf.shape[3]);
            
            // 转换为numpy数组
            outputs.emplace_back(create_numpy_array(out_tensor.get()));
        }

        // 拼接所有输出通道
        return concatenate_outputs(outputs);
    }

private:
    std::shared_ptr<Interpreter> feat_net, fusion_net;
    Session *feat_session, *fusion_session;
    std::vector<std::string> feature_outputs;
    std::unordered_map<std::string, std::shared_ptr<Tensor>> cached_features;

    void run_feature_model(void* img0_ptr, void* img1_ptr, void* coords_ptr,
                          int B, int C, int H, int W) {
        // 获取输入Tensor并拷贝数据
        auto feat_img0 = feat_net->getSessionInput(feat_session, "img0");
        auto feat_img1 = feat_net->getSessionInput(feat_session, "img1");
        auto feat_coords = feat_net->getSessionInput(feat_session, "coords");

        copy_data_to_tensor(feat_img0, img0_ptr, B, C, H, W);
        copy_data_to_tensor(feat_img1, img1_ptr, B, C, H, W);
        copy_data_to_tensor(feat_coords, coords_ptr, B, 2, H / 2, W / 2);

        // 运行推理
        feat_net->runSession(feat_session);

        // 缓存特征输出
        for (const auto& name : feature_outputs) {
            auto tensor = feat_net->getSessionOutput(feat_session, name.c_str());

            auto nchwTensor = std::make_shared<Tensor>(tensor, Tensor::CAFFE);
            tensor->copyToHostTensor(nchwTensor.get());
            
            cached_features[name] = nchwTensor;
        }
        
    }

    std::shared_ptr<Tensor> run_fusion_model(float* timestep, int B, int H, int W) {
        // 设置Fusion模型输入
        std::vector<std::string> input_names = {"img0", "img1", "flow01", "flow10",
                                               "metric0", "metric1", "feat11", "feat12",
                                               "feat13", "feat21", "feat22", "feat23",
                                               "timestep"};
        
        for (const auto& name : input_names) {
            auto input = fusion_net->getSessionInput(fusion_session, name.c_str());
            
            if (name == "timestep") {
                // 处理时间步输入
                std::shared_ptr<Tensor> ts_tensor(
                    Tensor::create<float>({B, 1, H, W}, timestep, Tensor::CAFFE));
                input->copyFromHostTensor(ts_tensor.get());
            } else {
                // 复制缓存特征
                input->copyFromHostTensor(cached_features[name].get());
            }
        }

        // 运行推理
        fusion_net->runSession(fusion_session);

        // 获取输出
        auto output = fusion_net->getSessionOutput(fusion_session, "out");
        std::shared_ptr<Tensor> output_tensor(new Tensor(output, output->getDimensionType()));
        output->copyToHostTensor(output_tensor.get());
        return output_tensor;
    }

    py::array_t<float> create_numpy_array(Tensor* tensor) {
        std::vector<size_t> shape;
        for (int dim : tensor->shape()) shape.push_back(dim);
        return py::array_t<float>(
            shape, 
            tensor->host<float>()
        );
    }

    py::list concatenate_outputs(const std::vector<py::array_t<float>>& outputs) {
        py::list result_list;
        
        // 验证所有输出形状一致性
        if (!outputs.empty()) {
            auto first_shape = outputs[0].request().shape;
            const int B = first_shape[0];
            const int C = first_shape[1];  // 3
            const int H = first_shape[2];
            const int W = first_shape[3];

            for (const auto& arr : outputs) {
                auto buf = arr.request();
                if (buf.ndim != 4 || 
                    buf.shape[0] != B || 
                    buf.shape[1] != C || 
                    buf.shape[2] != H || 
                    buf.shape[3] != W) 
                {
                    throw std::runtime_error("Inconsistent output shapes");
                }
                
                // 将每个输出转换为独立的numpy数组
                result_list.append(arr);
            }
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
        .def(py::init<const std::string&, const std::string&>())
        .def("process", &TariffProcessor::process);
}