// rife_py.cpp
// RIFE v4.22 MNN inference: ImageProcess for uint8 HWC RGB -> tensor (mean 0, norm 1/255).
// process(img0_hwc_uint8, img1_hwc_uint8, timestep) -> out_hwc_uint8.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/ImageProcess.hpp>
#include <vector>
#include <memory>
#include <string>
#include <cmath>
#include <algorithm>

namespace py = pybind11;
using namespace MNN;
using namespace MNN::CV;

class RIFEProcessor {
public:
    explicit RIFEProcessor(const std::string& model_path) {
        BackendConfig backendConfig;
        backendConfig.precision = BackendConfig::Precision_Low;
        backendConfig.power = BackendConfig::Power_High;

        ScheduleConfig config;
        config.backendConfig = &backendConfig;
        config.type = MNN_FORWARD_OPENCL;
        config.mode = MNN_GPU_TUNING_WIDE | MNN_GPU_MEMORY_BUFFER;

        net_.reset(Interpreter::createFromFile(model_path.c_str()), Interpreter::destroy);
        net_->setSessionMode(Interpreter::Session_Release);
        net_->setCacheFile("rife.cache");
        session_ = net_->createSession(config);

        // ImageProcess: uint8 RGB HWC -> float NCHW with mean=0, norm=1/255
        img_config_.sourceFormat = RGB;
        img_config_.destFormat = RGB;
        img_config_.filterType = BILINEAR;
        img_config_.wrap = CLAMP_TO_EDGE;
        img_config_.mean[0] = 0.0f;
        img_config_.mean[1] = 0.0f;
        img_config_.mean[2] = 0.0f;
        img_config_.mean[3] = 0.0f;
        const float norm_val = 1.0f / 255.0f;
        img_config_.normal[0] = norm_val;
        img_config_.normal[1] = norm_val;
        img_config_.normal[2] = norm_val;
        img_config_.normal[3] = 1.0f;
    }

    // img0, img1: HWC uint8 RGB, same shape (H, W, 3). timestep in [0, 1].
    // Returns HWC uint8 RGB.
    py::array_t<uint8_t> process(
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img0,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img1,
        float timestep = 0.5f) {

        auto buf0 = img0.request();
        auto buf1 = img1.request();
        if (buf0.ndim != 3 || buf1.ndim != 3)
            throw std::runtime_error("img0, img1 must be 3D (HWC)");
        if (buf0.shape[2] != 3 || buf1.shape[2] != 3)
            throw std::runtime_error("img0, img1 must have 3 channels (RGB)");

        const int H = buf0.shape[0], W = buf0.shape[1];
        if (buf1.shape[0] != H || buf1.shape[1] != W)
            throw std::runtime_error("img0 and img1 shape mismatch");

        const int stride = W * 3;
        const uint8_t* p0 = static_cast<const uint8_t*>(buf0.ptr);
        const uint8_t* p1 = static_cast<const uint8_t*>(buf1.ptr);

        // Temporary tensors (1,3,H,W) for ImageProcess output
        std::vector<int> chw_dims = { 1, 3, H, W };
        std::shared_ptr<Tensor> t0(Tensor::create<float>(chw_dims, nullptr, Tensor::CAFFE), Tensor::destroy);
        std::shared_ptr<Tensor> t1(Tensor::create<float>(chw_dims, nullptr, Tensor::CAFFE), Tensor::destroy);

        {
            std::unique_ptr<ImageProcess, decltype(&ImageProcess::destroy)> pretreat(
                ImageProcess::create(img_config_, t0.get()), ImageProcess::destroy);
            if (!pretreat || pretreat->convert(p0, W, H, stride, t0.get()) != NO_ERROR)
                throw std::runtime_error("ImageProcess convert img0 failed");
        }
        {
            std::unique_ptr<ImageProcess, decltype(&ImageProcess::destroy)> pretreat(
                ImageProcess::create(img_config_, t1.get()), ImageProcess::destroy);
            if (!pretreat || pretreat->convert(p1, W, H, stride, t1.get()) != NO_ERROR)
                throw std::runtime_error("ImageProcess convert img1 failed");
        }

        const int input_channels = 9;
        const size_t hw = static_cast<size_t>(H) * W;
        std::vector<float> input_data(input_channels * hw);
        const float* f0 = t0->host<float>();
        const float* f1 = t1->host<float>();

        for (int c = 0; c < 3; ++c) {
            for (size_t i = 0; i < hw; ++i) {
                input_data[c * hw + i] = f0[c * hw + i];
                input_data[(3 + c) * hw + i] = f1[c * hw + i];
            }
        }
        for (size_t i = 0; i < hw; ++i)
            input_data[6 * hw + i] = timestep;

        if (cached_H_ != H || cached_W_ != W) {
            cached_H_ = H;
            cached_W_ = W;
            cached_ch7_.resize(hw);
            cached_ch8_.resize(hw);
            for (int h = 0; h < H; ++h) {
                float y = (H > 1) ? (-1.0f + 2.0f * h / (H - 1)) : 0.0f;
                for (int w = 0; w < W; ++w) {
                    float x = (W > 1) ? (-1.0f + 2.0f * w / (W - 1)) : 0.0f;
                    cached_ch7_[h * W + w] = x;
                    cached_ch8_[h * W + w] = y;
                }
            }
        }
        for (size_t i = 0; i < hw; ++i) {
            input_data[7 * hw + i] = cached_ch7_[i];
            input_data[8 * hw + i] = cached_ch8_[i];
        }

        std::vector<int> dims = { 1, input_channels, H, W };
        auto input = net_->getSessionInput(session_, "input");
        if (!input)
            input = net_->getSessionInput(session_, nullptr);
        net_->resizeTensor(input, dims);
        net_->resizeSession(session_);

        auto host_tensor = std::shared_ptr<Tensor>(Tensor::create<float>(dims, input_data.data(), Tensor::CAFFE), Tensor::destroy);
        input->copyFromHostTensor(host_tensor.get());

        net_->runSession(session_);

        auto output = net_->getSessionOutput(session_, "out");
        if (!output)
            output = net_->getSessionOutput(session_, nullptr);
        std::shared_ptr<Tensor> out_tensor(new Tensor(output, Tensor::CAFFE));
        output->copyToHostTensor(out_tensor.get());

        const float* out_ptr = out_tensor->host<float>();
        std::vector<size_t> out_shape = { static_cast<size_t>(H), static_cast<size_t>(W), 3 };
        py::array_t<uint8_t> result(out_shape);
        uint8_t* dst = result.mutable_data();
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                for (int c = 0; c < 3; ++c) {
                    float v = out_ptr[c * hw + h * W + w];
                    v = std::max(0.0f, std::min(1.0f, v));
                    dst[(h * W + w) * 3 + c] = static_cast<uint8_t>(v * 255.0f + 0.5f);
                }
            }
        }
        return result;
    }

private:
    std::shared_ptr<Interpreter> net_;
    Session* session_ = nullptr;
    ImageProcess::Config img_config_;
    int cached_H_ = -1;
    int cached_W_ = -1;
    std::vector<float> cached_ch7_;
    std::vector<float> cached_ch8_;
};

PYBIND11_MODULE(rife_mnn, m) {
    py::class_<RIFEProcessor>(m, "RIFEProcessor")
        .def(py::init<const std::string&>(), py::arg("model_path"))
        .def("process", &RIFEProcessor::process,
             py::arg("img0"), py::arg("img1"), py::arg("timestep") = 0.5f);
}
