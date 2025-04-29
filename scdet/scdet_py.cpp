#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>

namespace py = pybind11;
using namespace MNN;
using namespace MNN::CV;

class SceneDetector {
public:
    SceneDetector(const std::string &model_path)
        : cached_w(0), cached_h(0) {
        // 加载模型
        net.reset(Interpreter::createFromFile(model_path.c_str()), Interpreter::destroy);
        net->setSessionMode(Interpreter::Session_Backend_Fix);
        net->setSessionHint(Interpreter::MAX_TUNING_NUMBER, 5);
        ScheduleConfig config;
        BackendConfig backendConfig;
        backendConfig.precision = BackendConfig::Precision_Low;
        config.backendConfig = &backendConfig;
        config.type = MNN_FORWARD_OPENCL;
        config.numThread = 4;
        session = net->createSession(config);

        // 获取模型输入（两路差分）
        input0 = net->getSessionInput(session, "input0");
        input1 = net->getSessionInput(session, "input1");
        auto shape = input0->shape();
        size_w = shape[3]; size_h = shape[2];

        // 配置统一的图像预处理（Resize + BGR + 归一化）
        trans.reset(new Matrix());
        ImageProcess::Config ipconfig;
        ipconfig.sourceFormat = RGB; ipconfig.destFormat = BGR;
        ipconfig.filterType   = BILINEAR;
        std::memset(ipconfig.mean,   0, sizeof(ipconfig.mean));
        for (int i = 0; i < 3; ++i) ipconfig.normal[i] = 1.0f / 255.0f;
        pretreat.reset(ImageProcess::create(ipconfig), ImageProcess::destroy);

        // 预分配三帧预处理后的主机端 Tensor
        temp_host0.reset(new Tensor(input0, Tensor::CAFFE));
        temp_host1.reset(new Tensor(input1, Tensor::CAFFE));
        temp_host2.reset(new Tensor(input1, Tensor::CAFFE));
    }

    std::pair<float, float> detect(
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img0,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img1,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img2)
    {
        // —— 获取 NumPy 缓冲区 —— 
        auto b0 = img0.request(), b1 = img1.request(), b2 = img2.request();
        int H = b0.shape[0], W = b0.shape[1];
        auto ptr0 = static_cast<uint8_t*>(b0.ptr);
        auto ptr1 = static_cast<uint8_t*>(b1.ptr);
        auto ptr2 = static_cast<uint8_t*>(b2.ptr);

        // —— 尺寸变更时，更新缩放矩阵 —— 
        if (cached_w != W || cached_h != H) {
            trans->reset();
            trans->setScale(
                float(W - 1) / float(size_w - 1),
                float(H - 1) / float(size_h - 1)
            );
            pretreat->setMatrix(*trans);
            cached_w = W; cached_h = H;
        }

        // —— 分别预处理三帧 —— 
        pretreat->convert(ptr0, W, H, 0,
                          temp_host0->host<uint8_t>(), size_w, size_h, 3, 0, temp_host0->getType());
        pretreat->convert(ptr1, W, H, 0,
                          temp_host1->host<uint8_t>(), size_w, size_h, 3, 0, temp_host1->getType());
        pretreat->convert(ptr2, W, H, 0,
                          temp_host2->host<uint8_t>(), size_w, size_h, 3, 0, temp_host2->getType());

        // —— 计算差分：img1−img0 和 img2−img1 —— 
        auto d0 = temp_host0->host<float>();
        auto d1 = temp_host1->host<float>();
        auto d2 = temp_host2->host<float>();
        int count = size_w * size_h * 3;
        for (int i = 0; i < count; ++i) {
            float v0 = d1[i] - d0[i];
            float v1 = d2[i] - d1[i];
            d0[i] = v0;   // 差分结果写回 temp_host0
            d1[i] = v1;   // 差分结果写回 temp_host1
        }  // TODO: https://mnn-docs.readthedocs.io/en/latest/inference/expr.html

        // —— 将差分图拷贝到模型输入，执行推理 —— 
        input0->copyFromHostTensor(temp_host0.get());
        input1->copyFromHostTensor(temp_host1.get());
        {
            py::gil_scoped_release release;
            net->runSession(session);
        }

        // —— 读取并返回输出 —— 
        auto output = net->getSessionOutput(session, nullptr);
        Tensor result(output, output->getDimensionType());
        output->copyToHostTensor(&result);
        float *out_data = result.host<float>();
        int stride = result.stride(1);
        return { out_data[0], out_data[stride] };
    }

private:
    std::shared_ptr<Interpreter> net;
    Session *session;
    Tensor *input0, *input1;
    int size_w, size_h, cached_w, cached_h;
    std::unique_ptr<Matrix> trans;
    std::shared_ptr<ImageProcess> pretreat;
    std::unique_ptr<Tensor> temp_host0, temp_host1, temp_host2;
};

PYBIND11_MODULE(scdet_mnn, m) {
    py::class_<SceneDetector>(m, "SceneDetector")
        .def(py::init<const std::string &>())
        .def("detect", &SceneDetector::detect);
}
