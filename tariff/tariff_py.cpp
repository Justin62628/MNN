// tariff_py.cpp
// NB202 and PWR: same inputs (img0, img1, timesteps). Coords generated internally in Feature ONNX.
//
// GPU path: feature outputs stay on OpenCL; fusion inputs are filled via copyBetweenDevice (no GPU->CPU->GPU).
// Session modes: Session_Output_User on feature net so outputs are separable; sync via tensor->wait() before use.
// (With Express/Module API, use Executor::RuntimeManager::createRuntimeManager + setMode(Session_Input_User) etc.)
//
// --- OpenCL GPU selection (MNNDeviceContext) ---
//   platform_size: Max number of platforms to enumerate (clGetPlatformIDs first arg). Use 1 to get at least one.
//   platform_id:  Index of the platform (vendor) to use. 0 = first (e.g. Intel), 1 = second (e.g. NVIDIA), etc.
//   device_id:    Index of the GPU device within that platform. 0 = first GPU, 1 = second GPU on same platform.
// So: one machine can have multiple platforms (e.g. Intel + NVIDIA); each platform can have multiple devices (GPUs).
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#define MNN_USER_SET_DEVICE
#include <MNN/MNNSharedContext.h>
#include <vector>
#include <memory>
#include <string>
#include <cstdio>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace py = pybind11;
using namespace MNN;
using namespace MNN::CV;

// Minimal OpenCL C API types/constants for runtime-loaded clinfo (no link-time OpenCL dep).
#ifndef CL_API_CALL
#ifdef _WIN32
#define CL_API_CALL __stdcall
#else
#define CL_API_CALL
#endif
#endif
typedef int cl_int;
typedef unsigned int cl_uint;
typedef void* cl_platform_id;
typedef void* cl_device_id;
#define CL_SUCCESS           0
#define CL_DEVICE_TYPE_GPU   4u
#define CL_PLATFORM_NAME     0x0902
#define CL_PLATFORM_VENDOR   0x0903
#define CL_DEVICE_NAME       0x102B
#define CL_DEVICE_VENDOR     0x102C

// Safe OpenCL info query: allocate size+1 and pass size+1 to avoid Intel driver AV (some return size w/o null).
static bool get_opencl_string(void* platform_or_device, bool is_platform, unsigned int param,
    cl_int (CL_API_CALL* p_get_platform_info)(cl_platform_id, unsigned int, size_t, void*, size_t*),
    cl_int (CL_API_CALL* p_get_device_info)(cl_device_id, unsigned int, size_t, void*, size_t*),
    std::string& out) {
    size_t sz = 0;
    cl_int err = is_platform
        ? p_get_platform_info((cl_platform_id)platform_or_device, param, 0, nullptr, &sz)
        : p_get_device_info((cl_device_id)platform_or_device, param, 0, nullptr, &sz);
    if (err != CL_SUCCESS || sz == 0 || sz > 65536) return false;
    std::vector<char> buf(sz + 1, '\0');
    err = is_platform
        ? p_get_platform_info((cl_platform_id)platform_or_device, param, buf.size(), buf.data(), nullptr)
        : p_get_device_info((cl_device_id)platform_or_device, param, buf.size(), buf.data(), nullptr);
    if (err != CL_SUCCESS) return false;
    out = buf.data();
    while (!out.empty() && (out.back() == '\0' || out.back() == '\n')) out.pop_back();
    return true;
}

static void print_opencl_devices_impl() {
    typedef cl_int (CL_API_CALL* p_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
    typedef cl_int (CL_API_CALL* p_clGetDeviceIDs)(cl_platform_id, cl_uint, cl_uint, cl_device_id*, cl_uint*);
    typedef cl_int (CL_API_CALL* p_clGetPlatformInfo)(cl_platform_id, unsigned int, size_t, void*, size_t*);
    typedef cl_int (CL_API_CALL* p_clGetDeviceInfo)(cl_device_id, unsigned int, size_t, void*, size_t*);
#ifdef _WIN32
    HMODULE h = LoadLibraryA("OpenCL.dll");
    if (!h) { fprintf(stderr, "[tariff_mnn] OpenCL: LoadLibrary(OpenCL.dll) failed.\n"); return; }
    auto p_get_platform_ids = (p_clGetPlatformIDs)GetProcAddress(h, "clGetPlatformIDs");
    auto p_get_device_ids   = (p_clGetDeviceIDs)GetProcAddress(h, "clGetDeviceIDs");
    auto p_get_platform_info= (p_clGetPlatformInfo)GetProcAddress(h, "clGetPlatformInfo");
    auto p_get_device_info = (p_clGetDeviceInfo)GetProcAddress(h, "clGetDeviceInfo");
#else
    void* h = dlopen("libOpenCL.so.1", RTLD_LAZY);
    if (!h) h = dlopen("libOpenCL.so", RTLD_LAZY);
    if (!h) { fprintf(stderr, "[tariff_mnn] OpenCL: dlopen(libOpenCL.so) failed.\n"); return; }
    auto p_get_platform_ids = (p_clGetPlatformIDs)dlsym(h, "clGetPlatformIDs");
    auto p_get_device_ids   = (p_clGetDeviceIDs)dlsym(h, "clGetDeviceIDs");
    auto p_get_platform_info= (p_clGetPlatformInfo)dlsym(h, "clGetPlatformInfo");
    auto p_get_device_info = (p_clGetDeviceInfo)dlsym(h, "clGetDeviceInfo");
#endif
    if (!p_get_platform_ids || !p_get_device_ids || !p_get_platform_info || !p_get_device_info) {
        fprintf(stderr, "[tariff_mnn] OpenCL: failed to get function pointers.\n"); return;
    }
    cl_uint num_platforms = 0;
    if (p_get_platform_ids(0, nullptr, &num_platforms) != CL_SUCCESS || num_platforms == 0) {
        fprintf(stderr, "[tariff_mnn] OpenCL: no platforms.\n"); return;
    }
    std::vector<cl_platform_id> platforms(num_platforms);
    if (p_get_platform_ids(num_platforms, platforms.data(), nullptr) != CL_SUCCESS) return;
    fprintf(stderr, "[tariff_mnn] OpenCL devices (use platform_id / device_id to select):\n");
    for (cl_uint p = 0; p < num_platforms; p++) {
        if (!platforms[p]) continue;
        std::string name;
        if (!get_opencl_string(platforms[p], true, CL_PLATFORM_NAME, p_get_platform_info, p_get_device_info, name))
            name = "(unknown platform)";
        fprintf(stderr, "  Platform %u: %s\n", p, name.c_str());
        cl_uint num_devices = 0;
        if (p_get_device_ids(platforms[p], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices) != CL_SUCCESS) continue;
        std::vector<cl_device_id> devices(num_devices);
        if (p_get_device_ids(platforms[p], CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr) != CL_SUCCESS) continue;
        for (cl_uint d = 0; d < num_devices; d++) {
            if (!devices[d]) continue;
            std::string dname;
            if (!get_opencl_string(devices[d], false, CL_DEVICE_NAME, p_get_platform_info, p_get_device_info, dname))
                dname = "(unknown device)";
            fprintf(stderr, "    Device %u: %s  --> platform_id=%u, device_id=%u\n", d, dname.c_str(), p, d);
        }
    }
}

// Log selected OpenCL platform/device at initiation (before gpu_device_config_ is used).
static void log_opencl_selection(int platform_size, int platform_id, int device_id) {
    typedef cl_int (CL_API_CALL* p_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
    typedef cl_int (CL_API_CALL* p_clGetDeviceIDs)(cl_platform_id, cl_uint, cl_uint, cl_device_id*, cl_uint*);
    typedef cl_int (CL_API_CALL* p_clGetPlatformInfo)(cl_platform_id, unsigned int, size_t, void*, size_t*);
    typedef cl_int (CL_API_CALL* p_clGetDeviceInfo)(cl_device_id, unsigned int, size_t, void*, size_t*);
#ifdef _WIN32
    HMODULE h = LoadLibraryA("OpenCL.dll");
    if (!h) { fprintf(stderr, "[tariff_mnn] OpenCL: platform_id=%d, device_id=%d (names unavailable)\n", platform_id, device_id); return; }
    auto p_get_platform_ids = (p_clGetPlatformIDs)GetProcAddress(h, "clGetPlatformIDs");
    auto p_get_device_ids   = (p_clGetDeviceIDs)GetProcAddress(h, "clGetDeviceIDs");
    auto p_get_platform_info= (p_clGetPlatformInfo)GetProcAddress(h, "clGetPlatformInfo");
    auto p_get_device_info = (p_clGetDeviceInfo)GetProcAddress(h, "clGetDeviceInfo");
#else
    void* h = dlopen("libOpenCL.so.1", RTLD_LAZY);
    if (!h) h = dlopen("libOpenCL.so", RTLD_LAZY);
    if (!h) { fprintf(stderr, "[tariff_mnn] OpenCL: platform_id=%d, device_id=%d (names unavailable)\n", platform_id, device_id); return; }
    auto p_get_platform_ids = (p_clGetPlatformIDs)dlsym(h, "clGetPlatformIDs");
    auto p_get_device_ids   = (p_clGetDeviceIDs)dlsym(h, "clGetDeviceIDs");
    auto p_get_platform_info= (p_clGetPlatformInfo)dlsym(h, "clGetPlatformInfo");
    auto p_get_device_info = (p_clGetDeviceInfo)dlsym(h, "clGetDeviceInfo");
#endif
    if (!p_get_platform_ids || !p_get_device_ids || !p_get_platform_info || !p_get_device_info) {
        fprintf(stderr, "[tariff_mnn] OpenCL: platform_id=%d, device_id=%d\n", platform_id, device_id); return;
    }
    cl_uint num_platforms = 0;
    if (p_get_platform_ids(0, nullptr, &num_platforms) != CL_SUCCESS || num_platforms == 0 ||
        (cl_uint)platform_id >= num_platforms) {
        fprintf(stderr, "[tariff_mnn] OpenCL: platform_id=%d, device_id=%d\n", platform_id, device_id); return;
    }
    std::vector<cl_platform_id> platforms(num_platforms);
    if (p_get_platform_ids(num_platforms, platforms.data(), nullptr) != CL_SUCCESS || !platforms[platform_id]) {
        fprintf(stderr, "[tariff_mnn] OpenCL: platform_id=%d, device_id=%d\n", platform_id, device_id); return;
    }
    std::string platform_name;
    get_opencl_string(platforms[platform_id], true, CL_PLATFORM_NAME, p_get_platform_info, p_get_device_info, platform_name);
    cl_uint num_devices = 0;
    if (p_get_device_ids(platforms[platform_id], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices) != CL_SUCCESS ||
        (cl_uint)device_id >= num_devices) {
        fprintf(stderr, "[tariff_mnn] OpenCL: platform_id=%d (%s), device_id=%d\n",
                platform_id, platform_name.empty() ? "?" : platform_name.c_str(), device_id); return;
    }
    std::vector<cl_device_id> devices(num_devices);
    if (p_get_device_ids(platforms[platform_id], CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr) != CL_SUCCESS ||
        !devices[device_id]) {
        fprintf(stderr, "[tariff_mnn] OpenCL: platform_id=%d (%s), device_id=%d\n",
                platform_id, platform_name.empty() ? "?" : platform_name.c_str(), device_id); return;
    }
    std::string device_name;
    get_opencl_string(devices[device_id], false, CL_DEVICE_NAME, p_get_platform_info, p_get_device_info, device_name);
    fprintf(stderr, "[tariff_mnn] OpenCL: platform_id=%d, device_id=%d, platform=%s, device=%s\n",
            platform_id, device_id,
            platform_name.empty() ? "?" : platform_name.c_str(),
            device_name.empty() ? "?" : device_name.c_str());
}

enum class ModelType { NB202, PWR };

class TariffProcessor {
public:
    TariffProcessor(const std::string& feat_path, const std::string& fusion_path,
                    const std::string& model_type = "nb202",
                    int platform_size = 1, int platform_id = 0, int device_id = 0) {
        model_type_ = (model_type == "pwr") ? ModelType::PWR : ModelType::NB202;

        log_opencl_selection(platform_size, platform_id, device_id);

        gpu_device_config_.platformSize = static_cast<uint32_t>(platform_size);
        gpu_device_config_.platformId   = static_cast<uint32_t>(platform_id);
        gpu_device_config_.deviceId    = static_cast<uint32_t>(device_id);

        BackendConfig backendConfig;
        backendConfig.precision = BackendConfig::Precision_Low;
        backendConfig.power = BackendConfig::Power_High;
        backendConfig.sharedContext = &gpu_device_config_;
        // backendConfig.memory = BackendConfig::Memory_Low;
        
        ScheduleConfig feat_config;
        feat_config.backendConfig = &backendConfig;
        // feat_config.type  = MNN_FORWARD_CPU;
        feat_config.type  = MNN_FORWARD_OPENCL;
        // feat_config.type  = MNN_FORWARD_VULKAN;
        feat_config.mode = MNN_GPU_TUNING_WIDE | MNN_GPU_MEMORY_BUFFER;

        ScheduleConfig fusion_config;
        fusion_config.backendConfig = &backendConfig;
        // fusion_config.type  = MNN_FORWARD_CPU;
        fusion_config.type  = MNN_FORWARD_OPENCL;
        // fusion_config.type  = MNN_FORWARD_VULKAN;
        fusion_config.mode = MNN_GPU_TUNING_WIDE | MNN_GPU_MEMORY_BUFFER;

        auto runtimeInfo = Interpreter::createRuntime({feat_config, fusion_config});

        feat_net.reset(Interpreter::createFromFile(feat_path.c_str()), Interpreter::destroy);
        feat_net->setSessionMode(Interpreter::Session_Release);
        feat_net->setSessionMode(Interpreter::Session_Output_User);  // outputs stay on GPU, separable for GPU->GPU feed
        feat_net->setCacheFile("feat.cache");
        feat_session = feat_net->createSession(feat_config, runtimeInfo);

        fusion_net.reset(Interpreter::createFromFile(fusion_path.c_str()), Interpreter::destroy);
        fusion_net->setSessionMode(Interpreter::Session_Release);
        // Session_Input_Inside (default): session allocates inputs; we feed via copyFromHostTensor (device->device when source is GPU)
        fusion_net->setCacheFile("fusion.cache");
        fusion_session = fusion_net->createSession(fusion_config, runtimeInfo);

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
        py::array_t<float, py::array::c_style | py::array::forcecast> timesteps) {

        auto img0_buf = img0.request();
        auto img1_buf = img1.request();
        auto ts_buf = timesteps.request();

        if (img0_buf.ndim != 4 || img1_buf.ndim != 4)
            throw std::runtime_error("img0, img1 must be 4D arrays (BCHW)");

        const int B = img0_buf.shape[0], C = img0_buf.shape[1], H = img0_buf.shape[2], W = img0_buf.shape[3];

        run_feature(img0_buf.ptr, img1_buf.ptr, B, C, H, W);

        if (model_type_ == ModelType::NB202) {
            cached_features["img0"] = std::shared_ptr<Tensor>(Tensor::create<float>({B, C, H, W}, img0_buf.ptr, Tensor::CAFFE), Tensor::destroy);
            cached_features["img1"] = std::shared_ptr<Tensor>(Tensor::create<float>({B, C, H, W}, img1_buf.ptr, Tensor::CAFFE), Tensor::destroy);
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
    MNNDeviceContext gpu_device_config_;
    std::shared_ptr<Interpreter> feat_net, fusion_net;
    Session *feat_session, *fusion_session;
    std::vector<std::string> feature_outputs_;
    std::unordered_map<std::string, std::shared_ptr<Tensor>> cached_features;

    void run_feature(void* img0_ptr, void* img1_ptr, int B, int C, int H, int W) {
        auto feat_img0 = feat_net->getSessionInput(feat_session, "img0");
        auto feat_img1 = feat_net->getSessionInput(feat_session, "img1");

        copy_data_to_tensor(feat_img0, img0_ptr, B, C, H, W);
        copy_data_to_tensor(feat_img1, img1_ptr, B, C, H, W);

        feat_net->runSession(feat_session);

        // Sync: wait for feature outputs on GPU before using as fusion inputs (avoids reading stale buffer)
        auto first_out = feat_net->getSessionOutput(feat_session, feature_outputs_[0].c_str());
        if (first_out && first_out->deviceId() != 0) {
            first_out->wait(Tensor::MAP_TENSOR_READ, true);
        }

        // Keep feature outputs on GPU (no copyToHostTensor); OpenCL backend will use copyBetweenDevice when feeding fusion
        for (const auto& name : feature_outputs_) {
            auto tensor = feat_net->getSessionOutput(feat_session, name.c_str());
            cached_features[name] = std::shared_ptr<Tensor>(tensor, [](Tensor*) {});
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
                // Cached feature: device tensor when from run_feature -> OpenCL copyBetweenDevice (GPU-GPU)
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
    m.def("print_opencl_devices", &print_opencl_devices_impl,
          "Print OpenCL platforms and GPU devices (clinfo-style). Use platform_id/device_id when creating TariffProcessor.");
    py::class_<TariffProcessor>(m, "TariffProcessor")
        .def(py::init<const std::string&, const std::string&, const std::string&, int, int, int>(),
             py::arg("feat_path"), py::arg("fusion_path"), py::arg("model_type") = "nb202",
             py::arg("platform_size") = 1, py::arg("platform_id") = 0, py::arg("device_id") = 0)
        .def("process", &TariffProcessor::process,
             py::arg("img0"), py::arg("img1"), py::arg("timesteps"));
}
