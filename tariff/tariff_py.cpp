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
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

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

// This is deliberately separate from MNN_VERSION.  Bump it whenever the
// embedded OpenCL source, custom kernel build options, or the tariff graph
// cache contract changes.  MNN's cache validator checks the device and MNN
// version, but it cannot know that a generated tariff kernel source changed.
static constexpr const char* kTariffOpenCLCacheSchema =
    "tariff-opencl-v3-fixed-softsplat";

static uint64_t tariff_fnv1a(const std::string& value) {
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char c : value) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

static std::string tariff_hash_hex(uint64_t value) {
    std::ostringstream stream;
    stream << std::hex << std::setw(16) << std::setfill('0') << value;
    return stream.str();
}

static std::string tariff_sanitize_tag(const std::string& value) {
    std::string result;
    result.reserve(value.size());
    for (unsigned char c : value) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_') {
            result.push_back(static_cast<char>(c));
        } else {
            result.push_back('_');
        }
    }
    return result.empty() ? "default" : result;
}

static std::string tariff_parent_path(const std::string& path) {
    const auto separator = path.find_last_of("/\\");
    if (separator == std::string::npos) {
        return ".";
    }
    return path.substr(0, separator);
}

static std::string tariff_join_path(const std::string& directory,
                                    const std::string& filename,
                                    const std::string& reference_path) {
    if (directory.empty() || directory == ".") {
        return filename;
    }
    const char separator = reference_path.find('\\') != std::string::npos ? '\\' : '/';
    return directory + separator + filename;
}

static bool tariff_file_exists(const std::string& path) {
    std::ifstream file(path.c_str(), std::ios::binary);
    return file.good();
}

static std::string make_tariff_opencl_cache_path(const std::string& feat_path,
                                                 const std::string& fusion_path,
                                                 const std::string& model_type,
                                                 int platform_size,
                                                 int platform_id,
                                                 int device_id) {
    const char* override_path = std::getenv("TARIFF_OPENCL_CACHE_FILE");
    if (override_path != nullptr && override_path[0] != '\0') {
        return std::string(override_path);
    }

    const char* namespace_env = std::getenv("TARIFF_OPENCL_CACHE_NAMESPACE");
    const std::string cache_namespace = tariff_sanitize_tag(
        namespace_env != nullptr && namespace_env[0] != '\0' ? namespace_env : "default");

    // The path hash keeps PWR/NB202, model revisions, and model pairs from
    // sharing a cache file.  The schema is also a manual invalidation point
    // for changes to generated OpenCL code or build macros.
    std::ostringstream identity;
    identity << kTariffOpenCLCacheSchema << '|'
             << cache_namespace << '|'
             << model_type << '|'
             << feat_path << '|'
             << fusion_path << '|'
             << platform_size << '|'
             << platform_id << '|'
             << device_id;

    const std::string filename =
        "tariff_opencl_" + cache_namespace + "_" + model_type +
        "_p" + std::to_string(platform_id) +
        "_d" + std::to_string(device_id) +
        "_" + tariff_hash_hex(tariff_fnv1a(identity.str())) + ".cache";
    return tariff_join_path(tariff_parent_path(feat_path), filename, feat_path);
}

static size_t tensor_element_count(const std::vector<int>& shape) {
    size_t result = 1;
    for (int dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error("tensor shape contains a non-positive dimension");
        }
        result *= static_cast<size_t>(dim);
    }
    return result;
}

struct HostTensorData {
    std::vector<int> shape;
    std::vector<float> data;

    explicit HostTensorData(std::vector<int> shapeValue)
        : shape(std::move(shapeValue)), data(tensor_element_count(shape), 0.0f) {}
};

class TariffProcessor {
public:
    TariffProcessor(const std::string& feat_path, const std::string& fusion_path,
                    const std::string& model_type = "nb202",
                    int platform_size = 1, int platform_id = 0, int device_id = 0)
        : feat_session(nullptr), fusion_session(nullptr) {
        model_type_ = (model_type == "pwr") ? ModelType::PWR : ModelType::NB202;
        const char* debug_env = std::getenv("TARIFF_DEBUG_TENSORS");
        debug_tensors_ = debug_env != nullptr && std::string(debug_env) != "0";
        const char* debug_ops_env = std::getenv("TARIFF_DEBUG_OPS");
        debug_ops_ = debug_ops_env != nullptr && std::string(debug_ops_env) != "0";
        const char* debug_softsplat_env = std::getenv("TARIFF_DEBUG_SOFTSPLAT");
        debug_softsplat_ = debug_softsplat_env != nullptr && std::string(debug_softsplat_env) != "0";
        const char* backend_env = std::getenv("TARIFF_BACKEND");
        use_cpu_ = backend_env != nullptr && std::string(backend_env) == "cpu";

        log_opencl_selection(platform_size, platform_id, device_id);

        gpu_device_config_.platformSize = static_cast<uint32_t>(platform_size);
        gpu_device_config_.platformId = static_cast<uint32_t>(platform_id);
        gpu_device_config_.deviceId = static_cast<uint32_t>(device_id);

        BackendConfig backendConfig;
        backendConfig.precision = BackendConfig::Precision_Low;
        backendConfig.power = BackendConfig::Power_High;
        backendConfig.sharedContext = &gpu_device_config_;

        ScheduleConfig feat_config;
        feat_config.backendConfig = &backendConfig;
        feat_config.type = use_cpu_ ? MNN_FORWARD_CPU : MNN_FORWARD_OPENCL;
        feat_config.mode = MNN_GPU_TUNING_WIDE | MNN_GPU_MEMORY_BUFFER;

        ScheduleConfig fusion_config;
        fusion_config.backendConfig = &backendConfig;
        fusion_config.type = use_cpu_ ? MNN_FORWARD_CPU : MNN_FORWARD_OPENCL;
        fusion_config.mode = MNN_GPU_TUNING_WIDE | MNN_GPU_MEMORY_BUFFER;

        auto runtimeInfo = Interpreter::createRuntime({feat_config, fusion_config});
        backend_config_ = backendConfig;
        feat_config_ = feat_config;
        feat_config_.backendConfig = &backend_config_;
        fusion_config_ = fusion_config;
        fusion_config_.backendConfig = &backend_config_;
        runtime_info_ = runtimeInfo;

        feat_net.reset(Interpreter::createFromFile(feat_path.c_str()), Interpreter::destroy);
        if (!feat_net) {
            throw std::runtime_error("failed to load feature MNN model: " + feat_path);
        }
        // Both networks share runtime_info_, hence one cache owner is enough.
        // Loading through only the feature interpreter avoids loading the same
        // OpenCL binary set twice into the same runtime.  The cache contains
        // runtime programs/tuning, not model weights or Tensor allocations.
        if (!use_cpu_) {
            const std::string cache_model_type =
                model_type_ == ModelType::PWR ? "pwr" : "nb202";
            opencl_cache_file_ = make_tariff_opencl_cache_path(
                feat_path, fusion_path, cache_model_type,
                platform_size, platform_id, device_id);
            opencl_cache_enabled_ = true;
            std::fprintf(stderr, "[tariff_mnn] opencl_cache=%s existing=%s\n",
                         opencl_cache_file_.c_str(),
                         tariff_file_exists(opencl_cache_file_) ? "yes" : "no");
            feat_net->setCacheFile(opencl_cache_file_.c_str());
        }

        // Keep output tensors separable so fusion can consume them on the
        // same OpenCL runtime.  Do not use the old global feat.cache file:
        // it can describe another graph and corrupt a joint feature/fusion
        // session before the first inference.
        feat_net->setSessionMode(Interpreter::Session_Output_User);
        feat_session = feat_net->createSession(feat_config, runtimeInfo);
        if (!feat_session) {
            throw std::runtime_error("failed to create feature MNN session");
        }

        fusion_net.reset(Interpreter::createFromFile(fusion_path.c_str()), Interpreter::destroy);
        if (!fusion_net) {
            throw std::runtime_error("failed to load fusion MNN model: " + fusion_path);
        }
        fusion_net->setSessionMode(Interpreter::Session_Output_User);
        fusion_session = fusion_net->createSession(fusion_config_, runtime_info_);
        if (!fusion_session) {
            throw std::runtime_error("failed to create fusion MNN session");
        }

        dynamic_feature_ = feat_net->getSessionInput(feat_session, "coords") != nullptr;
        dynamic_fusion_ = fusion_net->getSessionInput(fusion_session, "timestep") != nullptr;

        if (model_type_ == ModelType::NB202) {
            feature_outputs_ = {"flow01", "flow10", "metric0", "metric1",
                                "feat11", "feat12", "feat13", "feat21", "feat22", "feat23"};
        } else {
            feature_outputs_ = {"img0_out", "img1_out", "flow01", "flow10", "metric0", "metric1",
                                "f0", "f1"};
        }

        std::fprintf(stderr, "[tariff_mnn] backend=%s feature_inputs=%s fusion_inputs=%s model_type=%s\n",
                     use_cpu_ ? "cpu" : "opencl",
                     dynamic_feature_ ? "dynamic-grid" : "legacy",
                     dynamic_fusion_ ? "dynamic" : "legacy",
                     model_type_ == ModelType::PWR ? "pwr" : "nb202");
    }

    ~TariffProcessor() {
        release_fusion_sessions();
        if (feat_session != nullptr && feat_net) {
            feat_net->releaseSession(feat_session);
            feat_session = nullptr;
        }
    }

    void prepare_feature(
        py::array_t<float, py::array::c_style | py::array::forcecast> img0,
        py::array_t<float, py::array::c_style | py::array::forcecast> img1) {
        // A cached request owns one frame pair.  Rebuild the fusion session
        // once at this boundary so its execution-local OpenCL state cannot
        // leak from the preceding frame pair; all timesteps in this request
        // still reuse that one session.
        release_fusion_sessions();
        auto img0_buf = img0.request();
        auto img1_buf = img1.request();
        if (img0_buf.ndim != 4 || img1_buf.ndim != 4) {
            throw std::runtime_error("img0 and img1 must be 4D BCHW arrays");
        }
        for (int dim = 0; dim < 4; ++dim) {
            if (img0_buf.shape[dim] != img1_buf.shape[dim]) {
                throw std::runtime_error("img0 and img1 must have identical BCHW shapes");
            }
        }

        const int B = static_cast<int>(img0_buf.shape[0]);
        const int C = static_cast<int>(img0_buf.shape[1]);
        const int H = static_cast<int>(img0_buf.shape[2]);
        const int W = static_cast<int>(img0_buf.shape[3]);
        if (B <= 0 || C != 3 || H <= 0 || W <= 0) {
            throw std::runtime_error("Tariff expects BCHW with C=3 and positive dimensions");
        }

        // NB202 keeps the frame inputs in cached_features for fusion.  Own
        // those host buffers across Python calls; a Tensor that points into a
        // temporary py::array would become dangling before process_cached().
        cached_features.clear();
        const size_t frameElements = static_cast<size_t>(B) * C * H * W;
        prepared_img0_.assign(static_cast<const float*>(img0_buf.ptr),
                              static_cast<const float*>(img0_buf.ptr) + frameElements);
        prepared_img1_.assign(static_cast<const float*>(img1_buf.ptr),
                              static_cast<const float*>(img1_buf.ptr) + frameElements);
        run_feature(prepared_img0_.data(), prepared_img1_.data(), B, C, H, W);

        if (model_type_ == ModelType::NB202) {
            cached_features["img0"] = std::shared_ptr<Tensor>(
                Tensor::create<float>({B, C, H, W}, prepared_img0_.data(), Tensor::CAFFE), Tensor::destroy);
            cached_features["img1"] = std::shared_ptr<Tensor>(
                Tensor::create<float>({B, C, H, W}, prepared_img1_.data(), Tensor::CAFFE), Tensor::destroy);
        }
        reset_fusion_session();
        // The feature tensors are refreshed even when their shapes stay the
        // same.  Re-encode the fusion graph once after that refresh so the
        // first cached-timestep execution cannot retain input/device state
        // from the preceding normal process() call.  This is one fusion
        // resize per multi-timestep request, not a feature recomputation.
        fusion_input_shapes_.clear();
        prepared_B_ = B;
        prepared_H_ = H;
        prepared_W_ = W;
        feature_prepared_ = true;
    }

    py::list process_cached(
        py::array_t<float, py::array::c_style | py::array::forcecast> timesteps) {
        if (!feature_prepared_) {
            throw std::runtime_error("prepare_feature must be called before process_cached");
        }
        auto ts_buf = timesteps.request();
        if (ts_buf.ndim != 4) {
            throw std::runtime_error("timesteps must be a 4D BTHW array");
        }
        const int B = static_cast<int>(ts_buf.shape[0]);
        const int T = static_cast<int>(ts_buf.shape[1]);
        const int H = static_cast<int>(ts_buf.shape[2]);
        const int W = static_cast<int>(ts_buf.shape[3]);
        const int expectedH = model_type_ == ModelType::NB202 ? prepared_H_ / 2 : prepared_H_;
        const int expectedW = model_type_ == ModelType::NB202 ? prepared_W_ / 2 : prepared_W_;
        if (B != prepared_B_ || T <= 0 || H != expectedH || W != expectedW) {
            throw std::runtime_error("timesteps shape does not match prepared feature shape");
        }

        const float* timestepsPtr = static_cast<const float*>(ts_buf.ptr);
        const size_t timestepPlane = static_cast<size_t>(H) * W;
        const size_t timestepBatchStride = static_cast<size_t>(B) * timestepPlane;
        std::vector<float> timestepBatch(timestepBatchStride);
        std::vector<py::array_t<float>> outputs;
        outputs.reserve(static_cast<size_t>(T));
        for (int t = 0; t < T; ++t) {
            for (int b = 0; b < B; ++b) {
                const size_t sourceOffset =
                    (static_cast<size_t>(b) * T + t) * timestepPlane;
                const size_t targetOffset = static_cast<size_t>(b) * timestepPlane;
                std::memcpy(timestepBatch.data() + targetOffset,
                            timestepsPtr + sourceOffset,
                            timestepPlane * sizeof(float));
            }
            // OLS owns one resolution per process.  Reuse the already encoded
            // fusion session for every timestep: the feature graph has already
            // run once above, and the custom softsplat execution clears its
            // private accumulation buffer at the start of every run.
            auto out_tensor = run_fusion_model(timestepBatch.data(), B, H, W);
            outputs.emplace_back(create_numpy_array(out_tensor.get(), prepared_H_, prepared_W_));
        }
        return concatenate_outputs(outputs);
    }

    py::list process(
        py::array_t<float, py::array::c_style | py::array::forcecast> img0,
        py::array_t<float, py::array::c_style | py::array::forcecast> img1,
        py::array_t<float, py::array::c_style | py::array::forcecast> timesteps) {
        auto img0_buf = img0.request();
        auto img1_buf = img1.request();
        auto ts_buf = timesteps.request();

        if (img0_buf.ndim != 4 || img1_buf.ndim != 4) {
            throw std::runtime_error("img0 and img1 must be 4D BCHW arrays");
        }
        if (ts_buf.ndim != 4) {
            throw std::runtime_error("timesteps must be a 4D BTHW array");
        }
        for (int dim = 0; dim < 4; ++dim) {
            if (img0_buf.shape[dim] != img1_buf.shape[dim]) {
                throw std::runtime_error("img0 and img1 must have identical BCHW shapes");
            }
        }

        const int B = static_cast<int>(img0_buf.shape[0]);
        const int C = static_cast<int>(img0_buf.shape[1]);
        const int H = static_cast<int>(img0_buf.shape[2]);
        const int W = static_cast<int>(img0_buf.shape[3]);
        if (B <= 0 || C != 3 || H <= 0 || W <= 0) {
            throw std::runtime_error("Tariff expects BCHW with C=3 and positive dimensions");
        }

        const int tsB = static_cast<int>(ts_buf.shape[0]);
        const int T = static_cast<int>(ts_buf.shape[1]);
        const int tsH = static_cast<int>(ts_buf.shape[2]);
        const int tsW = static_cast<int>(ts_buf.shape[3]);
        const int expectedTsH = model_type_ == ModelType::NB202 ? H / 2 : H;
        const int expectedTsW = model_type_ == ModelType::NB202 ? W / 2 : W;
        if (tsB != B || T <= 0 || expectedTsH <= 0 || expectedTsW <= 0 ||
            tsH != expectedTsH || tsW != expectedTsW) {
            throw std::runtime_error(
                "timesteps shape must be [B,T,H/2,W/2] for NB202 or [B,T,H,W] for PWR");
        }

        const int paddedH = align_up_64(H);
        const int paddedW = align_up_64(W);
        const int paddedTsH = model_type_ == ModelType::NB202 ? paddedH / 2 : paddedH;
        const int paddedTsW = model_type_ == ModelType::NB202 ? paddedW / 2 : paddedW;
        const bool needsPadding = paddedH != H || paddedW != W;

        std::vector<float> paddedImg0;
        std::vector<float> paddedImg1;
        const float* img0Ptr = static_cast<const float*>(img0_buf.ptr);
        const float* img1Ptr = static_cast<const float*>(img1_buf.ptr);
        if (needsPadding) {
            paddedImg0 = pad_spatial(img0Ptr, B * C, H, W, paddedH, paddedW);
            paddedImg1 = pad_spatial(img1Ptr, B * C, H, W, paddedH, paddedW);
            img0Ptr = paddedImg0.data();
            img1Ptr = paddedImg1.data();
        }

        std::vector<float> paddedTimesteps;
        const float* timestepsPtr = static_cast<const float*>(ts_buf.ptr);
        if (needsPadding) {
            paddedTimesteps = pad_spatial(timestepsPtr, B * T, tsH, tsW, paddedTsH, paddedTsW);
            timestepsPtr = paddedTimesteps.data();
        }

        // A normal process() call is also a new frame-pair request.  Drop the
        // previous fusion session before refreshing feature outputs, then
        // create one clean fusion session for all timesteps below.  Without
        // this boundary, a direct batch call can retain execution-local
        // OpenCL state from the preceding request and disagree with three
        // independent T=1 calls even though the feature tensors are equal.
        release_fusion_sessions();
        feature_prepared_ = false;
        run_feature(const_cast<float*>(img0Ptr), const_cast<float*>(img1Ptr), B, C, paddedH, paddedW);

        if (model_type_ == ModelType::NB202) {
            cached_features["img0"] = std::shared_ptr<Tensor>(
                Tensor::create<float>({B, C, paddedH, paddedW}, const_cast<float*>(img0Ptr), Tensor::CAFFE), Tensor::destroy);
            cached_features["img1"] = std::shared_ptr<Tensor>(
                Tensor::create<float>({B, C, paddedH, paddedW}, const_cast<float*>(img1Ptr), Tensor::CAFFE), Tensor::destroy);
        }
        reset_fusion_session();

        std::vector<py::array_t<float>> outputs;
        outputs.reserve(static_cast<size_t>(T));
        const size_t timestepPlane = static_cast<size_t>(paddedTsH) * paddedTsW;
        const size_t timestepBatchStride = static_cast<size_t>(B) * timestepPlane;
        std::vector<float> timestepBatch(timestepBatchStride);
        for (int t = 0; t < T; ++t) {
            for (int b = 0; b < B; ++b) {
                const size_t sourceOffset =
                    (static_cast<size_t>(b) * T + t) * timestepPlane;
                const size_t targetOffset = static_cast<size_t>(b) * timestepPlane;
                std::memcpy(timestepBatch.data() + targetOffset,
                            timestepsPtr + sourceOffset,
                            timestepPlane * sizeof(float));
            }
            auto out_tensor = run_fusion_model(timestepBatch.data(), B, paddedTsH, paddedTsW);
            outputs.emplace_back(create_numpy_array(out_tensor.get(), H, W));
        }
        return concatenate_outputs(outputs);
    }

private:
    static int align_up_64(int value) {
        // IFNet has five spatial stages; its concat paths require 64-aligned
        // input sizes even though the feature/grid path itself needs 32.
        return ((value + 63) / 64) * 64;
    }

    static std::vector<float> pad_spatial(const float* source, int outer, int height, int width,
                                          int paddedHeight, int paddedWidth) {
        const size_t sourcePlane = static_cast<size_t>(height) * width;
        const size_t paddedPlane = static_cast<size_t>(paddedHeight) * paddedWidth;
        std::vector<float> result(static_cast<size_t>(outer) * paddedPlane);
        for (int n = 0; n < outer; ++n) {
            const float* sourceBase = source + static_cast<size_t>(n) * sourcePlane;
            float* targetBase = result.data() + static_cast<size_t>(n) * paddedPlane;
            for (int y = 0; y < paddedHeight; ++y) {
                const int sourceY = y < height - 1 ? y : height - 1;
                const float* sourceRow = sourceBase + static_cast<size_t>(sourceY) * width;
                float* targetRow = targetBase + static_cast<size_t>(y) * paddedWidth;
                for (int x = 0; x < paddedWidth; ++x) {
                    const int sourceX = x < width - 1 ? x : width - 1;
                    targetRow[x] = sourceRow[sourceX];
                }
            }
        }
        return result;
    }

    ModelType model_type_;
    MNNDeviceContext gpu_device_config_;
    BackendConfig backend_config_;
    ScheduleConfig fusion_config_;
    RuntimeInfo runtime_info_;
    std::shared_ptr<Interpreter> feat_net;
    std::shared_ptr<Interpreter> fusion_net;
    Session* feat_session;
    Session* fusion_session;
    std::vector<Session*> retired_fusion_sessions_;
    ScheduleConfig feat_config_;
    bool feature_shape_initialized_ = false;
    std::vector<int> feature_shape_;
    std::vector<std::string> feature_outputs_;
    std::unordered_map<std::string, std::shared_ptr<Tensor>> cached_features;
    std::unordered_map<std::string, std::vector<int>> fusion_input_shapes_;
    std::vector<float> prepared_img0_;
    std::vector<float> prepared_img1_;
    int prepared_B_ = 0;
    int prepared_H_ = 0;
    int prepared_W_ = 0;
    bool feature_prepared_ = false;
    bool dynamic_feature_ = false;
    bool dynamic_fusion_ = false;
    bool debug_tensors_ = false;
    bool debug_ops_ = false;
    bool debug_softsplat_ = false;
    bool use_cpu_ = false;
    bool opencl_cache_enabled_ = false;
    bool opencl_cache_feature_saved_ = false;
    bool opencl_cache_fusion_saved_ = false;
    std::string opencl_cache_file_;
    std::vector<float> debug_softsplat_input_;
    std::vector<float> debug_softsplat_flow_;
    std::vector<int> debug_softsplat_shape_;

    static std::unordered_map<std::string, HostTensorData> make_dynamic_cache(int B, int H, int W) {
        const int flowH = H / 2;
        const int flowW = W / 2;
        const int h16 = flowH / 16;
        const int w16 = flowW / 16;
        const int h8 = flowH / 8;
        const int w8 = flowW / 8;
        const int points16 = h16 * w16;
        const int points8 = h8 * w8;

        std::unordered_map<std::string, HostTensorData> cache;
        cache.emplace("coords", HostTensorData({B, 2, flowH, flowW}));
        cache.emplace("backwarp_grid", HostTensorData({B, 2, flowH, flowW}));
        cache.emplace("pos_s16", HostTensorData({B * 2, 2, h16, w16}));
        cache.emplace("grid_s16", HostTensorData({B, 2, h16, w16}));
        cache.emplace("flatten_grid_s16", HostTensorData({B, points16, 2}));
        cache.emplace("grid_s8", HostTensorData({B, 2, h8, w8}));
        cache.emplace("delta_s16", HostTensorData({B * points16, 9, 9, 2}));
        cache.emplace("delta_s8", HostTensorData({B * points8, 9, 9, 2}));
        cache.emplace("radius_emb_s16", HostTensorData({B, 1, h16, w16}));
        cache.emplace("radius_emb_s8", HostTensorData({B, 1, h8, w8}));
        cache.emplace("init_context_s16", HostTensorData({B, 64, h16, w16}));
        cache.emplace("init_context_s8", HostTensorData({B, 64, h8, w8}));

        auto& coords = cache.at("coords");
        auto& backwarp = cache.at("backwarp_grid");
        for (int b = 0; b < B; ++b) {
            for (int y = 0; y < flowH; ++y) {
                for (int x = 0; x < flowW; ++x) {
                    const size_t xIndex = (static_cast<size_t>(b) * 2 * flowH + y) * flowW + x;
                    const size_t yIndex = (static_cast<size_t>(b) * 2 + 1) * flowH * flowW +
                                          static_cast<size_t>(y) * flowW + x;
                    coords.data[xIndex] = static_cast<float>(x);
                    coords.data[yIndex] = static_cast<float>(y);
                    backwarp.data[xIndex] = flowW > 1 ? 2.0f * x / (flowW - 1.0f) - 1.0f : 0.0f;
                    backwarp.data[yIndex] = flowH > 1 ? 2.0f * y / (flowH - 1.0f) - 1.0f : 0.0f;
                }
            }
        }

        auto& pos = cache.at("pos_s16");
        for (int b = 0; b < B * 2; ++b) {
            for (int y = 0; y < h16; ++y) {
                for (int x = 0; x < w16; ++x) {
                    const size_t yIndex = ((static_cast<size_t>(b) * 2) * h16 + y) * w16 + x;
                    const size_t xIndex = ((static_cast<size_t>(b) * 2 + 1) * h16 + y) * w16 + x;
                    pos.data[yIndex] = static_cast<float>(y) - h16 / 2.0f;
                    pos.data[xIndex] = static_cast<float>(x) - w16 / 2.0f;
                }
            }
        }

        auto fill_grid = [](HostTensorData& target, int B, int h, int w) {
            for (int b = 0; b < B; ++b) {
                for (int y = 0; y < h; ++y) {
                    for (int x = 0; x < w; ++x) {
                        const size_t xIndex = (static_cast<size_t>(b) * 2 * h + y) * w + x;
                        const size_t yIndex = (static_cast<size_t>(b) * 2 + 1) * h * w +
                                              static_cast<size_t>(y) * w + x;
                        target.data[xIndex] = static_cast<float>(x);
                        target.data[yIndex] = static_cast<float>(y);
                    }
                }
            }
        };
        fill_grid(cache.at("grid_s16"), B, h16, w16);
        fill_grid(cache.at("grid_s8"), B, h8, w8);

        auto& flatten = cache.at("flatten_grid_s16");
        for (int b = 0; b < B; ++b) {
            for (int y = 0; y < h16; ++y) {
                for (int x = 0; x < w16; ++x) {
                    const int point = y * w16 + x;
                    flatten.data[(static_cast<size_t>(b) * points16 + point) * 2] = static_cast<float>(x);
                    flatten.data[(static_cast<size_t>(b) * points16 + point) * 2 + 1] = static_cast<float>(y);
                }
            }
        }

        auto fill_delta = [](HostTensorData& target, int points) {
            for (int point = 0; point < points; ++point) {
                for (int dy = 0; dy < 9; ++dy) {
                    for (int dx = 0; dx < 9; ++dx) {
                        const size_t index = ((static_cast<size_t>(point) * 9 + dy) * 9 + dx) * 2;
                        target.data[index] = static_cast<float>(dy - 4);
                        target.data[index + 1] = static_cast<float>(dx - 4);
                    }
                }
            }
        };
        fill_delta(cache.at("delta_s16"), B * points16);
        fill_delta(cache.at("delta_s8"), B * points8);
        std::fill(cache.at("radius_emb_s16").data.begin(), cache.at("radius_emb_s16").data.end(), 4.0f);
        std::fill(cache.at("radius_emb_s8").data.begin(), cache.at("radius_emb_s8").data.end(), 4.0f);
        return cache;
    }

    void resize_input(Interpreter* net, Session* session, const std::string& name,
                      const std::vector<int>& shape) {
        auto tensor = net->getSessionInput(session, name.c_str());
        if (!tensor) {
            throw std::runtime_error("MNN input not found: " + name);
        }
        net->resizeTensor(tensor, shape);
    }

    bool persist_opencl_cache(const char* stage) {
        if (!opencl_cache_enabled_ || opencl_cache_file_.empty() || feat_session == nullptr) {
            return true;
        }
        // The feature session and fusion session use the same OpenCL runtime.
        // Taking the cache from the feature interpreter therefore serializes
        // the complete program/tuning set into one tariff-specific file.
        const ErrorCode code = feat_net->updateCacheFile(feat_session);
        if (code != NO_ERROR) {
            std::fprintf(stderr,
                         "[tariff_mnn] opencl_cache update failed stage=%s code=%d file=%s\n",
                         stage, static_cast<int>(code), opencl_cache_file_.c_str());
            return false;
        }
        std::fprintf(stderr, "[tariff_mnn] opencl_cache synchronized stage=%s file=%s\n",
                     stage, opencl_cache_file_.c_str());
        return true;
    }

    void reset_fusion_session() {
        if (fusion_session != nullptr) {
            retired_fusion_sessions_.push_back(fusion_session);
            fusion_session = nullptr;
        }
        fusion_session = fusion_net->createSession(fusion_config_, runtime_info_);
        if (fusion_session == nullptr) {
            throw std::runtime_error("failed to recreate fusion MNN session");
        }
        fusion_input_shapes_.clear();
    }

    void reset_feature_session() {
        if (feat_session != nullptr) {
            feat_net->releaseSession(feat_session);
            feat_session = nullptr;
        }
        feat_session = feat_net->createSession(feat_config_, runtime_info_);
        if (feat_session == nullptr) {
            throw std::runtime_error("failed to recreate feature MNN session");
        }
        feature_shape_initialized_ = false;
        feature_shape_.clear();
    }

    void release_fusion_sessions() {
        if (fusion_session != nullptr) {
            fusion_net->releaseSession(fusion_session);
            fusion_session = nullptr;
        }
        for (auto* session : retired_fusion_sessions_) {
            fusion_net->releaseSession(session);
        }
        retired_fusion_sessions_.clear();
        fusion_input_shapes_.clear();
    }

    void run_feature(void* img0_ptr, void* img1_ptr, int B, int C, int H, int W) {
        const std::vector<int> feature_shape = {B, C, H, W};
        const bool had_feature_shape = feature_shape_initialized_;
        if (had_feature_shape && feature_shape_ != feature_shape) {
            throw std::runtime_error(
                "TariffProcessor is single-resolution; create a new processor for a different shape");
        }
        if (had_feature_shape) {
            // A same-shape feature session can retain state in the dynamic
            // grid/correlation branches.  Recreate it once per frame pair;
            // the session is still reused for every timestep in that pair.
            cached_features.clear();
            reset_feature_session();
        }
        const bool first_feature_shape = !feature_shape_initialized_;

        std::unordered_map<std::string, HostTensorData> cache;
        if (first_feature_shape) {
            resize_input(feat_net.get(), feat_session, "img0", feature_shape);
            resize_input(feat_net.get(), feat_session, "img1", feature_shape);
        }
        if (dynamic_feature_) {
            cache = make_dynamic_cache(B, H, W);
            if (first_feature_shape) {
                for (const auto& item : cache) {
                    resize_input(feat_net.get(), feat_session, item.first, item.second.shape);
                }
            }
        }
        // Re-encode the feature session at the frame-pair boundary even when
        // the spatial shape is unchanged.  Dynamic grid/cache inputs share
        // the OpenCL graph and some correlation branches retain execution
        // local state after a run; re-encoding resets that state without
        // rerunning the feature graph for each timestep.
        feat_net->resizeSession(feat_session);
        if (debug_tensors_) {
            int resize_status = -1;
            feat_net->getSessionInfo(feat_session, Interpreter::RESIZE_STATUS, &resize_status);
            std::fprintf(stderr, "[tariff_mnn] feature resize_status=%d\n", resize_status);
        }
        if (!opencl_cache_feature_saved_) {
            opencl_cache_feature_saved_ = persist_opencl_cache("feature-resize");
        }

        copy_data_to_tensor(feat_net->getSessionInput(feat_session, "img0"), img0_ptr, {B, C, H, W});
        copy_data_to_tensor(feat_net->getSessionInput(feat_session, "img1"), img1_ptr, {B, C, H, W});
        if (dynamic_feature_) {
            for (const auto& item : cache) {
                copy_data_to_tensor(feat_net->getSessionInput(feat_session, item.first.c_str()),
                                    item.second.data.data(), item.second.shape);
            }
        }
        if (first_feature_shape) {
            feature_shape_ = feature_shape;
            feature_shape_initialized_ = true;
        }

        const auto code = feat_net->runSession(feat_session);
        if (debug_tensors_) {
            std::fprintf(stderr, "[tariff_mnn] feature run_code=%d\n", static_cast<int>(code));
        }
        if (code != NO_ERROR) {
            throw std::runtime_error("feature MNN runSession failed");
        }

        // Feature and fusion use separate sessions on one OpenCL runtime.
        // runSession() may only enqueue the feature graph; wait for every
        // output before copying the cached device tensors into fusion.  One
        // output wait is not sufficient on the cold-start path when the
        // independent output branches are still being produced.
        for (const auto& name : feature_outputs_) {
            auto sync_tensor = feat_net->getSessionOutput(feat_session, name.c_str());
            if (sync_tensor) {
                sync_tensor->wait(Tensor::MAP_TENSOR_READ, true);
            }
        }

        cached_features.clear();
        for (const auto& name : feature_outputs_) {
            auto tensor = feat_net->getSessionOutput(feat_session, name.c_str());
            if (!tensor) {
                throw std::runtime_error("feature output not found: " + name);
            }
            cached_features[name] = std::shared_ptr<Tensor>(tensor, [](Tensor*) {});
            if (debug_tensors_) {
                dump_tensor_summary("feature/" + name, tensor);
            }
        }
    }

    std::shared_ptr<Tensor> run_fusion_model(const float* timestep, int B, int H, int W) {
        std::vector<std::string> input_names;
        if (model_type_ == ModelType::NB202) {
            input_names = {"img0", "img1", "flow01", "flow10", "metric0", "metric1",
                           "feat11", "feat12", "feat13", "feat21", "feat22", "feat23", "timestep"};
        } else {
            input_names = {"img0_out", "img1_out", "flow01", "flow10", "metric0", "metric1",
                           "f0", "f1", "timestep"};
        }

        std::unordered_map<std::string, std::vector<int>> fusion_shapes;
        for (const auto& name : input_names) {
            if (name == "timestep") {
                fusion_shapes[name] = {B, 1, H, W};
            } else {
                auto found = cached_features.find(name);
                if (found == cached_features.end() || !found->second) {
                    throw std::runtime_error("fusion input has no feature tensor: " + name);
                }
                fusion_shapes[name] = found->second->shape();
            }
        }

        // OpenCL resizeSession is not re-entrant for repeated runs with the
        // same dynamic shape: resizing before every timestep corrupts the
        // earlier outputs in a multi-timestep request. Resize only when the
        // actual fusion input shapes change.
        if (fusion_shapes != fusion_input_shapes_) {
            // Keep the model's declared input order.  MNN/OpenCL dynamic
            // graph setup is sensitive to the order in which input tensors
            // are resized, even though the shape cache itself is keyed by
            // name.
            for (const auto& name : input_names) {
                if (debug_tensors_) {
                    const auto& shape = fusion_shapes.at(name);
                    std::fprintf(stderr, "[tariff_mnn] fusion resize input=%s shape=", name.c_str());
                    for (int dim : shape) {
                        std::fprintf(stderr, "%d,", dim);
                    }
                    std::fprintf(stderr, "\n");
                }
                resize_input(fusion_net.get(), fusion_session, name, fusion_shapes.at(name));
            }
            if (debug_tensors_) {
                std::fprintf(stderr, "[tariff_mnn] fusion resizeSession begin\n");
            }
            fusion_net->resizeSession(fusion_session);
            if (debug_tensors_) {
                std::fprintf(stderr, "[tariff_mnn] fusion resizeSession done\n");
            }
            fusion_input_shapes_ = fusion_shapes;
            if (debug_tensors_) {
                int resize_status = -1;
                fusion_net->getSessionInfo(fusion_session, Interpreter::RESIZE_STATUS, &resize_status);
                std::fprintf(stderr, "[tariff_mnn] fusion resize_status=%d\n", resize_status);
            }
            if (!opencl_cache_fusion_saved_) {
                opencl_cache_fusion_saved_ = persist_opencl_cache("fusion-resize");
            }
        }

        for (const auto& name : input_names) {
            auto input = fusion_net->getSessionInput(fusion_session, name.c_str());
            if (!input) {
                throw std::runtime_error("fusion MNN input not found: " + name);
            }
            if (name == "timestep") {
                std::shared_ptr<Tensor> ts_tensor(
                    Tensor::create<float>({B, 1, H, W}, const_cast<float*>(timestep), Tensor::CAFFE),
                    Tensor::destroy);
                input->copyFromHostTensor(ts_tensor.get());
            } else {
                input->copyFromHostTensor(cached_features.at(name).get());
            }
        }

        ErrorCode code = static_cast<ErrorCode>(NO_ERROR);
        if (debug_tensors_ || debug_ops_ || debug_softsplat_) {
            TensorCallBackWithInfo before = [this](const std::vector<Tensor*>& tensors, const OperatorInfo* info) {
                if (debug_softsplat_ && info != nullptr && info->type() == "CustomSoftsplat" && tensors.size() >= 2) {
                    capture_tensor_data(tensors[0], debug_softsplat_input_);
                    capture_tensor_data(tensors[1], debug_softsplat_flow_);
                    debug_softsplat_shape_ = tensors[0]->shape();
                }
                return true;
            };
            TensorCallBackWithInfo after = [this](const std::vector<Tensor*>& tensors, const OperatorInfo* info) {
                const bool trace_op = debug_ops_ || (info != nullptr && info->type() == "CustomSoftsplat");
                if (trace_op && info != nullptr) {
                    for (size_t i = 0; i < tensors.size(); ++i) {
                        dump_tensor_summary("fusion/" + info->name() + "/" + std::to_string(i), tensors[i]);
                    }
                }
                if (debug_softsplat_ && info != nullptr && info->type() == "CustomSoftsplat" && !tensors.empty()) {
                    compare_softsplat(tensors[0]);
                }
                return true;
            };
            code = fusion_net->runSessionWithCallBackInfo(fusion_session, before, after, true);
        } else {
            code = fusion_net->runSession(fusion_session);
        }
        if (debug_tensors_) {
            std::fprintf(stderr, "[tariff_mnn] fusion run_code=%d\n", static_cast<int>(code));
        }
        if (code != NO_ERROR) {
            throw std::runtime_error("fusion MNN runSession failed");
        }
        auto output = fusion_net->getSessionOutput(fusion_session, "out");
        if (!output) {
            throw std::runtime_error("fusion output not found: out");
        }
        if (debug_tensors_) {
            dump_tensor_summary("fusion/out", output);
        }
        return copy_tensor_to_host(output);
    }

    std::shared_ptr<Tensor> copy_tensor_to_host(Tensor* source) {
        if (!source) {
            throw std::runtime_error("cannot copy a null MNN tensor");
        }
        auto host = std::shared_ptr<Tensor>(
            Tensor::create<float>(source->shape(), nullptr, Tensor::CAFFE), Tensor::destroy);
        source->wait(Tensor::MAP_TENSOR_READ, true);
        const size_t bytes = static_cast<size_t>(source->elementSize()) * sizeof(float);
        if (source->copyToHostTensor(host.get())) {
            return host;
        }
        if (source->buffer().host != nullptr) {
            std::memcpy(host->host<float>(), source->host<float>(), bytes);
            return host;
        }
        void* mapped = source->map(Tensor::MAP_TENSOR_READ, Tensor::CAFFE);
        if (mapped == nullptr) {
            throw std::runtime_error("copy MNN tensor to host failed");
        }
        std::memcpy(host->host<float>(), mapped, bytes);
        source->unmap(Tensor::MAP_TENSOR_READ, Tensor::CAFFE, mapped);
        return host;
    }

    void dump_tensor_summary(const std::string& name, Tensor* tensor) {
        auto shape = tensor->shape();
        std::shared_ptr<Tensor> host(
            Tensor::create<float>(shape, nullptr, Tensor::CAFFE), Tensor::destroy);
        if (!tensor->copyToHostTensor(host.get())) {
            std::fprintf(stderr, "[tariff_mnn] %s shape=%s host-copy=failed\n", name.c_str(), shape_string(shape).c_str());
            return;
        }
        const float* values = host->host<float>();
        const size_t count = static_cast<size_t>(host->elementSize());
        float minValue = std::numeric_limits<float>::infinity();
        float maxValue = -std::numeric_limits<float>::infinity();
        double sum = 0.0;
        size_t nonzero = 0;
        bool finite = true;
        for (size_t i = 0; i < count; ++i) {
            const float value = values[i];
            finite = finite && std::isfinite(value);
            minValue = value < minValue ? value : minValue;
            maxValue = value > maxValue ? value : maxValue;
            sum += value;
            nonzero += value != 0.0f ? 1 : 0;
        }
        std::fprintf(stderr, "[tariff_mnn] %s shape=%s min=%g max=%g mean=%g nonzero=%zu/%zu finite=%s\n",
                     name.c_str(), shape_string(shape).c_str(), minValue, maxValue,
                     count ? sum / count : 0.0, nonzero, count, finite ? "yes" : "no");
    }

    void capture_tensor_data(Tensor* tensor, std::vector<float>& target) {
        std::shared_ptr<Tensor> host(
            Tensor::create<float>(tensor->shape(), nullptr, Tensor::CAFFE), Tensor::destroy);
        if (!tensor->copyToHostTensor(host.get())) {
            target.clear();
            return;
        }
        target.assign(host->host<float>(), host->host<float>() + tensor->elementSize());
    }

    void compare_softsplat(Tensor* output) {
        if (debug_softsplat_input_.empty() || debug_softsplat_flow_.empty() || debug_softsplat_shape_.size() != 4) {
            std::fprintf(stderr, "[tariff_mnn] softsplat/compare missing host input\n");
            return;
        }
        const auto out_shape = output->shape();
        if (out_shape.size() != 4 || out_shape != debug_softsplat_shape_) {
            std::fprintf(stderr, "[tariff_mnn] softsplat/compare shape mismatch\n");
            return;
        }
        std::shared_ptr<Tensor> host(
            Tensor::create<float>(out_shape, nullptr, Tensor::CAFFE), Tensor::destroy);
        if (!output->copyToHostTensor(host.get())) {
            std::fprintf(stderr, "[tariff_mnn] softsplat/compare output copy failed\n");
            return;
        }
        const int B = out_shape[0];
        const int C = out_shape[1];
        const int H = out_shape[2];
        const int W = out_shape[3];
        const size_t plane = static_cast<size_t>(H) * W;
        std::vector<float> expected(static_cast<size_t>(B) * C * plane, 0.0f);
        const size_t flow_plane = plane;
        for (int n = 0; n < B; ++n) {
            for (int c = 0; c < C; ++c) {
                const size_t nc = static_cast<size_t>(n) * C + c;
                for (int y = 0; y < H; ++y) {
                    for (int x = 0; x < W; ++x) {
                        const size_t hw = static_cast<size_t>(y) * W + x;
                        const size_t flow_base = static_cast<size_t>(n) * 2 * flow_plane + hw;
                        const float flow_x = debug_softsplat_flow_[flow_base];
                        const float flow_y = debug_softsplat_flow_[flow_base + flow_plane];
                        const float out_x = static_cast<float>(x) + flow_x;
                        const float out_y = static_cast<float>(y) + flow_y;
                        const int x0 = static_cast<int>(std::floor(out_x));
                        const int y0 = static_cast<int>(std::floor(out_y));
                        const float dx = out_x - static_cast<float>(x0);
                        const float dy = out_y - static_cast<float>(y0);
                        const float weights[4] = {
                            (1.0f - dx) * (1.0f - dy), dx * (1.0f - dy),
                            (1.0f - dx) * dy, dx * dy};
                        const int xs[4] = {x0, x0 + 1, x0, x0 + 1};
                        const int ys[4] = {y0, y0, y0 + 1, y0 + 1};
                        const float value = debug_softsplat_input_[(nc * H + y) * W + x];
                        for (int k = 0; k < 4; ++k) {
                            if (xs[k] >= 0 && xs[k] < W && ys[k] >= 0 && ys[k] < H) {
                                expected[nc * plane + static_cast<size_t>(ys[k]) * W + xs[k]] += value * weights[k];
                            }
                        }
                    }
                }
            }
        }
        const float* actual = host->host<float>();
        double abs_sum = 0.0;
        double sq_sum = 0.0;
        float max_abs = 0.0f;
        for (size_t i = 0; i < expected.size(); ++i) {
            const float error = actual[i] - expected[i];
            abs_sum += std::fabs(error);
            sq_sum += static_cast<double>(error) * error;
            const float abs_error = std::fabs(error);
            max_abs = max_abs > abs_error ? max_abs : abs_error;
        }
        const double count = static_cast<double>(expected.size());
        std::fprintf(stderr, "[tariff_mnn] softsplat/compare shape=%s mae=%g rmse=%g max=%g\n",
                     shape_string(out_shape).c_str(), abs_sum / count,
                     std::sqrt(sq_sum / count), max_abs);
        debug_softsplat_input_.clear();
        debug_softsplat_flow_.clear();
    }

    static std::string shape_string(const std::vector<int>& shape) {
        std::string result = "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i) result += ",";
            result += std::to_string(shape[i]);
        }
        result += "]";
        return result;
    }

    py::array_t<float> create_numpy_array(Tensor* tensor, int cropH = -1, int cropW = -1) {
        const auto sourceShape = tensor->shape();
        if (sourceShape.size() != 4) {
            throw std::runtime_error("Tariff output must be a 4D BCHW tensor");
        }
        const int sourceH = sourceShape[2];
        const int sourceW = sourceShape[3];
        if (cropH < 0) cropH = sourceH;
        if (cropW < 0) cropW = sourceW;
        if (cropH <= 0 || cropW <= 0 || cropH > sourceH || cropW > sourceW) {
            throw std::runtime_error("Tariff output crop is outside the MNN tensor shape");
        }

        const std::vector<size_t> outputShape = {
            static_cast<size_t>(sourceShape[0]), static_cast<size_t>(sourceShape[1]),
            static_cast<size_t>(cropH), static_cast<size_t>(cropW)};
        py::array_t<float> result(outputShape);
        const float* source = tensor->host<float>();
        float* target = static_cast<float*>(result.mutable_data());
        const size_t sourcePlane = static_cast<size_t>(sourceH) * sourceW;
        const size_t targetPlane = static_cast<size_t>(cropH) * cropW;
        for (int b = 0; b < sourceShape[0]; ++b) {
            for (int c = 0; c < sourceShape[1]; ++c) {
                const float* sourceBase = source +
                    (static_cast<size_t>(b) * sourceShape[1] + c) * sourcePlane;
                float* targetBase = target +
                    (static_cast<size_t>(b) * sourceShape[1] + c) * targetPlane;
                for (int y = 0; y < cropH; ++y) {
                    std::memcpy(targetBase + static_cast<size_t>(y) * cropW,
                                sourceBase + static_cast<size_t>(y) * sourceW,
                                static_cast<size_t>(cropW) * sizeof(float));
                }
            }
        }
        return result;
    }

    py::list concatenate_outputs(const std::vector<py::array_t<float>>& outputs) {
        py::list result_list;
        for (const auto& arr : outputs) {
            result_list.append(arr);
        }
        return result_list;
    }

    void copy_data_to_tensor(Tensor* dest, const void* src, const std::vector<int>& shape) {
        if (!dest) {
            throw std::runtime_error("MNN destination tensor is null");
        }
        auto hostTensor = Tensor::create<float>(shape, const_cast<void*>(src), Tensor::CAFFE);
        dest->copyFromHostTensor(hostTensor);
        Tensor::destroy(hostTensor);
    }
};

PYBIND11_MODULE(tariff_mnn, m) {
    m.def("print_opencl_devices", &print_opencl_devices_impl,
          "Print OpenCL platforms and GPU devices (clinfo-style). Use platform_id/device_id when creating TariffProcessor.");
    py::class_<TariffProcessor>(m, "TariffProcessor")
        .def(py::init<const std::string&, const std::string&, const std::string&, int, int, int>(),
             py::arg("feat_path"), py::arg("fusion_path"), py::arg("model_type") = "nb202",
             py::arg("platform_size") = 1, py::arg("platform_id") = 0, py::arg("device_id") = 0)
        .def("prepare_feature", &TariffProcessor::prepare_feature,
             py::arg("img0"), py::arg("img1"))
        .def("process_cached", &TariffProcessor::process_cached,
             py::arg("timesteps"))
        .def("process", &TariffProcessor::process,
             py::arg("img0"), py::arg("img1"), py::arg("timesteps"));
}
