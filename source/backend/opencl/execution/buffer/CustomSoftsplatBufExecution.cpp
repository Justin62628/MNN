//
//  CustomSoftsplatBufExecution.cpp
//  MNN
//
//  Created by MNN on 2021/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/CustomSoftsplatBufExecution.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>

namespace MNN {
namespace OpenCL {

void CustomSoftsplatBufExecution::clearAccumulationBuffer() {
    if (!mAccumBuffer || mAccumBufferSize == 0) {
        return;
    }

    if (std::getenv("TARIFF_DEBUG_SOFTSPLAT") != nullptr) {
        std::fprintf(stderr, "[softsplat] clear buffer=%p bytes=%zu\n",
                     static_cast<void*>(mAccumBuffer.get()), mAccumBufferSize);
    }

    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    cl_int error;
    void *ptrCL = runtime->commandQueue().enqueueMapBuffer(
        *mAccumBuffer, CL_TRUE, CL_MAP_WRITE, 0, mAccumBufferSize, nullptr, nullptr, &error);
    if (ptrCL == nullptr || error != CL_SUCCESS) {
        MNN_ERROR("Failed to map CustomSoftsplat accumulation buffer for clearing\n");
        return;
    }
    ::memset(ptrCL, 0, mAccumBufferSize);
    error = runtime->commandQueue().enqueueUnmapMemObject(*mAccumBuffer, ptrCL);
    if (error != CL_SUCCESS) {
        MNN_ERROR("Failed to unmap CustomSoftsplat accumulation buffer after clearing\n");
    }
}

CustomSoftsplatBufExecution::CustomSoftsplatBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : CommonExecution(backend, op) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    const auto dims = inputs[0]->buffer().dimensions;
    mNeedUnpackC4 = TensorUtils::getDescribe(inputs[0])->dimensionFormat == MNN_DATA_FORMAT_NC4HW4;
}

ErrorCode CustomSoftsplatBufExecution::onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto runtime = mOpenCLBackend->getOpenCLRuntime();
    auto input = inputs[0];
    auto flow = inputs[1];
    auto output = outputs[0];

    const int batch = input->buffer().dim[0].extent;
    const int channels = input->buffer().dim[1].extent;
    const int inH = input->buffer().dim[2].extent;
    const int inW = input->buffer().dim[3].extent;
    const int outH = output->buffer().dim[2].extent;
    const int outW = output->buffer().dim[3].extent;
    // Keep the accumulation buffer independent from the output tensor even in
    // Precision_High.  NB202 feeds non-4-aligned channel counts (33/65/97)
    // into the joint feature/fusion graph; writing atomics directly into the
    // graph output makes that path sensitive to output-buffer reuse and
    // backend packing decisions.  The dedicated scalar int32 buffer gives the
    // splat and the final tensor write separate storage in every precision.
    if (std::getenv("TARIFF_DEBUG_TENSORS") != nullptr) {
        MNN_PRINT("[softsplat] input=%dx%dx%dx%d output=%dx%dx%dx%d input_format=%d output_format=%d precision=%d input_type=%d/%d output_type=%d/%d bytes=%g\n",
                  batch, channels, inH, inW, output->buffer().dim[0].extent,
                  output->buffer().dim[1].extent, outH, outW,
                  static_cast<int>(TensorUtils::getDescribe(input)->dimensionFormat),
                  static_cast<int>(TensorUtils::getDescribe(output)->dimensionFormat),
                  static_cast<int>(mOpenCLBackend->getPrecision()),
                  static_cast<int>(input->getType().code), input->getType().bits,
                  static_cast<int>(output->getType().code), output->getType().bits,
                  mOpenCLBackend->getBytes(output));
    }

    // The accumulation buffer is scalar int32 storage.  The previous
    // implementation treated the backend tensor buffer as float in
    // Precision_Low/Normal, which could overwrite past an fp16 allocation
    // and perform 32-bit CAS operations on half elements.  Convert the
    // fixed-point int32 result to the backend tensor type only once.
    // Integer atomics make the reduction independent of the order in which
    // NVIDIA executes the scatter work-items.  That order is not stable for
    // the non-64-aligned NB202 shapes and fp32 CAS accumulation can amplify
    // into visibly different fusion results.
    const float fixedPointScale = 8192.0f;
    auto clearBuffer = [&](cl::Buffer& buffer, size_t bufferSize) {
        cl_int error;
        void *ptrCL = runtime->commandQueue().enqueueMapBuffer(
            buffer, CL_TRUE, CL_MAP_WRITE, 0, bufferSize, nullptr, nullptr, &error);
        if (ptrCL != nullptr && error == CL_SUCCESS) {
            ::memset(ptrCL, 0, bufferSize);
            runtime->commandQueue().enqueueUnmapMemObject(buffer, ptrCL);
        } else {
            MNN_ERROR("Failed to map output buffer for clearing\n");
        }
    };

    const size_t accumSize = static_cast<size_t>(batch) * channels * outH * outW * sizeof(int32_t);
    mAccumBufferSize = accumSize;
    cl_int error = CL_SUCCESS;
    mAccumBuffer.reset(new cl::Buffer(runtime->context(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                      accumSize, nullptr, &error));
    MNN_CHECK_CL_SUCCESS(error, "allocate CustomSoftsplat fixed-point accumulation buffer");
    clearBuffer(*mAccumBuffer, accumSize);

    std::set<std::string> buildOptions;
    buildOptions.insert("-DUSE_ATOMIC_ADD");
    buildOptions.insert("-DUSE_FIXED_POINT_ACCUM");

    mUnits.resize(2);
    auto &unit = mUnits[0];
    unit.kernel = runtime->buildKernel("custom_softsplat_buf", "custom_softsplat_buf", buildOptions, mOpenCLBackend->getPrecision());
    mMaxWorkGroupSize = static_cast<uint32_t>(runtime->getMaxWorkGroupSize(unit.kernel));

    mGlobalWorkSize = {
        static_cast<uint32_t>(batch),
        static_cast<uint32_t>(channels),
        static_cast<uint32_t>(inH * inW)
    };

    uint32_t idx = 0;
    cl_int ret = CL_SUCCESS;
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[0]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[1]);
    ret |= unit.kernel->get().setArg(idx++, mGlobalWorkSize[2]);
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(input));
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(flow));
    ret |= unit.kernel->get().setArg(idx++, *mAccumBuffer);
    ret |= unit.kernel->get().setArg(idx++, inH);
    ret |= unit.kernel->get().setArg(idx++, inW);
    ret |= unit.kernel->get().setArg(idx++, outH);
    ret |= unit.kernel->get().setArg(idx++, outW);
    ret |= unit.kernel->get().setArg(idx++, channels);
    ret |= unit.kernel->get().setArg(idx++, batch);
    ret |= unit.kernel->get().setArg(idx++, BorderMode_ZEROS);
    ret |= unit.kernel->get().setArg(idx++, fixedPointScale);
    MNN_CHECK_CL_SUCCESS(ret, "setArg CustomSoftsplatBufExecution");

    // The forward splat uses atomic accumulation.  Keep its dispatch stable
    // instead of selecting a shape-specific local size through the generic
    // tuner; on the NVIDIA driver the tuned sizes intermittently lose
    // contributions for the 65/97-channel stages at larger resolutions.
    uint32_t fixedLws = mMaxWorkGroupSize > 64 ? 64 : mMaxWorkGroupSize;
    if (fixedLws == 0) {
        fixedLws = 1;
    }
    mLocalWorkSize = {1, 1, fixedLws};
    
    mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};

    auto &convertUnit = mUnits[1];
    convertUnit.kernel = runtime->buildKernel("custom_softsplat_buf", "custom_softsplat_convert",
                                               buildOptions, mOpenCLBackend->getPrecision());
    const std::vector<uint32_t> convertGlobalWorkSize = {
        static_cast<uint32_t>(batch),
        static_cast<uint32_t>(channels),
        static_cast<uint32_t>(outH * outW)
    };
    const uint32_t convertMaxWorkGroupSize =
        static_cast<uint32_t>(runtime->getMaxWorkGroupSize(convertUnit.kernel));
    // This kernel is a pure one-element copy.  Do not run the generic
    // autotuner here: it probes the kernel before the recordable queue is
    // fully established on the NVIDIA OpenCL path and reports
    // CL_INVALID_KERNEL_ARGS.  A fixed 1x1x64 group is valid for the
    // supported tensor sizes and DEAL_NON_UNIFORM_DIM3 handles the tail.
    uint32_t convertLws = convertMaxWorkGroupSize > 64 ? 64 : convertMaxWorkGroupSize;
    if (convertLws == 0) {
        convertLws = 1;
    }
    const std::vector<uint32_t> convertLocalWorkSize = {1, 1, convertLws};

    uint32_t convertIdx = 0;
    cl_int convertRet = CL_SUCCESS;
    convertRet |= convertUnit.kernel->get().setArg(convertIdx++, convertGlobalWorkSize[0]);
    convertRet |= convertUnit.kernel->get().setArg(convertIdx++, convertGlobalWorkSize[1]);
    convertRet |= convertUnit.kernel->get().setArg(convertIdx++, convertGlobalWorkSize[2]);
    convertRet |= convertUnit.kernel->get().setArg(convertIdx++, *mAccumBuffer);
    convertRet |= convertUnit.kernel->get().setArg(convertIdx++, openCLBuffer(output));
    convertRet |= convertUnit.kernel->get().setArg(convertIdx++, fixedPointScale);
    MNN_CHECK_CL_SUCCESS(convertRet, "setArg CustomSoftsplat convert");

    mOpenCLBackend->recordKernel3d(convertUnit.kernel, convertGlobalWorkSize, convertLocalWorkSize);
    convertUnit.globalWorkSize = {convertGlobalWorkSize[0], convertGlobalWorkSize[1], convertGlobalWorkSize[2]};
    convertUnit.localWorkSize = {convertLocalWorkSize[0], convertLocalWorkSize[1], convertLocalWorkSize[2]};

    // 强制同步执行
    // cl::Event event;
    // ret = runtime->commandQueue().enqueueNDRangeKernel(
    //     unit.kernel->get(),
    //     cl::NullRange,
    //     cl::NDRange(mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]),
    //     cl::NDRange(mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]),
    //     nullptr,
    //     &event
    // );
    // event.wait();
    // MNN_CHECK_CL_SUCCESS(ret, "custom_softsplat_buf");
    // TODO: event wait recordKernel3d
    
    // 强制所有命令执行完成
    // runtime->commandQueue().finish();

    return NO_ERROR;
}

ErrorCode CustomSoftsplatBufExecution::onExecute(const std::vector<Tensor *> &inputs,
                                                   const std::vector<Tensor *> &outputs) {
    // onEncode is called only when the session is resized.  A fusion session
    // can then run several timesteps at the same shape, so the private
    // fixed-point int32 accumulation buffer must be cleared before every
    // execution rather than only during graph encoding.
    clearAccumulationBuffer();
    return CommonExecution::onExecute(inputs, outputs);
}

class CustomSoftsplatBufCreator : public OpenCLBackend::Creator {
public:
    virtual ~CustomSoftsplatBufCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                 const MNN::Op *op, Backend *backend) const override {
        for (int i = 0; i < inputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); ++i) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        return new CustomSoftsplatBufExecution(inputs, op, backend);
    }
};

REGISTER_OPENCL_OP_CREATOR(CustomSoftsplatBufCreator, OpType_CustomSoftsplat, BUFFER);

} // namespace OpenCL
} // namespace MNN

#endif // MNN_OPENCL_BUFFER_CLOSED
