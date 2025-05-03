//
//  CustomSoftsplatBufExecution.cpp
//  MNN
//
//  Created by MNN on 2021/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#include "backend/opencl/execution/buffer/CustomSoftsplatBufExecution.hpp"

namespace MNN {
namespace OpenCL {

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

    // 清空输出缓冲区
    {
        cl_int error;
        auto outputBuffer = openCLBuffer(output);
        size_t buffer_size = output->elementSize() * sizeof(float);
        
        auto ptrCL = runtime->commandQueue().enqueueMapBuffer(
            outputBuffer, CL_TRUE, CL_MAP_WRITE, 0, buffer_size, nullptr, nullptr, &error
        );
        if(ptrCL != nullptr && error == CL_SUCCESS) {
            ::memset(ptrCL, 0, buffer_size);
        } else {
            MNN_ERROR("Failed to map output buffer for clearing\n");
        }
        runtime->commandQueue().enqueueUnmapMemObject(outputBuffer, ptrCL);
    }

    mUnits.resize(1);
    auto &unit = mUnits[0];
    const int batch = input->buffer().dim[0].extent;
    const int channels = input->buffer().dim[1].extent;
    const int inH = input->buffer().dim[2].extent;
    const int inW = input->buffer().dim[3].extent;
    const int outH = output->buffer().dim[2].extent;
    const int outW = output->buffer().dim[3].extent;

    std::set<std::string> buildOptions;
    buildOptions.insert("-DUSE_ATOMIC_ADD");
    
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
    ret |= unit.kernel->get().setArg(idx++, openCLBuffer(output));
    ret |= unit.kernel->get().setArg(idx++, inH);
    ret |= unit.kernel->get().setArg(idx++, inW);
    ret |= unit.kernel->get().setArg(idx++, outH);
    ret |= unit.kernel->get().setArg(idx++, outW);
    ret |= unit.kernel->get().setArg(idx++, channels);
    ret |= unit.kernel->get().setArg(idx++, batch);
    ret |= unit.kernel->get().setArg(idx++, BorderMode_ZEROS);
    MNN_CHECK_CL_SUCCESS(ret, "setArg CustomSoftsplatBufExecution");

    mLocalWorkSize = localWS3DDefault(mGlobalWorkSize, mMaxWorkGroupSize, runtime, "custom_softsplat_buf", unit.kernel, mOpenCLBackend->getCLTuneLevel()).first;
    
    mOpenCLBackend->recordKernel3d(unit.kernel, mGlobalWorkSize, mLocalWorkSize);
    unit.globalWorkSize = {mGlobalWorkSize[0], mGlobalWorkSize[1], mGlobalWorkSize[2]};
    unit.localWorkSize = {mLocalWorkSize[0], mLocalWorkSize[1], mLocalWorkSize[2]};

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
