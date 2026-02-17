//
//  VulkanCustomSoftsplat.cpp
//  MNN
//
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "VulkanCustomSoftsplat.hpp"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
#include "execution/VulkanImageConverter.hpp"

namespace MNN {

struct GpuSoftsplatParam {
    int input_height;
    int input_width;
    int output_height;
    int output_width;
    int channels;
    int batch;
};

VulkanCustomSoftsplat::VulkanCustomSoftsplat(const Op* op, Backend* bn) : VulkanBasicExecution(bn) {
    auto vkBn = static_cast<VulkanBackend*>(bn);
    mCustomSoftsplatPipeline = vkBn->getPipeline("glsl_customSoftsplat_comp", {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    });
    mParamBuffer = std::make_shared<VulkanBuffer>(vkBn->getMemoryPool(), false, sizeof(GpuSoftsplatParam), nullptr,
                                                  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
}

VulkanCustomSoftsplat::~VulkanCustomSoftsplat() {
}

ErrorCode VulkanCustomSoftsplat::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                         const VulkanCommandPool::Buffer* cmdBuffer) {
    auto input = inputs[0];
    auto flow = inputs[1];
    auto output = outputs[0];
    auto vkBn = static_cast<VulkanBackend*>(backend());

    const int batch = input->batch();
    const int channels = input->channel();
    const int inH = input->height();
    const int inW = input->width();
    const int outH = output->height();
    const int outW = output->width();

    int inputBufferSize = sizeof(float);
    for (int i = 0; i < input->dimensions(); i++) {
        inputBufferSize *= input->length(i);
    }
    int flowBufferSize = sizeof(float);
    for (int i = 0; i < flow->dimensions(); i++) {
        flowBufferSize *= flow->length(i);
    }
    int outputBufferSize = sizeof(float);
    for (int i = 0; i < output->dimensions(); i++) {
        outputBufferSize *= output->length(i);
    }

    auto mInputConvert = std::make_shared<VulkanImageConverter>(vkBn);
    auto mFlowConvert = std::make_shared<VulkanImageConverter>(vkBn);
    auto mOutputConvert = std::make_shared<VulkanImageConverter>(vkBn);

    auto mInputBuffer = std::make_shared<VulkanBuffer>(vkBn->getDynamicMemoryPool(), false, inputBufferSize, nullptr,
                                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto mFlowBuffer = std::make_shared<VulkanBuffer>(vkBn->getDynamicMemoryPool(), false, flowBufferSize, nullptr,
                                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto mOutputBuffer = std::make_shared<VulkanBuffer>(vkBn->getDynamicMemoryPool(), false, outputBufferSize, nullptr,
                                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    mInputConvert->encodeTensorToBuffer(input, mInputBuffer->buffer(), mInputBuffer->size(), 0,
                                        VulkanImageConverter::getTensorLinearFormat(input), cmdBuffer);
    mFlowConvert->encodeTensorToBuffer(flow, mFlowBuffer->buffer(), mFlowBuffer->size(), 0,
                                       VulkanImageConverter::getTensorLinearFormat(flow), cmdBuffer);

    {
        auto ptr = mOutputBuffer->map();
        ::memset(ptr, 0, outputBufferSize);
        mOutputBuffer->unmap();
        mOutputBuffer->flush(true, 0, outputBufferSize);
    }

    cmdBuffer->barrierSource(mInputBuffer->buffer(), 0, mInputBuffer->size());
    cmdBuffer->barrierSource(mFlowBuffer->buffer(), 0, mFlowBuffer->size());

    GpuSoftsplatParam* param = reinterpret_cast<GpuSoftsplatParam*>(mParamBuffer->map());
    ::memset(param, 0, sizeof(GpuSoftsplatParam));
    param->input_height = inH;
    param->input_width = inW;
    param->output_height = outH;
    param->output_width = outW;
    param->channels = channels;
    param->batch = batch;
    mParamBuffer->flush(true, 0, sizeof(GpuSoftsplatParam));
    mParamBuffer->unmap();

    mDescriptorSet.reset(mCustomSoftsplatPipeline->createSet());
    mDescriptorSet->writeBuffer(mOutputBuffer->buffer(), 0, mOutputBuffer->size());
    mDescriptorSet->writeBuffer(mInputBuffer->buffer(), 1, mInputBuffer->size());
    mDescriptorSet->writeBuffer(mFlowBuffer->buffer(), 2, mFlowBuffer->size());
    mDescriptorSet->writeBuffer(mParamBuffer->buffer(), 3, mParamBuffer->size());

    mCustomSoftsplatPipeline->bind(cmdBuffer->get(), mDescriptorSet->get());

    int totalElements = batch * channels * inH * inW;
    int groupCount = (totalElements + 255) / 256;
    vkCmdDispatch(cmdBuffer->get(), groupCount, 1, 1);

    cmdBuffer->barrierSource(mOutputBuffer->buffer(), 0, mOutputBuffer->size());
    mOutputConvert->encodeBufferToTensor(mOutputBuffer->buffer(), output, mOutputBuffer->size(), 0,
                                         VulkanImageConverter::getTensorLinearFormat(output), cmdBuffer);

    mInputBuffer->release();
    mFlowBuffer->release();
    mOutputBuffer->release();

    return NO_ERROR;
}

class VulkanCustomSoftsplatCreator : public VulkanBackend::Creator {
public:
    virtual VulkanBasicExecution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                           const MNN::Op* op, Backend* bn) const override {
        for (int i = 0; i < inputs.size(); i++) {
            TensorUtils::setTensorSupportPack(inputs[i], false);
        }
        for (int i = 0; i < outputs.size(); i++) {
            TensorUtils::setTensorSupportPack(outputs[i], false);
        }
        return new VulkanCustomSoftsplat(op, bn);
    }
};

static bool gResistor = []() {
    VulkanBackend::addCreator(OpType_CustomSoftsplat, new VulkanCustomSoftsplatCreator);
    return true;
}();

} // namespace MNN
