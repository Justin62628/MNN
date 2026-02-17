//
//  VulkanCustomSoftsplat.hpp
//  MNN
//
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef VulkanCustomSoftsplat_hpp
#define VulkanCustomSoftsplat_hpp

#include "VulkanBasicExecution.hpp"

namespace MNN {
class VulkanCustomSoftsplat : public VulkanBasicExecution {
public:
    VulkanCustomSoftsplat(const Op* op, Backend* bn);
    virtual ~VulkanCustomSoftsplat();
    ErrorCode onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                       const VulkanCommandPool::Buffer* cmdBuffer) override;

private:
    std::shared_ptr<VulkanBuffer> mParamBuffer;
    const VulkanPipeline* mCustomSoftsplatPipeline;
    std::shared_ptr<VulkanLayout::DescriptorSet> mDescriptorSet;
};
} // namespace MNN

#endif /* VulkanCustomSoftsplat_hpp */
