//
//  CustomSoftsplatBufExecution.hpp
//  MNN
//
//  Created by MNN on 2021/08/11.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MNN_OPENCL_BUFFER_CLOSED

#ifndef CustomSoftsplatBufExecution_hpp
#define CustomSoftsplatBufExecution_hpp

#include "backend/opencl/execution/image/CommonExecution.hpp"

namespace MNN {
namespace OpenCL {
class CustomSoftsplatBufExecution : public CommonExecution {
public:
    CustomSoftsplatBufExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~CustomSoftsplatBufExecution() = default;

    virtual ErrorCode onEncode(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::vector<uint32_t> mGlobalWorkSize{ 0,0,0,0 };
    std::vector<uint32_t> mLocalWorkSize{ 0,0,0,0 };

    std::string	mKernelName;
    uint32_t mMaxWorkGroupSize;
    OpenCLBackend *mOpenCLBackend;
    bool mNeedUnpackC4 = false;
    std::unique_ptr<Tensor> mTempInput;
    std::unique_ptr<Tensor> mTempOutput;
};
} // namespace OpenCL
} // namespace MNN

#endif // CustomSoftsplatBufExecution_hpp
#endif // MNN_OPENCL_BUFFER_CLOSED
