//
//  CPUSoftsplat.hpp
//  MNN
//
//  Created by MNN on 2023/06/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUCustomSoftsplat_hpp
#define CPUCustomSoftsplat_hpp

#include "backend/cpu/CPUBackend.hpp"

namespace MNN {

class CPUCustomSoftsplat : public Execution {
public:
    CPUCustomSoftsplat(Backend* backend);
    virtual ~CPUCustomSoftsplat() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;

protected:
    std::shared_ptr<Tensor> mTempCordBuffer;
};

} // namespace MNN

#endif /* CPUCustomSoftsplat_hpp */