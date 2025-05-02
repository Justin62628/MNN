//
//  ShapeCustomSoftsplat.cpp
//  MNN
//
//  Created by MNN on 2021/03/24.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN {
class CustomSoftsplatSizeComputer : public SizeComputer {
    virtual bool onComputeSize(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) const override {
        // inputs[0] is input, inputs[1] is flow, output shape same with input
        MNN_ASSERT(2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        auto &ibInput0 = inputs[0]->buffer();
        auto &ob = outputs[0]->buffer();
        ob.type = ibInput0.type;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(
                inputs[0])->dimensionFormat;

        int input_dim = inputs[0]->buffer().dimensions;
        int flow_dim = inputs[1]->buffer().dimensions;
        MNN_ASSERT((4 == input_dim && 4 == flow_dim));
        if (inputs[0]->buffer().dim[0].extent != inputs[1]->buffer().dim[0].extent) {
            return false;
        }

        ob.dimensions = ibInput0.dimensions;
        ob.dim[0].extent = ibInput0.dim[0].extent;
        ob.dim[1].extent = ibInput0.dim[1].extent;
        ob.dim[2].extent = ibInput0.dim[2].extent;
        ob.dim[3].extent = ibInput0.dim[3].extent;
        return true;
    }

    virtual float onComputeFlops(const MNN::Op *op, const std::vector<Tensor *> &inputs,
                                 const std::vector<Tensor *> &outputs) const override {
        return SizeComputer::onComputeFlops(op, inputs, outputs);
    }
};

REGISTER_SHAPE_INPUTS(CustomSoftsplatSizeComputer, OpType_CustomSoftsplat, {2});

} // namespace MNN
