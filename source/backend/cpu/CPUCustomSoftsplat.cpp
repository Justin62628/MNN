//
//  CPUCustomSoftsplat.cpp
//  MNN
//
//  Created by MNN on 2023/06/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/cpu/CPUCustomSoftsplat.hpp"
#include <math.h>
#include <string.h>
#include "core/Concurrency.h"
#include <algorithm>
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "backend/cpu/compute/ConvOpt.h"
#include "core/Macro.h"
#include <math/Vec.hpp>
using Vec4 = MNN::Math::Vec<float, 4>;
typedef float_t scalar_t;

namespace MNN {

CPUCustomSoftsplat::CPUCustomSoftsplat(Backend* backend) : Execution(backend) {
    // Constructor
}

ErrorCode CPUCustomSoftsplat::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    int numberThread = static_cast<CPUBackend*>(backend())->threadNumber();
    auto core = static_cast<CPUBackend*>(backend())->functions();
    auto input = inputs[0];
    int inD, inH, inW;

    inH = input->buffer().dim[2].extent;
    inW = input->buffer().dim[3].extent;
    mTempCordBuffer.reset(Tensor::createDevice<uint8_t>({1, inH * inW * 2 * core->bytes}));

    auto res = backend()->onAcquireBuffer(mTempCordBuffer.get(), Backend::DYNAMIC);
    if (!res) {
        return OUT_OF_MEMORY;
    }
    backend()->onReleaseBuffer(mTempCordBuffer.get(), Backend::DYNAMIC);
    return NO_ERROR;
}

ErrorCode CPUCustomSoftsplat::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto core = static_cast<CPUBackend*>(backend())->functions();
    
    auto input = inputs[0];
    auto flow = inputs[1];
    auto output = outputs[0];
    
    const int b = input->batch();
    const int c = input->channel();
    const int h = input->height();
    const int w = input->width();

    auto inputPtr = input->host<scalar_t>();
    auto flowPtr = flow->host<scalar_t>();
    auto outputPtr = output->host<scalar_t>();

    ::memset(outputPtr, 0, output->size());

    const int n = b * c * h * w;
    const int threadNumber = static_cast<CPUBackend*>(backend())->threadNumber();
    const int blockSize = threadNumber; // 保持与CUDA相同的BLOCK_DIM
    const int gridSize = (n + blockSize - 1) / (blockSize);
    const int intWH = w * h;

    MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
        const int32_t start = tId * gridSize;
        const int32_t end = std::min(start + gridSize, n);
        
        for (int32_t intIndex = start; intIndex < end; ++intIndex) {
            if (intIndex >= n) break;
            
            // 分解索引
            const int32_t intN = (intIndex / w / h / c) % 1;
            const int32_t intC = (intIndex / w / h) % c;
            const int32_t intY = (intIndex / w ) % h;
            const int32_t intX = intIndex % w;

            int32_t input_i = ((intN)*intWH * c) + ((intC)*intWH) + ((intY)*w) + ((intX) * 1);

            scalar_t fltInput = inputPtr[input_i];
            scalar_t fltOutputX = (scalar_t)(intX) + flowPtr[((intN)*intWH * 2) + ((0) * intWH) + ((intY)*w) + ((intX) * 1)];
            scalar_t fltOutputY = (scalar_t)(intY) + flowPtr[((intN)*intWH * 2) + ((1) * intWH) + ((intY)*w) + ((intX) * 1)];

            int32_t intNorthwestX = (int)(floor(fltOutputX));
            int32_t intNorthwestY = (int)(floor(fltOutputY));
            int32_t intNortheastX = intNorthwestX + 1;
            int32_t intNortheastY = intNorthwestY;
            int32_t intSouthwestX = intNorthwestX;
            int32_t intSouthwestY = intNorthwestY + 1;
            int32_t intSoutheastX = intNorthwestX + 1;
            int32_t intSoutheastY = intNorthwestY + 1;

            scalar_t fltNorthwest = ((scalar_t)(intSoutheastX)-fltOutputX) * ((scalar_t)(intSoutheastY)-fltOutputY);
            scalar_t fltNortheast = (fltOutputX - (scalar_t)(intSouthwestX)) * ((scalar_t)(intSouthwestY)-fltOutputY);
            scalar_t fltSouthwest = ((scalar_t)(intNortheastX)-fltOutputX) * (fltOutputY - (scalar_t)(intNortheastY));
            scalar_t fltSoutheast = (fltOutputX - (scalar_t)(intNorthwestX)) * (fltOutputY - (scalar_t)(intNorthwestY));

            if ((intNorthwestX >= 0) & (intNorthwestX < w) & (intNorthwestY >= 0) & (intNorthwestY < h))
            {
                int32_t output_i = (((intN)*intWH * c) + ((intC)*intWH) + ((intNorthwestY)*w) + ((intNorthwestX) * 1));
                outputPtr[output_i] += (scalar_t)fltInput * fltNorthwest;
            }

            if ((intNortheastX >= 0) & (intNortheastX < w) & (intNortheastY >= 0) & (intNortheastY < h))
            {
                int32_t output_i = (((intN)*intWH * c) + ((intC)*intWH) + ((intNortheastY)*w) + ((intNortheastX) * 1));
                outputPtr[output_i] += (scalar_t)fltInput * fltNortheast;
            }

            if ((intSouthwestX >= 0) & (intSouthwestX < w) & (intSouthwestY >= 0) & (intSouthwestY < h))
            {
                int32_t output_i = (((intN)*intWH * c) + ((intC)*intWH) + ((intSouthwestY)*w) + ((intSouthwestX) * 1));
                outputPtr[output_i] += (scalar_t)fltInput * fltSouthwest;
            }

            if ((intSoutheastX >= 0) & (intSoutheastX < w) & (intSoutheastY >= 0) & (intSoutheastY < h))
            {
                int32_t output_i = (((intN)*intWH * c) + ((intC)*intWH) + ((intSoutheastY)*w) + ((intSoutheastX) * 1));
                outputPtr[output_i] += (scalar_t)fltInput * fltSoutheast;
            }
        }
    }
    MNN_CONCURRENCY_END();
    
    return NO_ERROR;
}

class CPUCustomSoftsplatCreator : public CPUBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
                                const MNN::Op* op, Backend* backend) const override {
        return new CPUCustomSoftsplat(backend);
    }
};

REGISTER_CPU_OP_CREATOR(CPUCustomSoftsplatCreator, OpType_CustomSoftsplat);

} // namespace MNN