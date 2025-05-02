#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(CustomSoftsplatOnnx);

MNN::OpType CustomSoftsplatOnnx::opType(){
    return MNN::OpType_CustomSoftsplat;
}

MNN::OpParameter CustomSoftsplatOnnx::type(){
    return MNN::OpParameter_NONE;
}

void CustomSoftsplatOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode, OnnxScope* scope){
    return;
}

REGISTER_CONVERTER(CustomSoftsplatOnnx, CustomSoftsplat);
