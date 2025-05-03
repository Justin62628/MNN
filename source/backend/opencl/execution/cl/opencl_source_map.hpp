#include <map> 
#include <string> 
#include <vector> 
namespace MNN { 
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* argmax_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* attention_buf;
#endif
extern const char* binary;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* binary_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
extern const char* binary_subgroup_buf;
#endif
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* buffer_convert_buf;
#endif
extern const char* buffer_convert_quant;
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
extern const char* buffer_convert_subgroup_buf;
#endif
#endif
extern const char* buffer_to_image;
extern const char* cast;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* cast_buf;
#endif
extern const char* conv_2d;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* conv_2d_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
extern const char* conv_2d_c16_subgroup_buf;
#endif
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
extern const char* conv_2d_c1_subgroup_buf;
#endif
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* conv_2d_int_buf;
#endif
extern const char* copy_buffer_to_image2d;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* custom_softsplat_buf;
#endif
extern const char* deconv_2d;
extern const char* depthwise_conv2d;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* depthwise_conv2d_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
extern const char* depthwise_conv2d_subgroup_buf;
#endif
#endif
extern const char* depthwise_deconv2d;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* gather_buf;
#endif
extern const char* gemm;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* gemm_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* gemm_conv1x1_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* gemv_conv1x1_buf;
#endif
extern const char* glmem_convert;
extern const char* grid_sample;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* grid_sample_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* groupnorm_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* input_transe_buf;
#endif
extern const char* interp;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* interp_buf;
#endif
extern const char* layernorm;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* layernorm_buf;
#endif
extern const char* loop;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* loop_buf;
#endif
extern const char* matmul;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* matmul_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* matmul_local_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* matmul_params_buf;
#endif
extern const char* nearest;
extern const char* performance;
extern const char* pooling;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* pooling_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
extern const char* pooling_subgroup_buf;
#endif
#endif
extern const char* range;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* range_buf;
#endif
extern const char* raster;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* raster_buf;
#endif
extern const char* reduction;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* reduction_buf;
#endif
extern const char* roi_pooling;
extern const char* scale;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* scale_buf;
#endif
extern const char* select;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* select_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* self_attention_buf;
#endif
extern const char* softmax;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* softmax_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* splitgelu_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* strassen_binary_buf;
#endif
extern const char* unary;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* unary_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
extern const char* unary_subgroup_buf;
#endif
#endif
extern const char* winogradTransformDest2_3_1;
extern const char* winogradTransformDest2_5_1;
extern const char* winogradTransformSource2_3_1;
extern const char* winogradTransformSource2_5_1;
#ifndef MNN_OPENCL_BUFFER_CLOSED
extern const char* winogradTransform_buf;
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
extern const char* winogradTransform_subgroup_buf;
#endif
#endif
const std::map<std::string, const char*> OpenCLProgramMap = 
 { 
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "argmax_buf", argmax_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "attention_buf", attention_buf },
#endif
  { "binary", binary },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "binary_buf", binary_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
  { "binary_subgroup_buf", binary_subgroup_buf },
#endif
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "buffer_convert_buf", buffer_convert_buf },
#endif
  { "buffer_convert_quant", buffer_convert_quant },
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
  { "buffer_convert_subgroup_buf", buffer_convert_subgroup_buf },
#endif
#endif
  { "buffer_to_image", buffer_to_image },
  { "cast", cast },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "cast_buf", cast_buf },
#endif
  { "conv_2d", conv_2d },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "conv_2d_buf", conv_2d_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
  { "conv_2d_c16_subgroup_buf", conv_2d_c16_subgroup_buf },
#endif
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
  { "conv_2d_c1_subgroup_buf", conv_2d_c1_subgroup_buf },
#endif
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "conv_2d_int_buf", conv_2d_int_buf },
#endif
  { "copy_buffer_to_image2d", copy_buffer_to_image2d },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "custom_softsplat_buf", custom_softsplat_buf },
#endif
  { "deconv_2d", deconv_2d },
  { "depthwise_conv2d", depthwise_conv2d },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "depthwise_conv2d_buf", depthwise_conv2d_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
  { "depthwise_conv2d_subgroup_buf", depthwise_conv2d_subgroup_buf },
#endif
#endif
  { "depthwise_deconv2d", depthwise_deconv2d },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "gather_buf", gather_buf },
#endif
  { "gemm", gemm },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "gemm_buf", gemm_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "gemm_conv1x1_buf", gemm_conv1x1_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "gemv_conv1x1_buf", gemv_conv1x1_buf },
#endif
  { "glmem_convert", glmem_convert },
  { "grid_sample", grid_sample },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "grid_sample_buf", grid_sample_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "groupnorm_buf", groupnorm_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "input_transe_buf", input_transe_buf },
#endif
  { "interp", interp },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "interp_buf", interp_buf },
#endif
  { "layernorm", layernorm },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "layernorm_buf", layernorm_buf },
#endif
  { "loop", loop },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "loop_buf", loop_buf },
#endif
  { "matmul", matmul },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "matmul_buf", matmul_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "matmul_local_buf", matmul_local_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "matmul_params_buf", matmul_params_buf },
#endif
  { "nearest", nearest },
  { "performance", performance },
  { "pooling", pooling },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "pooling_buf", pooling_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
  { "pooling_subgroup_buf", pooling_subgroup_buf },
#endif
#endif
  { "range", range },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "range_buf", range_buf },
#endif
  { "raster", raster },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "raster_buf", raster_buf },
#endif
  { "reduction", reduction },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "reduction_buf", reduction_buf },
#endif
  { "roi_pooling", roi_pooling },
  { "scale", scale },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "scale_buf", scale_buf },
#endif
  { "select", select },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "select_buf", select_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "self_attention_buf", self_attention_buf },
#endif
  { "softmax", softmax },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "softmax_buf", softmax_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "splitgelu_buf", splitgelu_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "strassen_binary_buf", strassen_binary_buf },
#endif
  { "unary", unary },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "unary_buf", unary_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
  { "unary_subgroup_buf", unary_subgroup_buf },
#endif
#endif
  { "winogradTransformDest2_3_1", winogradTransformDest2_3_1 },
  { "winogradTransformDest2_5_1", winogradTransformDest2_5_1 },
  { "winogradTransformSource2_3_1", winogradTransformSource2_3_1 },
  { "winogradTransformSource2_5_1", winogradTransformSource2_5_1 },
#ifndef MNN_OPENCL_BUFFER_CLOSED
  { "winogradTransform_buf", winogradTransform_buf },
#endif
#ifndef MNN_OPENCL_BUFFER_CLOSED
#ifdef MNN_SUPPORT_INTEL_SUBGROUP
  { "winogradTransform_subgroup_buf", winogradTransform_subgroup_buf },
#endif
#endif
};
}
