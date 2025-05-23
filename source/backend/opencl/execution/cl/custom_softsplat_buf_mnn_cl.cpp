#include "opencl_source_map.hpp" 
namespace MNN { 
#ifndef MNN_OPENCL_BUFFER_CLOSED
const char* custom_softsplat_buf = 
"#ifdef MNN_SUPPORT_FP16\n"
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"#endif\n"
"#pragma OPENCL EXTENSION cl_khr_int32_base_atomics : enable\n"
"#define GLOBAL_SIZE_3_DIMS "" __private const int global_size_dim0,__private const int global_size_dim1,__private const int global_size_dim2,\n"
"#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3) "" if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { "" return; "" }\n"
"enum BorderMode {\n"
" BorderMode_ZEROS=0,\n"
" BorderMode_CLAMP=1,\n"
" BorderMode_REFLECTION=2,\n"
" BorderMode_MIN=BorderMode_ZEROS,\n"
" BorderMode_MAX=BorderMode_REFLECTION\n"
"};\n"
"inline void atomic_add_float(volatile __global float *source,const float operand) { \n"
" union { \n"
" unsigned int intVal; \n"
" float floatVal; \n"
" } newVal; \n"
" union { \n"
" unsigned int intVal; \n"
" float floatVal; \n"
" } prevVal; \n"
" do { \n"
" mem_fence(CLK_GLOBAL_MEM_FENCE);\n"
" prevVal.floatVal=*source; \n"
" newVal.floatVal=prevVal.floatVal+operand; \n"
" } while (atomic_cmpxchg((volatile __global unsigned int *)source,\n"
" prevVal.intVal,newVal.intVal) \n"
" != prevVal.intVal); \n"
"} \n"
"__kernel void custom_softsplat_buf(GLOBAL_SIZE_3_DIMS\n"
" __global const FLOAT *input,\n"
" __global const FLOAT *flow,\n"
" __global FLOAT *output,\n"
" __private const int input_height,\n"
" __private const int input_width,\n"
" __private const int output_height,\n"
" __private const int output_width,\n"
" __private const int channels,\n"
" __private const int batch,\n"
" __private const enum BorderMode paddingMode) {\n"
" const int n=get_global_id(0);\n"
" const int c=get_global_id(1);\n"
" const int hw=get_global_id(2);\n"
" const int h=hw/input_width;\n"
" const int w=hw % input_width;\n"
" DEAL_NON_UNIFORM_DIM3(n,c,hw);\n"
" \n"
" if (c >= channels) return;\n"
" const int flow_offset=n*2*input_height*input_width+h*input_width+w;\n"
" float flow_x=flow[flow_offset];\n"
" float flow_y=flow[flow_offset+input_height*input_width];\n"
" float fltOutputX=(float)w+flow_x;\n"
" float fltOutputY=(float)h+flow_y;\n"
" int intNorthwestX=(int)floor(fltOutputX);\n"
" int intNorthwestY=(int)floor(fltOutputY);\n"
" int intNortheastX=intNorthwestX+1;\n"
" int intNortheastY=intNorthwestY;\n"
" int intSouthwestX=intNorthwestX;\n"
" int intSouthwestY=intNorthwestY+1;\n"
" int intSoutheastX=intNorthwestX+1;\n"
" int intSoutheastY=intNorthwestY+1;\n"
" float fltNorthwest=(intSoutheastX-fltOutputX)*(intSoutheastY-fltOutputY);\n"
" float fltNortheast=(fltOutputX-intSouthwestX)*(intSouthwestY-fltOutputY);\n"
" float fltSouthwest=(intNortheastX-fltOutputX)*(fltOutputY-intNortheastY);\n"
" float fltSoutheast=(fltOutputX-intNorthwestX)*(fltOutputY-intNorthwestY);\n"
" const int input_offset=((n*channels+c)*input_height+h)*input_width+w;\n"
" float val=input[input_offset];\n"
" if (intNorthwestX >= 0 && intNorthwestX<output_width && intNorthwestY >= 0 && intNorthwestY<output_height) {\n"
" int out_offset=((n*channels+c)*output_height+intNorthwestY)*output_width+intNorthwestX;\n"
" atomic_add_float(output+out_offset,val*fltNorthwest);\n"
" }\n"
" if (intNortheastX >= 0 && intNortheastX<output_width && intNortheastY >= 0 && intNortheastY<output_height) {\n"
" int out_offset=((n*channels+c)*output_height+intNortheastY)*output_width+intNortheastX;\n"
" atomic_add_float(output+out_offset,val*fltNortheast);\n"
" }\n"
" if (intSouthwestX >= 0 && intSouthwestX<output_width && intSouthwestY >= 0 && intSouthwestY<output_height) {\n"
" int out_offset=((n*channels+c)*output_height+intSouthwestY)*output_width+intSouthwestX;\n"
" atomic_add_float(output+out_offset,val*fltSouthwest);\n"
" }\n"
" if (intSoutheastX >= 0 && intSoutheastX<output_width && intSoutheastY >= 0 && intSoutheastY<output_height) {\n"
" int out_offset=((n*channels+c)*output_height+intSoutheastY)*output_width+intSoutheastX;\n"
" atomic_add_float(output+out_offset,val*fltSoutheast);\n"
" }\n"
"}\n"
;
#endif
}
