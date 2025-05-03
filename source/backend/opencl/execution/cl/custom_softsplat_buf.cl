#ifdef MNN_SUPPORT_FP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#pragma OPENCL EXTENSION cl_khr_int32_base_atomics : enable

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

enum BorderMode {
  BorderMode_ZEROS = 0,
  BorderMode_CLAMP = 1,
  BorderMode_REFLECTION = 2,
  BorderMode_MIN = BorderMode_ZEROS,
  BorderMode_MAX = BorderMode_REFLECTION
};

inline void atomic_add_float(volatile __global float *source, const float operand) {  
    union {  
        unsigned int intVal;  
        float floatVal;  
    } newVal;  
    union {  
        unsigned int intVal;  
        float floatVal;  
    } prevVal;  
    do {  
        mem_fence(CLK_GLOBAL_MEM_FENCE);
        prevVal.floatVal = *source;  
        newVal.floatVal = prevVal.floatVal + operand;  
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, 
                             prevVal.intVal, newVal.intVal) 
                             != prevVal.intVal);  
}  

__kernel void custom_softsplat_buf(GLOBAL_SIZE_3_DIMS
                                  __global const FLOAT *input,
                                  __global const FLOAT *flow,
                                  __global FLOAT *output,
                                  __private const int input_height,
                                  __private const int input_width,
                                  __private const int output_height,
                                  __private const int output_width,
                                  __private const int channels,
                                  __private const int batch,
                                  __private const enum BorderMode paddingMode) {
    const int n = get_global_id(0);
    const int c = get_global_id(1);
    const int hw = get_global_id(2);
    const int h = hw / input_width;
    const int w = hw % input_width;

    DEAL_NON_UNIFORM_DIM3(n, c, hw);
    
    if (c >= channels) return;

    const int flow_offset = n * 2 * input_height * input_width + h * input_width + w;
    float flow_x = flow[flow_offset];
    float flow_y = flow[flow_offset + input_height * input_width];

    float fltOutputX = (float)w + flow_x;
    float fltOutputY = (float)h + flow_y;

    int intNorthwestX = (int)floor(fltOutputX);
    int intNorthwestY = (int)floor(fltOutputY);
    int intNortheastX = intNorthwestX + 1;
    int intNortheastY = intNorthwestY;
    int intSouthwestX = intNorthwestX;
    int intSouthwestY = intNorthwestY + 1;
    int intSoutheastX = intNorthwestX + 1;
    int intSoutheastY = intNorthwestY + 1;

    float fltNorthwest = (intSoutheastX - fltOutputX) * (intSoutheastY - fltOutputY);
    float fltNortheast = (fltOutputX - intSouthwestX) * (intSouthwestY - fltOutputY);
    float fltSouthwest = (intNortheastX - fltOutputX) * (fltOutputY - intNortheastY);
    float fltSoutheast = (fltOutputX - intNorthwestX) * (fltOutputY - intNorthwestY);

    const int input_offset = ((n * channels + c) * input_height + h) * input_width + w;
    float val = input[input_offset];

    if (intNorthwestX >= 0 && intNorthwestX < output_width && intNorthwestY >= 0 && intNorthwestY < output_height) {
        int out_offset = ((n * channels + c) * output_height + intNorthwestY) * output_width + intNorthwestX;
        atomic_add_float(output + out_offset, val * fltNorthwest);
    }

    if (intNortheastX >= 0 && intNortheastX < output_width && intNortheastY >= 0 && intNortheastY < output_height) {
        int out_offset = ((n * channels + c) * output_height + intNortheastY) * output_width + intNortheastX;
        atomic_add_float(output + out_offset, val * fltNortheast);
    }

    if (intSouthwestX >= 0 && intSouthwestX < output_width && intSouthwestY >= 0 && intSouthwestY < output_height) {
        int out_offset = ((n * channels + c) * output_height + intSouthwestY) * output_width + intSouthwestX;
        atomic_add_float(output + out_offset, val * fltSouthwest);
    }

    if (intSoutheastX >= 0 && intSoutheastX < output_width && intSoutheastY >= 0 && intSoutheastY < output_height) {
        int out_offset = ((n * channels + c) * output_height + intSoutheastY) * output_width + intSoutheastX;
        atomic_add_float(output + out_offset, val * fltSoutheast);
    }
}