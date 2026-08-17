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
#ifdef USE_FLOAT_ACCUM
                                   __global float *output,
#elif defined(USE_FIXED_POINT_ACCUM)
                                   __global int *output,
#else
                                   __global FLOAT *output,
#endif
                                  __private const int input_height,
                                  __private const int input_width,
                                  __private const int output_height,
                                  __private const int output_width,
                                  __private const int channels,
                                  __private const int batch,
                                  __private const enum BorderMode paddingMode,
                                  __private const float fixed_point_scale) {
    const int n = get_global_id(0);
    const int c = get_global_id(1);
    const int hw = get_global_id(2);
    const int h = hw / input_width;
    const int w = hw % input_width;

    DEAL_NON_UNIFORM_DIM3(n, c, hw);
    if (c >= channels) return;

    const int flow_xy_stride = input_height * input_width;
    const int flow_offset = n * 2 * flow_xy_stride + h * input_width + w;
    float flow_x = flow[flow_offset];
    float flow_y = flow[flow_offset + flow_xy_stride];

    float fltOutputX = (float)w + flow_x;
    float fltOutputY = (float)h + flow_y;
    int x0 = (int)floor(fltOutputX);
    int y0 = (int)floor(fltOutputY);

    /* Bilinear weights for the four corners (NW, NE, SW, SE). */
    float dx = fltOutputX - (float)x0;
    float dy = fltOutputY - (float)y0;
    float w_nw = (1.0f - dx) * (1.0f - dy);
    float w_ne = dx * (1.0f - dy);
    float w_sw = (1.0f - dx) * dy;
    float w_se = dx * dy;

    const int nc = n * channels + c;
    const int input_offset = (nc * input_height + h) * input_width + w;
    float val = (float)input[input_offset];

    const int out_plane = output_height * output_width;
    const int nc_out_base = nc * out_plane;
    int cx[4], cy[4];
    float cw[4];
    cx[0] = x0;     cy[0] = y0;     cw[0] = w_nw;
    cx[1] = x0 + 1; cy[1] = y0;     cw[1] = w_ne;
    cx[2] = x0;     cy[2] = y0 + 1; cw[2] = w_sw;
    cx[3] = x0 + 1; cy[3] = y0 + 1; cw[3] = w_se;

    for (int k = 0; k < 4; k++) {
        int ox = cx[k], oy = cy[k];
        if (ox >= 0 && ox < output_width && oy >= 0 && oy < output_height) {
            int out_offset = nc_out_base + oy * output_width + ox;
#ifdef USE_FIXED_POINT_ACCUM
            atomic_add(output + out_offset, convert_int_rte(val * cw[k] * fixed_point_scale));
#else
            atomic_add_float(output + out_offset, val * cw[k]);
#endif
        }
    }
}

// Convert the fixed-point int32 accumulation buffer back to the backend's
// tensor type. The execution uses this separate write for every precision.
__kernel void custom_softsplat_convert(GLOBAL_SIZE_3_DIMS
#ifdef USE_FIXED_POINT_ACCUM
                                       __global const int *input,
#else
                                       __global const float *input,
#endif
                                       __global FLOAT *output,
                                       __private const float fixed_point_scale) {
    const int n = get_global_id(0);
    const int c = get_global_id(1);
    const int hw = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(n, c, hw);

    const int plane = global_size_dim2;
    const int offset = (n * global_size_dim1 + c) * plane + hw;
#ifdef USE_FIXED_POINT_ACCUM
    output[offset] = (FLOAT)((float)input[offset] / fixed_point_scale);
#else
    output[offset] = (FLOAT)input[offset];
#endif
}
