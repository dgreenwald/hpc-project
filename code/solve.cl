#pragma OPENCL EXTENSION cl_khr_fp64: enable

kernel void solve(read_only image2d_t c_next,
		  read_only image2d_t V_next)
{
  constant sampler_t interp2 = 
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_NONE |
    CLK_FILTER_LINEAR;
}
