// Order of variables in max_vals (w, q, x, s)

#pragma OPENCL EXTENSION cl_khr_fp64: enable

kernel void solve_endog(read_only image3d_t EV,
                        write_only image3d_t V,
                        global float ***c,
                        float* x_grid, float* max_vals,
                        float gam, float bet)
{

  constant sampler_t interp3 =
    CLK_NORMALIZED_COORDS_TRUE |
    CLK_ADDRESS_NONE |
    CLK_FILTER_LINEAR;

  int is, iq, iw;
  int4 write_coords;
  float s_i, q_i, w_i, EV_i, V_i;
  float4 read_coords, y;

  iw = get_global_id(0);
  iq = get_global_id(1);
  ix = get_global_id(2);

  s_i = (x_grid[ix] - c[iw][iq][ix])/max_vals[3];
  q_i = q_grid[iq]/max_vals[1];
  w_i = w_grid[iw]/max_vals[0];

  read_coords = (float4)(w_i, q_i, s_i, 0.0);
  y = read_imagef(EV, interp3, read_coords);
  EV_i = y.w; // information in alpha channel

  V_i = pow(c[iw][iq][ix], 1-gam)/(1-gam) + bet*EV_i;

  write_coords = (int4) (iw, iq, ix, 0);
  write_imagef(V, write_coords, V_i);

}
