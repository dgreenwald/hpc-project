#pragma OPENCL EXTENSION cl_khr_fp64: enable

kernel void get_exp(read_only image3d_t c_next,
                  read_only image3d_t V_next,
		  write_only image3d_t EV,
		  write_only image3d_t EdU,
		  float* a_next, float* w_grid, float* qbar,
		  float gam)
{

  constant sampler_t interp3 =
    CLK_NORMALIZED_COORDS_FALSE |
    CLK_ADDRESS_NONE |
    CLK_FILTER_LINEAR;

  int ix, iq, iw, vec_ix;
  int4 write_coords;
  float x_next, q_next, w_next;
  float4 read_coords, y, EV_i, EdU_i, V_j, dU_j;

  ix = get_global_id(0);
  iq = get_global_id(1);
  iw = get_global_id(2);

  vec_ix = Nq*Nw*ix + Nw*iq + iw;

  x_next = a_next[vec_ix] + w_grid[iw];
  q_next = q_bar[iq];

  EV_i = 0.0;
  EdU_i = 0.0;

  for (int jw = 0; jw < Nw; ++jw)
    {
      w_next = w_grid[jw];
      coords = (float4)(x_next, q_next, w_next, 0.0);      

      y = read_imagef(V_next, interp3, coords);
      V_j = y[3]; // information in alpha channel      
      EV_i += Pt[Nw*iw + jw] * Vw; // Pt is P' (or P stored in row-major form)

      y = read_imagef(c_next, interp3, coords);
      dU_j = pow(y[3], -gam); // information in alpha channel      
      EdU_i += Pt[Nw*iw + jw] * dUw; // Pt is P' (or P stored in row-major form)      
    }

  write_coords = (int4)(ix, iq, iw, 0);

  write_imagef(EV, write_coords, EV_i);
  write_image(EdU, write_coords, EdU_i);

}
