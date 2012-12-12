// Order of variables in max_vals (w, q, x, s)

#pragma OPENCL EXTENSION cl_khr_fp64: enable

kernel void solve_endog(read_only image3d_t c_next,
                        read_only image3d_t V_next,
                        write_only image3d_t EV,
                        global float ***c_endog,
                        global float ***x_endog,
                        float* s_grid, float* q_grid, float* w_grid,
                        float* a_next, float* qbar, float* max_vals,
                        float gam, float bet)
{

  constant sampler_t interp3 =
    CLK_NORMALIZED_COORDS_TRUE |
    CLK_ADDRESS_NONE |
    CLK_FILTER_LINEAR;

  int is, iq, iw, vec_ix;
  int4 write_coords;
  float x_next, q_next, w_next, euler_rhs, EV_i, EdU_i, c_i, x_i, V_j, dU_j;;
  float4 read_coords, y;

  iw = get_global_id(0);
  iq = get_global_id(1);
  is = get_global_id(2);

  vec_ix = Nq*Nw*is + Nw*iq + iw;

  x_next = (a_next[vec_ix] + w_grid[iw])/max_vals[2];
  q_next = q_bar[iq]/max_vals[1];

  EV_i = 0.0;
  EdU_i = 0.0;

  for (int jw = 0; jw < Nw; ++jw)
    {
      w_next = w_grid[jw]/max_vals[0];
      read_coords = (float4)(w_next, q_next, x_next, 0.0);

<<<<<<< HEAD
      y = read_imagef(V_next, interp3, coords); // These coords are incorrect as written
=======
      y = read_imagef(V_next, interp3, read_coords);
>>>>>>> 270d6d5a37b22b9040b09ed8db2cda92e2ca9977
      V_j = y.w; // information in alpha channel
      EV_i += Pt[Nw*iw + jw] * Vw; // Pt is P' (or P stored in row-major form)

      y = read_imagef(c_next, interp3, read_coords);
      dU_j = pow(y.w, -gam); // information in alpha channel
      EdU_i += Pt[Nw*iw + jw] * dUw; // Pt is P' (or P stored in row-major form)
    }

  euler_rhs = bet*EdU_i/q_grid[iq];
  /* c_endog[vec_ix] = pow(euler_rhs, -1/gam); */
  /* x_endog[vec_ix] = c_endog[vec_ix] + s_grid[is]; */
  c_endog[iw][iq][is] = pow(euler_rhs, -1/gam);
  x_endog[iw][iq][is] = c_endog[vec_ix] + s_grid[is];

  write_coords = (int4)(iw, iq, is, 0);
  write_imagef(EV, write_coords, EV_i);

}
