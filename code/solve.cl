#pragma OPENCL EXTENSION cl_khr_fp64: enable

// No-lookup bilinear interpolation (indices are known, coefficients pre-calculated)
double interp2(global double* const f_all, double b_x, double b_q,
               int jx, int jq, int jz, int je)
{
  double f_0a, f_0b, f_1a, f_1b, f_0, f_1;

  f_0a = f_all[NE*(NZ*(NQ*jx + jq) + jz) + je];
  f_0b = f_all[NE*(NZ*(NQ*jx + (jq+1)) + jz) + je];
  f_1a = f_all[NE*(NZ*(NQ*(jx+1) + jq) + jz) + je];
  f_1b = f_all[NE*(NZ*(NQ*(jx+1) + (jq+1)) + jz) + je];

  f_0 = f_0a + b_q*(f_0b - f_0a);
  f_1 = f_1a + b_q*(f_1b - f_1a);

  return f_0 + b_x*(f_1 - f_0);
}

kernel void solve(global double* c_all, global double* V_all,
                  constant double* x_grid, constant double* q_grid,
                  constant double* w_grid, constant double* e_grid,
                  constant double* P, constant double* q_bar,
                  constant double* params)
{

  int gix = get_global_id(0);

  if (gix < NX)
    {
      int giq = get_global_id(1);
      int lix = get_local_id(0);
      int liq = get_local_id(1);
      int x_groups = get_num_groups(0);

      int jx, jq;
      double x_next, q_next, b_x, b_q, V_j, dU_j;
      double EV_i[NS], EdU_i[NS];

      local double c_endog_loc[NX_LOC][NS];
      local double x_endog_loc[NX_LOC][NS];
      local double x_grid_loc[NX_LOC];

      // Unpack parameters
      const double bet = params[0];
      const double gam = params[1];
      const double x_min = params[2];
      const double x_max = params[3];
      const double q_min = params[4];
      const double q_max = params[5];
      const double k = params[6];

      /*
        x_next = x_grid[gix];
        jx = floor((NX-1)*pow((x_next - x_min)/(x_max - x_min), k));
        b_x = (x_next - x_grid[jx])/(x_grid[jx+1] - x_grid[jx]);
      */

      // Initialize local vectors
      for (int is = 0; is < NS; ++is)
        {
          EV_i[is] = 0.0;
          EdU_i[is] = 0.0;
        }

      x_next = x_grid[gix];
      jx = gix;
      b_x = 0;

      for (int jz = 0; jz < NZ; ++jz)
        {
          q_next = q_bar[jz];
          jq = floor((NQ-1)*(q_next - q_min)/(q_max - q_min));
          b_q = (q_next - q_grid[jq])/(q_grid[jq+1] - q_grid[jq]);
          for (int je = 0; je < NE; ++je)
            {
              V_j = interp2(V_all, b_x, b_q, jx, jq, jz, je);
              dU_j = pow(interp2(c_all, b_x, b_q, jx, jq, jz, je), -gam);
              for (int iz = 0; iz < NZ; ++iz)
                {
                  for (int ie = 0; ie < NE; ++ie)
                    {
                      EV_i[NE*iz + ie] += P[NS*(NE*jz+je) + (NE*iz+ie)]*V_j;
                      EdU_i[NE*iz + ie] += P[NS*(NE*jz+je) + (NE*iz+ie)]*dU_j;
                    }
                }
            }
        }

      for (int iz = 0; iz < NZ; ++iz)
        {
          for (int ie = 0; ie < NE; ++ie)
            {
              c_endog_loc[lix][NE*iz + ie] = pow(bet*EdU_i[NE*iz + ie]/q_grid[giq], -1/gam);
              x_endog_loc[lix][NE*iz + ie] = c_endog_loc[lix][NE*iz + ie] + x_grid[gix]*q_grid[giq] - w_grid[iz]*e_grid[ie];
            }
        }

      barrier(CLK_LOCAL_MEM_FENCE);

      for (int iwg = 0; iwg < x_groups; ++iwg)
        {
          x_grid_loc[lix] = x_grid[(NX_LOC-1)*iwg + lix];
          for (int iz = 0; iz < NZ; ++iz)
            for (int ie = 0; ie < NE; ++ie)
              if ((x_endog_loc[0] < x_grid_loc[0]
                   && x_grid_loc[0] < x_endog_loc[NX_LOC-1])
                  || (x_endog_loc[0] < x_grid_loc[NX_LOC-1]
                      && x_grid_loc[NX_LOC-1] < x_endog_loc[NX_LOC-1]))
                {

                }
        }

    }

  return;
}
