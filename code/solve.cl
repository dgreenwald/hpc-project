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

  int gx = get_global_id(0);
  int lx = get_local_id(0);
  int x_groups = get_num_groups(0);

  if (gx < NX)
    {
      int gq = get_global_id(1);
      int gs = get_global_id(2);
      int gz = gs/NZ;
      int ge = gs - gz;

      int jx, jq;
      double x_next, q_next, b_x, b_q, V_next, dU_next;

      local double EV[NX_LOC][NS];
      local double EdU[NX_LOC][NS];
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
        x_next = x_grid[gx];
        jx = floor((NX-1)*pow((x_next - x_min)/(x_max - x_min), k));
        b_x = (x_next - x_grid[jx])/(x_grid[jx+1] - x_grid[jx]);
      */

      // Initialize local vectors
      EV[lx][gs] = 0.0;
      EdU[lx][gs] = 0.0;

      x_next = x_grid[gx];
      jx = gx;
      b_x = 0;

      q_next = q_bar[gz];
      jq = floor((NQ-1)*(q_next - q_min)/(q_max - q_min));
      b_q = (q_next - q_grid[jq])/(q_grid[jq+1] - q_grid[jq]);

      V_next = interp2(V_all, b_x, b_q, jx, jq, gz, ge);
      dU_next = pow(interp2(c_all, b_x, b_q, jx, jq, gz, ge), -gam);
      for (int is = 0; is < NS; ++is)
        {
          EV[lx][is] += P[NS*gs + is]*V_next;
          EdU[lx][is] += P[NS*gs + is]*dU_next;
        }

      c_endog_loc[lx][gs] = pow(bet*EdU[lx][gs]/q_grid[gq], -1/gam);
      x_endog_loc[lx][gs] = c_endog_loc[lx][gs] + x_grid[gx]*q_grid[gq] - w_grid[gz]*e_grid[ge];

      barrier(CLK_LOCAL_MEM_FENCE);

      /*
      for (int iwg = 0; iwg < x_groups; ++iwg)
        {
          x_grid_loc[lx] = x_grid[(NX_LOC-1)*iwg + lx];
        }
      */

    }

  return;
}
