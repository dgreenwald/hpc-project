#pragma OPENCL EXTENSION cl_khr_fp64: enable

// bisection lookup algorithm
int2 bisect(local double const grid[NX_LOC][NS], double newval, int2 bnds, int gs)
{
  int mid = (bnds.s0 + bnds.s1)/2;
  if (grid[mid][gs] <= newval)
    return (int2) (mid, bnds.s1);
  else
    return (int2) (bnds.s0, mid);
}

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

kernel void solve_iter(global double* c_all, global double* V_all,
                       global double* V_old, constant double* x_grid,
                       constant double* q_grid, constant double* w_grid,
                       constant double* e_grid, constant double* P,
                       constant double* q_bar, global uint* done)
{

  int gx = get_global_id(0);

  if (gx < NX)
    {
      int gq = get_global_id(1);
      int gs = get_global_id(2);
      int gz = gs/NZ;
      int ge = gs - gz;
      int lx = get_local_id(0);
      int ls = get_local_id(2);
      int grp_x = get_group_id(0);
      int Ngrp_x = get_num_groups(0);
      int grp_q = get_group_id(1);
      int Ngrp_q = get_num_groups(1);

      int jx, jq;
      double x_next, q_next, b_x, b_q, V_next, dU_next,
        x_i, c_i, EV_i, V_i, y_i, err_i;

      local double EV_loc[NX_LOC][NS];
      local double EdU_loc[NX_LOC][NS];
      local double c_endog_loc[NX_LOC][NS];
      local double x_endog_loc[NX_LOC][NS];

      int2 bnds;

      // Initialize local vectors

      EV_loc[lx][gs] = 0.0;
      EdU_loc[lx][gs] = 0.0;
      c_endog_loc[lx][gs] = 0.0;
      x_endog_loc[lx][gs] = 0.0;

      barrier(CLK_LOCAL_MEM_FENCE);

      // Calculate current step error
      err_i = fabs(V_all[NS*(NQ*gx + gq) + gs] - V_old[NS*(NQ*gx + gq) + gs]);
      if (err_i > TOL)
        done[0] = 0;

      // Update V_old array
      V_old[NS*(NQ*gx + gq) + gs] = V_all[NS*(NQ*gx + gq) + gs];

      // Calculate expectations
      x_next = x_grid[gx];
      jx = gx;
      b_x = 0;

      q_next = q_bar[gz];
      jq = floor((NQ-1)*(q_next - Q_MIN)/(Q_MAX - Q_MIN));
      b_q = (q_next - q_grid[jq])/(q_grid[jq+1] - q_grid[jq]);

      y_i = w_grid[gz]*e_grid[ge];

      V_next = interp2(V_all, b_x, b_q, jx, jq, gz, ge);
      dU_next = pow(interp2(c_all, b_x, b_q, jx, jq, gz, ge), -GAM);
      for (int is = 0; is < NS; ++is)
        {
          EV_loc[lx][is] += P[NS*gs + is]*V_next;
          EdU_loc[lx][is] += P[NS*gs + is]*dU_next;
        }

      barrier(CLK_LOCAL_MEM_FENCE);

      c_endog_loc[lx][gs] = pow(BET*EdU_loc[lx][gs]/q_grid[gq], -1/GAM);
      x_endog_loc[lx][gs] = c_endog_loc[lx][gs] + x_grid[gx]*q_grid[gq] - y_i;

      barrier(CLK_LOCAL_MEM_FENCE);

      for (int iwg = 0; iwg < Ngrp_x; ++iwg)
        {

          x_i = x_grid[(NX_LOC-1)*iwg + lx];

          // Boundary case
          if (grp_x == 0 && x_i < x_endog_loc[0][gs])
            {
              b_x = (x_i - X_MIN)/(x_endog_loc[1][gs] - X_MIN);

              c_i = y_i + b_x*(c_endog_loc[1][gs] - y_i);
              EV_i = EV_loc[0][gs];
              V_i = pow(c_i, 1-GAM)/(1-GAM) + BET*EV_i;

              // write to global memory
              c_all[NS*(NQ*gx + gq) + gs] = c_i;
              V_all[NS*(NQ*gx + gq) + gs] = V_i;
            }
          else if (lx < (NX_LOC-1)
                   && x_i >= x_endog_loc[0][gs]
                   && x_i <= x_endog_loc[NX_LOC-1][gs])
            {
              // look up index for interpolation
              bnds = (int2) (0, NX_LOC-1);
              while(bnds.s1 > bnds.s0 + 1)
                {
                  bnds = bisect(x_endog_loc, x_i, bnds, gs);
                }
              jx = bnds.s0;
              b_x = (x_i - x_endog_loc[jx][gs])/(x_endog_loc[jx+1][gs] - x_endog_loc[jx][gs]);

              // interpolate to calculate c, EV, then calculate V
              c_i = c_endog_loc[jx][gs] + b_x*(c_endog_loc[jx+1][gs] - c_endog_loc[jx][gs]);
              EV_i = EV_loc[jx][gs] + b_x*(EV_loc[jx+1][gs] - EV_loc[jx][gs]);
              // V_i = pow(c_i, 1-GAM)/(1-GAM) + BET*EV_i;

              // write to global memory
              c_all[NS*(NQ*gx + gq) + gs] = c_i;
              V_all[NS*(NQ*gx + gq) + gs] = pow(c_i, 1-GAM)/(1-GAM) + BET*EV_i;
            }
        }
    }

  return;
}
