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

kernel void solve(global double* c_all, global double* V_all,
                  constant double* x_grid, constant double* q_grid,
                  constant double* w_grid, constant double* e_grid,
                  constant double* P, constant double* q_bar,
                  constant double* params)
{

  int gx = get_global_id(0);

  if (gx < NX)
    {
      int gq = get_global_id(1);
      int gs = get_global_id(2);
      int gz = gs/NZ;
      int ge = gs - gz;
      int lx = get_local_id(0);
      int x_groups = get_num_groups(0);

      int jx, jq;
      double x_next, q_next, b_x, b_q, V_next, dU_next, x_i, c_i, EV_i, V_i;

      local double EV_loc[NX_LOC][NS];
      local double EdU_loc[NX_LOC][NS];
      local double c_endog_loc[NX_LOC][NS];
      local double x_endog_loc[NX_LOC][NS];

      int2 bnds;

      // Unpack parameters
      const double bet = params[0];
      const double gam = params[1];
      const double x_min = params[2];
      const double x_max = params[3];
      const double q_min = params[4];
      const double q_max = params[5];
      const double k = params[6];

      // Initialize local vectors

      EV_loc[lx][gs] = 0.0;
      EdU_loc[lx][gs] = 0.0;

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
          EV_loc[lx][is] += P[NS*gs + is]*V_next;
          EdU_loc[lx][is] += P[NS*gs + is]*dU_next;
        }

      c_endog_loc[lx][gs] = pow(bet*EdU_loc[lx][gs]/q_grid[gq], -1/gam);
      x_endog_loc[lx][gs] = c_endog_loc[lx][gs] + x_grid[gx]*q_grid[gq] - w_grid[gz]*e_grid[ge];

      barrier(CLK_LOCAL_MEM_FENCE);

      for (int iwg = 0; iwg < x_groups; ++iwg)
        {
          x_i = x_grid[(NX_LOC-1)*iwg + lx];
          if (lx < (NX_LOC-1)
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
              V_i = pow(c_i, 1-gam)/(1-gam) + bet*EV_i;

              // write to global memory
              c_all[NS*(NQ*gx + gq) + gs] = c_i;
              V_all[NS*(NQ*gx + gq) + gs] = V_i;

              if (gx == 1)
                printf("(%d, %d, %d): x_i = %g, x_lo = %g, x_hi = %g, c_i = %g, c_lo = %g, c_hi = %g \n",
                       gx, gq, gs,
                       x_i, x_endog_loc[jx][gs], x_endog_loc[jx+1][gs],
                       c_i, c_endog_loc[jx][gs], c_endog_loc[jx+1][gs]);

            }
        }
    }

  return;
}
