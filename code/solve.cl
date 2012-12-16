#pragma OPENCL EXTENSION cl_khr_fp64: enable

// bisection lookup algorithm
int2 bisect(local double* grid, double newval, int2 bnds, int gs)
{
  int mid = (bnds.s0 + bnds.s1)/2;
  if (grid[NS*mid + gs] <= newval)
    return (int2) (mid, bnds.s1);
  else
    return (int2) (bnds.s0, mid);
}

// No-lookup bilinear interpolation (indices are known, coefficients pre-calculated)
double interp2(global double* const f_all, double b_x, double b_q,
               int jx, int jq, int js)
{
  double f_0a, f_0b, f_1a, f_1b, f_0, f_1;

  f_0a = f_all[NS*(NQ*jx + jq) + js];
  f_0b = f_all[NS*(NQ*jx + (jq+1)) + js];
  f_1a = f_all[NS*(NQ*(jx+1) + jq) + js];
  f_1b = f_all[NS*(NQ*(jx+1) + (jq+1)) + js];

  f_0 = f_0a + b_q*(f_0b - f_0a);
  f_1 = f_1a + b_q*(f_1b - f_1a);

  return (f_0 + b_x*(f_1 - f_0));
}

kernel void solve_iter(global double* c_all, global double* V_all,
                       global double* V_old, constant double* x_grid,
                       constant double* q_grid, constant double* y_grid,
                       constant double* P, constant double* q_bar,
                       constant double* params,  global double* done,
                       local double* V_next_loc, local double* dU_next_loc,
                       local double* EV_loc, local double* EdU_loc,
                       local double* x_endog_loc, local double* c_endog_loc)
{

  int gx = get_global_id(0);
  int gq = get_global_id(1);
  int gs = get_global_id(2);
  int lx = get_local_id(0);
  int grp_x = get_group_id(0);

  int ix, jx, jq;
  int written = 0;
  double x_next, q_next, b_x, b_q, dU_next,
    x_i, c_i, EV_i, V_i, EdU_i, y_i, err_i;

  int2 bnds;

  // Unpack parameters

  double bet = params[0];
  double gam = params[1];
  double x_min = params[2];
  double x_max = params[3];
  double q_min = params[4];
  double q_max = params[5];
  double kk = params[6];
  double tol = params[7];

  local double done_loc;
  if (lx == 0 && gs == 0)
    done_loc = 1;

  barrier(CLK_LOCAL_MEM_FENCE);

  /*
    if (gx == 0 && gq == 0 && gs == 0)
    printf("NX = %d, NX_LOC = %d, NX_TOT = %d, NX_BLKS = %d, NQ = %d, NZ = %d, NE = %d, NS = %d \n",
    NX, NX_LOC, NX_TOT, NX_BLKS, NQ, NZ, NE, NS);
  */

  /*
    printf("(%d, %d, %d): c = %g, V = %g \n",
    gx, gq, gs, c_all[NS*(NQ*gx + gq) + gs], V_all[NS*(NQ*gx + gq) + gs]);
  */

  // This section calculates the change over the previous
  // iteration. Each work item evaluates one point.  If the error is
  // too high, a flag is triggered in local memory, which will
  // eventually trigger a flag in global memory, indicating that the
  // algorithm is not done.

  // Afterwards, the "old" function array is updated with the current
  // values, so that it will be the correct "old" matrix for
  // calculating the error on the next step.

  if (gx < NX)
    {
      // Calculate current step error
      err_i = fabs(V_all[NS*(NQ*gx + gq) + gs] - V_old[NS*(NQ*gx + gq) + gs]);
      if (err_i > tol)
        done_loc = 0.0;

      // Update V_old
      V_old[NS*(NQ*gx + gq) + gs] = V_all[NS*(NQ*gx + gq) + gs];
    }

  if (gx < NX_PAD)
    {
      // Calculate expectations
      jx = (NX_LOC-1)*grp_x + lx;
      x_next = x_grid[jx];
      if (gx < NX-2)
        {
          b_x = 0;
        }
      else
        {
          --jx;
          b_x = 1;
        }

      q_next = q_bar[gs/NZ];
      jq = floor((NQ-1)*(q_next - q_min)/(q_max - q_min));
      b_q = (q_next - q_grid[jq])/(q_grid[jq+1] - q_grid[jq]);

      y_i = y_grid[gs];

      V_next_loc[NS*lx + gs] = interp2(V_all, b_x, b_q, jx, jq, gs);
      dU_next_loc[NS*lx + gs] = pow(interp2(c_all, b_x, b_q, jx, jq, gs), -gam);

      /*
        printf("(%d, %d, %d): V_next_loc = %g, dU_next_loc = %g, c = %g \n",
        gx, gq, gs, V_next_loc[NS*lx + gs], dU_next_loc[NS*lx + gs], c_i);
      */

    }

  // This section calculates the optimal consumption policy, c, on an
  // endogenous (i.e. not fixed) grid of x points, both of which are
  // stored in local memory. Each work item is responsible for a
  // single point, and the ends are "padded" so that the intervals are
  // continuous.

  barrier(CLK_LOCAL_MEM_FENCE);

  if (gx < NX_PAD)
    {
      EV_loc[NS*lx + gs] = 0.0;
      EdU_loc[NS*lx + gs] = 0.0;
      for (int is = 0; is < NS; ++is)
        {
          EV_loc[NS*lx + gs] += P[NS*is + gs]*V_next_loc[NS*lx + is];
          EdU_loc[NS*lx + gs] += P[NS*is + gs]*dU_next_loc[NS*lx + is];
        }

      c_endog_loc[NS*lx + gs] = pow(bet*EdU_loc[NS*lx + gs]/q_grid[gq], -1/gam);
      x_endog_loc[NS*lx + gs] = c_endog_loc[NS*lx + gs] + x_grid[gx]*q_grid[gq] - y_i;

      /*
        if ((get_group_id(0) == 0) && gq == 0 && gs == 0)
        {
        printf("(%d, %d, %d): EdU_loc[NS*lx + gs] = %g \n", gx, gq, gs, EdU_loc[NS*lx + gs]);
        printf("(%d, %d, %d): EV_loc[NS*lx + gs] = %g \n", gx, gq, gs, EV_loc[NS*lx + gs]);
        printf("(%d, %d, %d): c_endog_loc[NS*lx + gs] = %g \n", gx, gq, gs, c_endog_loc[NS*lx + gs]);
        printf("(%d, %d, %d): x_endog_loc[NS*lx + gs] = %g \n", gx, gq, gs, x_endog_loc[NS*lx + gs]);
        }
      */

    }

  barrier(CLK_LOCAL_MEM_FENCE);

  /*
    if ((get_group_id(0) == 0) && gq == 0 && gs == 0)
    {
    printf("(%d, %d, %d): x_endog_loc[0] = %g \n", gx, gq, gs, x_endog_loc[0]);
    printf("(%d, %d, %d): x_endog_loc[NX_LOC-1] = %g \n", gx, gq, gs, x_endog_loc[NS*(NX_LOC-1) + gs]);
    }
  */

  // This section loops through blocks of the x grid.

  // Each block gets loaded into memory. Each work item checks if its
  // x point falls in the bounds of its work-group's endogenous grid,
  // if so, it interpolates and writes the value.

  // The one special case is that there will be a number of values for
  // which x falls below the bottom of the lowest x_endog interval. In
  // this case, the agent saves nothing (sets x_next = x_min) and
  // consumes his or her entire income, from which we can obtain the
  // correct consumption value.

  // The code then proceeds to the next block, etc.

  for (int iblk = 0; iblk < NX_BLKS; ++iblk)
    {

      ix = iblk*NX_LOC + lx;
      if (ix < NX)
        {
          x_i = x_grid[ix];

          /*
            if ((get_group_id(0) == 0) && gq == 0 && gs == 0)
            printf("lx = %d, x = %g, x_endog_loc[0] = %g, x_endog_loc[NX_LOC-1] = %g \n",
            lx, x_i, x_endog_loc[gs], x_endog_loc[NS*(NX_LOC-1) + gs]);
          */

          // Boundary case
          if (get_group_id(0) == 0 && x_i < x_endog_loc[gs])
            {
              b_x = (x_i - x_min)/(x_endog_loc[gs] - x_min);

              c_i = y_i + b_x*(c_endog_loc[gs] - y_i);
              EV_i = EV_loc[gs];
              V_i = pow(c_i, 1-gam)/(1-gam) + bet*EV_i;

              // write to global memory
              c_all[NS*(NQ*gx + gq) + gs] = c_i;
              V_all[NS*(NQ*gx + gq) + gs] = V_i;
            }
          else if (x_i >= x_endog_loc[gs]
                   && x_i <= x_endog_loc[NS*(NX_LOC-1) + gs])
            {
              // look up index for interpolation
              bnds = (int2) (0, NX_LOC-1);
              while(bnds.s1 > bnds.s0 + 1)
                {
                  bnds = bisect(x_endog_loc, x_i, bnds, gs);
                }
              jx = bnds.s0;
              b_x = (x_i - x_endog_loc[NS*jx + gs])/(x_endog_loc[NS*(jx+1) + gs] - x_endog_loc[NS*jx + gs]);

              // interpolate to calculate c, EV, then calculate V
              c_i = c_endog_loc[NS*jx + gs] + b_x*(c_endog_loc[NS*(jx+1) + gs] - c_endog_loc[NS*jx + gs]);
              EV_i = EV_loc[NS*jx + gs] + b_x*(EV_loc[NS*(jx+1) + gs] - EV_loc[NS*jx + gs]);
              V_i = pow(c_i, 1-gam)/(1-gam) + bet*EV_i;

              /*
                if ((get_group_id(0) == 0) && gq == 0 && gs == 0)
                printf("(%d, %d, %d): jx = %d, x_i = %g, c_i = %g, EV_i = %g, V_i = %g \n",
                ix, gq, gs, jx, x_i, c_i, EV_i, V_i);
              */

              // write to global memory
              c_all[NS*(NQ*gx + gq) + gs] = c_i;
              V_all[NS*(NQ*gx + gq) + gs] = V_i;
            }
        }
    }

  barrier(CLK_LOCAL_MEM_FENCE);

  // Aggregate local flags for convergence into a global flag (so
  // everyone is not writing the global flag at once)
  if (lx == 0 && gs == 0)
    if (done_loc < 0.5)
      done[0] = 0;

  return;
}
