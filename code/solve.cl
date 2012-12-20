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

kernel void solve_iter(global double* c_all, global double* c_old,
                       constant double* x_grid, constant double* q_grid, constant double* y_grid,
                       constant double* P, constant double* q_bar, global int* done,
                       local double* dU_next_loc, local double* EV_loc,
                       local double* x_endog_loc, local double* c_endog_loc)
{

  int gx = get_global_id(0);
  int gq = get_global_id(1);
  int gs = get_global_id(2);
  int lx = get_local_id(0);
  int grp_x = get_group_id(0);
  int Ngrp_x = get_num_groups(0);

  int ix, jx, kx, jq;
  double q_next, b_x, b_q,
    x_i, q_i, EdU_i, y_i, c_min;

  int2 bnds;

  // Initialize local done variable

  local int done_loc;
  if (lx == 0 && gs == 0)
    done_loc = 1;

  q_i = q_grid[gq];

  barrier(CLK_LOCAL_MEM_FENCE);

  /*
    if (gx == 0 && gq == 0 && gs == 0)
    printf("NX = %d, NX_LOC = %d, NX_PAD = %d, NX_TOT = %d, NX_BLKS = %d, NQ = %d, NZ = %d, NS = %d \n",
    NX, NX_LOC, NX_PAD, NX_TOT, NX_BLKS, NQ, NZ, NS);
  */

  /*
    if (gx < NX)
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
      // err_i = fabs(V_all[NS*(NQ*gx + gq) + gs] - V_old[NS*(NQ*gx + gq) + gs]);
      // if (err_i > TOL)
      if (fabs(c_all[NS*(NQ*gx + gq) + gs] - c_old[NS*(NQ*gx + gq) + gs]) > TOL)
        done_loc = 0;

      // Update c_old
      // V_old[NS*(NQ*gx + gq) + gs] = V_all[NS*(NQ*gx + gq) + gs];
      c_old[NS*(NQ*gx + gq) + gs] = c_all[NS*(NQ*gx + gq) + gs];
    }

  if (gx < NX_PAD)
    {
      // Calculate next period values
      jx = (NX_LOC-1)*grp_x + lx;
      // x_next = x_grid[jx];
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
      jq = floor((NQ-1)*(q_next - Q_MIN)/(Q_MAX - Q_MIN));
      b_q = (q_next - q_grid[jq])/(q_grid[jq+1] - q_grid[jq]);

      y_i = y_grid[gs];

      // V_next_loc[NS*lx + gs] = interp2(V_all, b_x, b_q, jx, jq, gs);
      dU_next_loc[NS*lx + gs] = pow(interp2(c_all, b_x, b_q, jx, jq, gs), -GAM);

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
      jx = (NX_LOC-1)*grp_x + lx;

      // EV_loc[NS*lx + gs] = 0.0;
      EdU_i = 0.0;
      for (int is = 0; is < NS; ++is)
        {
          // EV_loc[NS*lx + gs] += P[NS*is + gs]*V_next_loc[NS*lx + is];
          EdU_i += P[NS*is + gs]*dU_next_loc[NS*lx + is];
        }

      c_endog_loc[NS*lx + gs] = pow(BET*EdU_i/q_i, -1/GAM);
      x_endog_loc[NS*lx + gs] = c_endog_loc[NS*lx + gs] + x_grid[jx]*q_i - y_i;

      /*
        if ((get_group_id(0) == 1) && gq == 0 && gs == 0)
        {
        printf("(%d, %d, %d): EdU_loc[NS*lx + gs] = %g \n", jx, gq, gs, EdU_i);
        printf("(%d, %d, %d): EV_loc[NS*lx + gs] = %g \n", jx, gq, gs, EV_loc[NS*lx + gs]);
        printf("(%d, %d, %d): c_endog_loc[NS*lx + gs] = %g \n", jx, gq, gs, c_endog_loc[NS*lx + gs]);
        printf("(%d, %d, %d): x_endog_loc[NS*lx + gs] = %g \n", jx, gq, gs, x_endog_loc[NS*lx + gs]);
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
          // This finds the last relevant local index
          if (grp_x < Ngrp_x - 1)
            kx = NX_LOC-1;
          else
            kx = NX_PAD - (NX_TOT - NX_LOC) - 1;

#if 0
          if (grp_x == Ngrp_x && lx == 0)
            if (x_endog_loc[NS*kx + gs] < X_MAX)
              printf("bad x grid! \n");
#endif

          // kx = min(NX_LOC-1, NX - (NX_LOC-1)*grp_x - 1);

          /*
            if (lx == 0 && gq == 0 && gs == 0)
            {
            printf("lx = %d, group = %d, x_endog_loc[0] = %g, x_endog_loc[kx] = %g \n",
            lx, get_group_id(0), x_endog_loc[gs], x_endog_loc[NS*kx + gs]);
            printf("lx = %d, group = %d, c_endog_loc[0] = %g, c_endog_loc[kx] = %g \n",
            lx, get_group_id(0), c_endog_loc[gs], c_endog_loc[NS*kx + gs]);
            }
          */

          // Boundary case
          if (get_group_id(0) == 0 && x_i < x_endog_loc[gs])
            {
              b_x = (x_i - X_MIN)/(x_endog_loc[gs] - X_MIN);

              c_min = y_i + (1 - q_i)*X_MIN;
              c_all[NS*(NQ*ix + gq) + gs]= max(c_min + b_x*(c_endog_loc[gs] - c_min), 1e-6);
              // EV_i = EV_loc[gs];
              // V_i = pow(c_i, 1-GAM)/(1-GAM) + BET*EV_i;

              /*
                if (gq == 0 && gs == 0)
                {
                printf("(%d, %d, %d): jx = %d, x_i = %g, xlo = %g, xhi = %g \n",
                ix, gq, gs, jx, x_i, X_MIN, x_endog_loc[gs]);
                printf("(%d, %d, %d): jx = %d, c_i = %g, clo = %g, chi = %g \n",
                ix, gq, gs, jx, c_i, y_i, c_endog_loc[gs]);
                }
              */

              // write to global memory
              // c_all[NS*(NQ*ix + gq) + gs] = c_i;
              // V_all[NS*(NQ*ix + gq) + gs] = V_i;

            }
          else if (x_i >= x_endog_loc[gs]
                   && x_i <= x_endog_loc[NS*kx + gs])
            {
              // look up index for interpolation
              bnds = (int2) (0, kx);
              while(bnds.s1 > bnds.s0 + 1)
                {
                  bnds = bisect(x_endog_loc, x_i, bnds, gs);
                }
              jx = bnds.s0;
              b_x = (x_i - x_endog_loc[NS*jx + gs])/(x_endog_loc[NS*(jx+1) + gs] - x_endog_loc[NS*jx + gs]);

              // interpolate to calculate c, EV, then calculate V
              c_all[NS*(NQ*ix + gq) + gs] = max(c_endog_loc[NS*jx + gs] + b_x*(c_endog_loc[NS*(jx+1) + gs] - c_endog_loc[NS*jx + gs]), 1e-6);
              // EV_i = EV_loc[NS*jx + gs] + b_x*(EV_loc[NS*(jx+1) + gs] - EV_loc[NS*jx + gs]);
              // V_i = pow(c_i, 1-GAM)/(1-GAM) + BET*EV_i;

              /*
                if ((get_group_id(0) == 0) && gq == 0 && gs == 0)
                printf("(%d, %d, %d): jx = %d, x_i = %g, c_i = %g, EV_i = %g, V_i = %g \n",
                ix, gq, gs, jx, x_i, c_i, EV_i, V_i);
              */

              /*
                if (gq == 0 && gs == 0)
                {
                printf("(%d, %d, %d): jx = %d, x_i = %g, xlo = %g, xhi = %g \n",
                ix, gq, gs, jx, x_i, x_endog_loc[NS*jx + gs], x_endog_loc[NS*(jx+1) + gs]);
                printf("(%d, %d, %d): jx = %d, c_i = %g, clo = %g, chi = %g \n",
                ix, gq, gs, jx, c_i, c_endog_loc[NS*jx + gs], c_endog_loc[NS*(jx+1) + gs]);
                }
              */

              // write to global memory
              // c_all[NS*(NQ*ix + gq) + gs] = c_i;
              // V_all[NS*(NQ*ix + gq) + gs] = V_i;
            }
        }
    }

  barrier(CLK_LOCAL_MEM_FENCE);

  // Aggregate local flags for convergence into a global flag (so
  // everyone is not writing the global flag at once)
  if (lx == 0 && gs == 0)
    if (done_loc == 0)
      done[0] = 0;

  return;
}

kernel void sim_psums(global double* x_sim, global double* y_sim,
                      global int* z_sim, global int* e_sim,
                      global double* c_all, constant double* x_grid, constant double* q_grid, 
		      global double* a_psums,
                      double q, int tt,
                      local double* a_psums_loc)
{
  double x, y, a, c, b_x, b_q;
  int jx, jq, js;

  int gsim = get_global_id(0);
  int lsim = get_local_id(0);

  if (lsim == 0)
    a_psums[get_group_id(0)] = 0.0;

  barrier(CLK_LOCAL_MEM_FENCE);

  /*
    if (gsim == 0)
    printf("q = %g, tt = %d \n", q, tt);
  */

  x = x_sim[NSIM*tt + gsim];
  y = y_sim[NSIM*tt + gsim];

  jx = (NX - 1)*pow((x - X_MIN)/(X_MAX - X_MIN), KK);
  jq = (NQ - 1)*(q - Q_MIN)/(Q_MAX - Q_MIN);
  js = NE*z_sim[tt] + e_sim[NSIM*tt + gsim];

  /*
    if (tt >= 20)
    printf("worker %d: x = %g, jx = %d, jq = %d, js = %d \n", lsim, x, jx, jq, js);
  */

  b_x = (x - x_grid[jx])/(x_grid[jx+1] - x_grid[jx]);
  b_q = (q - q_grid[jq])/(q_grid[jq+1] - q_grid[jq]);

  /*
    if (tt >= 20)
    printf("worker %d: x = %g, jx = %d, b_x = %g, b_q = %g \n", lsim, x, jx, b_x, b_q);
  */

  c = interp2(c_all, b_x, b_q, jx, jq, js);
  a = x + y - c;

  a_psums_loc[lsim] = a;

  /*
    if (tt >= 20)
    printf("worker %d starting reduction \n", lsim);
  */

  barrier(CLK_LOCAL_MEM_FENCE);
  for (int ii = NSIM_LOC/2; ii > 0; ii >>= 1)
    {
      if (lsim < ii)
        a_psums_loc[lsim] += a_psums_loc[lsim + ii];
      barrier(CLK_LOCAL_MEM_FENCE);
    }

  if (lsim == 0)
    {
      // printf("group %d, a_psum = %g \n", get_group_id(0), a_psums_loc[0]);
      a_psums[get_group_id(0)] = a_psums_loc[0];
    }

  return;
}

kernel void add_psums(global double* psums, global double* sum,
                      local double* psums_loc)
{
  int lsim = get_local_id(0);

  if (lsim < NGRPS_SIM)
    psums_loc[lsim] = psums[lsim];
  else
    psums_loc[lsim] = 0;

  barrier(CLK_LOCAL_MEM_FENCE);
  for (int ii = NSIM_LOC/2; ii > 0; ii >>= 1)
    {
      if (lsim < ii)
        psums_loc[lsim] += psums_loc[lsim + ii];
      barrier(CLK_LOCAL_MEM_FENCE);
    }

  if (lsim == 0)
    {
      *sum = psums_loc[0];
      // printf("sum = %g \n", *sum);
    }

}

kernel void sim_update(global double* x_sim, global double* y_sim,
                       global int* z_sim, global int* e_sim,
                       global double* c_all, constant double* x_grid, constant double* q_grid,
                       double q, int tt)
{
  double x, y, a, c, b_x, b_q;
  int jx, jq, js;

  int gsim = get_global_id(0);
  int lsim = get_local_id(0);

  /*
    if (gsim == 0)
    printf("q = %g, tt = %d, z = %d \n", q, tt, z_sim[tt]);
  */

  x = x_sim[NSIM*tt + gsim];
  y = y_sim[NSIM*tt + gsim];

  jx = (NX - 1)*pow((x - X_MIN)/(X_MAX - X_MIN), KK);
  jq = (NQ - 1)*(q - Q_MIN)/(Q_MAX - Q_MIN);
  js = NE*z_sim[tt] + e_sim[NSIM*tt + gsim];

  b_x = (x - x_grid[jx])/(x_grid[jx+1] - x_grid[jx]);
  b_q = (q - q_grid[jq])/(q_grid[jq+1] - q_grid[jq]);

  c = interp2(c_all, b_x, b_q, jx, jq, js);
  a = max(x + y - c, q*X_MIN);

  x_sim[NSIM*(tt+1) + gsim] = a/q;

  // printf("worker %d: x = %g, y = %g, c = %g, a = %g \n", x, y, c, a);

  return;
}
