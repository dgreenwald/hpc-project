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
                       constant double* q_grid, constant double* w_grid,
                       constant double* e_grid, constant double* P,
                       constant double* q_bar, constant double* params,
                       global double* done,
                       local double* V_next_loc, local double* dU_next_loc,
                       local double* EV_loc, local double* EdU_loc,
                       local double* x_endog_loc, local double* c_endog_loc)
{

  int gx = get_global_id(0);
  int gq = get_global_id(1);
  int gs = get_global_id(2);
  int gz = gs/NZ;
  int ge = gs - gz;
  int lx = get_local_id(0);
  int ls = get_local_id(2);

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

  if (gx == 0 && gq == 0 && gs == 0)
    printf("NX = %d, NX_LOC = %d, NX_TOT = %d, NX_BLKS = %d, NQ = %d, NZ = %d, NE = %d, NS = %d \n",
           NX, NX_LOC, NX_TOT, NX_BLKS, NQ, NZ, NE, NS);

  if (gx < NX_TOT)
    {
      /*
      // Calculate current step error
      err_i = fabs(V_all[NS*(NQ*gx + gq) + gs] - V_old[NS*(NQ*gx + gq) + gs]);
      if (err_i > 1e-6)
      done[0] = 0;
      else
      done[0] = 1;
      */

      // Update V_old
      V_old[NS*(NQ*gx + gq) + gs] = V_all[NS*(NQ*gx + gq) + gs];

      // Calculate expectations
      x_next = x_grid[gx];
      if (gx < NX-2)
        {
          jx = gx;
          b_x = 0;
        }
      else
        {
          jx = gx-1;
          b_x = 1;
        }

      q_next = q_bar[gz];
      jq = floor((NQ-1)*(q_next - q_min)/(q_max - q_min));
      b_q = (q_next - q_grid[jq])/(q_grid[jq+1] - q_grid[jq]);

      y_i = w_grid[gz]*e_grid[ge];

      V_next_loc[NS*lx + gs] = interp2(V_all, b_x, b_q, jx, jq, gs);
      c_i = interp2(c_all, b_x, b_q, jx, jq, gs);
      dU_next_loc[NS*lx + gs] = pow(interp2(c_all, b_x, b_q, jx, jq, gs), -gam);

      /*
        if (gx < NX_TOT)
        {
        printf("(%d, %d, %d): V_next_loc = %g, dU_next_loc = %g, c = %g \n",
        gx, gq, gs, V_next_loc[NS*lx + gs], dU_next_loc[NS*lx + gs], c_i);
        }
      */

    }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (gx < NX_TOT)
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

  return;
}
