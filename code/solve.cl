#pragma OPENCL EXTENSION cl_khr_fp64: enable

// No-lookup bilinear interpolation (indices are known, coefficients pre-calculated)
double interp2(double* f, double b_x, double b_q, int jx, int jq, int jz, int je, int Nq, int Nz, int Nh)
{
  double f_0a, f_0b, f_1a, f_1b, f_0, f_1;

  f_0a = V[Ne*(Nz*(Nq*jx + jq) + jz) + je];
  f_0b = V[Ne*(Nz*(Nq*jx + (jq+1)) + jz) + je];
  f_1a = V[Ne*(Nz*(Nq*(jx+1) + jq) + jz) + je];
  f_1b = V[Ne*(Nz*(Nq*(jx+1) + (jq+1)) + jz) + je];

  f_0 = f_0a + b_q*(f_0b - f_0a);
  f_1 = f_1a + b_q*(f_1b - f_1a);

  return f_0 + b_x*(f_1 - f_0);
}

kernel void solve(global double* c_all, global double* V_all,
                  constant double* x_grid, constant double* q_grid, 
		  constant double* w_grid, constant double* e_grid,
                  int Nq, int Nz, int Ne, double bet, double gam)
{

  int ix, iq, jx, jq;
  const int Ns = Ne*Nz;

  double x_next, b_x, b_q, V_j, dU_j;
  double c_endog[Ns], EV_i[Ns], EdU_i[Ns], c[Ns], V[Ns];

  local double c_endog_loc[Nx, Ns];
  local double x_endog_loc[Nx, Ns];

  // Calculate expectations given policy function
  ix = get_global_id(0);
  iq = get_global_id(1);
  // is = get_global_id(2);

  x_next = s_grid[ix]/q_grid[iq];
  jx = floor((Nx-1)*pow((x_next - x_min)/(x_max - x_min), k));
  b_x = (x_grid[jx+1] - x_next)/(x_grid[jx+1] = x_grid[jx]);

  for (int jz = 0; jz < Nz; ++jz)
    {
      q_next = q_bar[jz];
      jq = floor((Nq-1)*(q_next - q_min)/(q_max - q_min));
      b_q = (q_grid[jq+1] - q_next)/(q_grid[jq+1] - q_grid[jq]);
      for (int je = 0; je < Ne; ++je)
        {
          V_j = interp2(V_all, b_x, b_q, jx, jq, jz, je, Nq, Nz, Ne);
          dU_j = pow(interp2(c_all, b_x, b_q, jx, jq, jz, je, Nq, Nz, Ne), -gam);
          for (int iz = 0; iz < Nz; ++iz)
            {
              for (int ie = 0; ie < Ne; ++ie)
                {
                  EV_i[Ne*iz + ie] += P[Ns*(Ne*jz+je) + (Ne*iz+ie)]*V_j;
                  EdU_i[Ne*iz + ie] += P[Ns*(Ne*jz+je) + (Ne*iz+ie)]*dU_j;
                }
            }
        }
    }

  // Not sure if this is actually necessary

  for (int iz = 0; iz < Nz; ++iz)
    {
      for (int ie = 0; ie < Ne; ++ie)
        {
          c_endog_loc[ix][Ne*iz + ie] = pow(bet*EdU_i[Ne*iz + ie]/q_grid[iq], -1/gam);
          x_endog_loc[ix][Ne*iz + ie] = c_endog_loc[ix][Ne*iz + ie] + s_grid[ix] - w_grid[iz]*e_grid[ie];
	  c_all[Ne*(Nz*(Nq*ix + iq) + iz) + ie] = c_endog_loc[ix][Ne*iz + ie];
	  V_all[Ne*(Nz*(Nq*ix + iq) + iz) + ie] = pow(c_endog_loc[ix][Ne*iz + ie], 1-gam)/(1-gam);
        }
    }  

}
