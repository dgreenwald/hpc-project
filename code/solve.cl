#pragma OPENCL EXTENSION cl_khr_fp64: enable

kernel void solve()
{

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
	  f_0a = V[Ne*(Nz*(Nq*jx + jq) + jz) + je];
	  f_0b = V[Ne*(Nz*(Nq*jx + (jq+1)) + jz) + je];
	  f_1a = V[Ne*(Nz*(Nq*(jx+1) + jq) + jz) + je];
	  f_1b = V[Ne*(Nz*(Nq*(jx+1) + (jq+1)) + jz) + je];

	  f_0 = f_0a + b_q*(f_0b - f_0a);
	  f_1 = f_1a + b_q*(f_1b - f_1a);
	  
	  f = f_0 + b_x*(f_1 - f_0);
          for (int iz = 0; iz < Nz; ++iz)
            {
              for (int ie = 0; ie < Ne; ++ie)
                {
		  EV[Ne*iz + ie] += P[Nz*Ne*(Ne*jz+je) + (Ne*iz+ie)]*f;
                }
            }
        }
    }

}
