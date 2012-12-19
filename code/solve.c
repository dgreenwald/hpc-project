#include "timing.h"
#include "cl-helper.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define OUTPUT 1

// Polynomial spaced grid
cl_double* poly_grid(cl_double f_min, cl_double f_max, cl_double k, cl_long N)
{
  cl_double *f = malloc(sizeof(cl_double)*N);
  if (!f) { perror("alloc error in poly_grid"); abort(); }

  for (cl_int ii = 0; ii < N; ++ii)
    {
      f[ii] = f_min + (f_max - f_min) * pow(((cl_double) ii)/((cl_double) (N-1)), 1/k);
    }
  return f;
}

// Transition probability matrix
cl_double* getP(cl_double u_b, cl_double u_g, cl_double dur_b, cl_double dur_g, cl_double udur_b, cl_double udur_g,
                cl_double rat_bg, cl_double rat_gb)
{
  cl_double *P = malloc(sizeof(cl_double)*16);
  if (!P) { perror("alloc error in getP"); abort(); }

  const cl_double pbb = 1 - 1/dur_b;
  const cl_double pgg = 1 - 1/dur_g;

  P[4*1 + 0] = pbb/udur_b;
  P[4*3 + 2] = pgg/udur_g;
  P[4*0 + 0] = pbb - P[4*1 + 0];
  P[4*2 + 2] = pgg - P[4*3 + 2];

  P[4*2 + 0] = rat_bg*(1-pbb)*P[4*2 + 2]/pgg;
  P[4*0 + 2] = rat_gb*(1-pgg)*P[4*0 + 0]/pbb;
  P[4*3 + 0] = 1 - pbb - P[4*2 + 0];
  P[4*1 + 2] = 1 - pgg - P[4*0 + 2];

  P[4*0 + 1] = (pbb*u_b - P[4*0 + 0]*u_b)/(1 - u_b);
  P[4*2 + 1] = ((1-pbb)*u_g - P[4*2 + 0]*u_b)/(1 - u_b);
  P[4*0 + 3] = ((1-pgg)*u_b - P[4*0 + 2]*u_g)/(1 - u_g);
  P[4*2 + 3] = (pgg*u_g - P[4*2 + 2]*u_g)/(1 - u_g);

  P[4*1 + 1] = pbb - P[4*0 + 1];
  P[4*3 + 1] = 1 - pbb - P[4*2 + 1];
  P[4*1 + 3] = 1 - pgg - P[4*0 + 3];
  P[4*3 + 3] = pgg - P[4*2 + 3];

  // Check matrix
  cl_double Psum;
  for (cl_int ii = 0; ii < 4; ++ii)
    {
      Psum = 0;
      for (cl_int jj = 0; jj < 4; ++jj)
        {
          if (P[4*jj + ii] < 0) { perror("P < 0"); abort(); }
          Psum += P[4*jj + ii];
        }
      if (Psum != 1) { perror("Psum != 1"); abort(); }
    }

  return P;

}

// Allocates cl_double buffer
cl_mem alloc_dbuf(cl_context ctx, cl_int N, cl_int read, cl_int write)
{
  cl_mem_flags flag;
  cl_int status;
  if (read == 1 && write == 1)
    flag = CL_MEM_READ_WRITE;
  else if (read == 1)
    flag = CL_MEM_READ_ONLY;
  else if (write == 1)
    flag = CL_MEM_WRITE_ONLY;
  else
    { perror("bad flags in alloc_dbuf"); abort(); }

  cl_mem buf = clCreateBuffer(ctx, flag, N*sizeof(cl_double), 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");
  return buf;
}

// Write to cl_double buffer
void write_dbuf(cl_command_queue queue, cl_mem buf, const cl_double *arr, cl_int N)
{

  CALL_CL_GUARDED(clEnqueueWriteBuffer,
                  (queue, buf, /*blocking*/ CL_FALSE, /*offset*/ 0,
                   N*sizeof(cl_double), arr,
                   0, NULL, NULL));
  return;
}

// Read from cl_double buffer
void read_dbuf(cl_command_queue queue, cl_mem buf, cl_double *arr, cl_int N)
{

  CALL_CL_GUARDED(clEnqueueReadBuffer,
                  (queue, buf, /*blocking*/ CL_FALSE, /*offset*/ 0,
                   N*sizeof(cl_double), arr,
                   0, NULL, NULL));
  return;
}

// Allocates cl_int buffer
cl_mem alloc_ibuf(cl_context ctx, cl_int N, cl_int read, cl_int write)
{
  cl_mem_flags flag;
  cl_int status;
  if (read == 1 && write == 1)
    flag = CL_MEM_READ_WRITE;
  else if (read == 1)
    flag = CL_MEM_READ_ONLY;
  else if (write == 1)
    flag = CL_MEM_WRITE_ONLY;
  else
    { perror("bad flags in alloc_dbuf"); abort(); }

  cl_mem buf = clCreateBuffer(ctx, flag, N*sizeof(cl_int), 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");
  return buf;
}

// Write to cl_int buffer
void write_ibuf(cl_command_queue queue, cl_mem buf, const cl_int *arr, cl_int N)
{

  CALL_CL_GUARDED(clEnqueueWriteBuffer,
                  (queue, buf, /*blocking*/ CL_FALSE, /*offset*/ 0,
                   N*sizeof(cl_int), arr,
                   0, NULL, NULL));
  return;
}

// Read from cl_int buffer
void read_ibuf(cl_command_queue queue, cl_mem buf, cl_int *arr, cl_int N)
{

  CALL_CL_GUARDED(clEnqueueReadBuffer,
                  (queue, buf, /*blocking*/ CL_FALSE, /*offset*/ 0,
                   N*sizeof(cl_int), arr,
                   0, NULL, NULL));
  return;
}

cl_int update_qbar(cl_double* q_bar, cl_double* q_bar_old, const cl_double* q_sim, const cl_int* z_sim, cl_int Nt, cl_int Nburn, cl_double tol)
{

  if (fabs(q_bar_old[0] - q_bar[0]) + fabs(q_bar_old[1] - q_bar[1]) >= tol)
    {
      cl_int N[2] = {0, 0};

      // update old value
      q_bar_old[0] = q_bar[0];
      q_bar_old[1] = q_bar[1];

      // calculate new value
      q_bar[0] = 0;
      q_bar[1] = 0;
      for (cl_int tt = Nburn; tt < Nt; ++tt)
        {
          if (z_sim[tt] == 0)
            {
              ++N[0];
              q_bar[0] += q_sim[tt];
            }
          else if (z_sim[tt] == 1)
            {
              ++N[1];
              q_bar[1] += q_sim[tt];
            }
        }

      q_bar[0] /= (cl_double) N[0];
      q_bar[1] /= (cl_double) N[1];

      return 0;
    }
  else
    {
      return 1;
    }
}

cl_int main(cl_int argc, char **argv)
{

  // Timing Setup

  timestamp_type time1, time2;
  cl_double elapsed;

  // OpenCL Setup

  print_platforms_devices();

  cl_context ctx;
  cl_command_queue queue;
  cl_int status;

  create_context_on("NVIDIA", NULL, 0, &ctx, &queue, 0);
  // create_context_on("Intel", NULL, 0, &ctx, &queue, 0);
  // create_context_on("Advanced", NULL, 0, &ctx, &queue, 0);

  // Define parameters

  const cl_double freq = 4;
  const cl_double tol = 1e-8;
  const cl_double k = 0.4;
  const cl_double alp = 0.36;
  const cl_double gam = 2.0;
  const cl_double bet = pow(0.90, 1/freq);
  // const cl_double q_min = 0.9;
  // const cl_double q_max = 1.1;
  const cl_double q_min = 1/bet - 0.25;
  const cl_double q_max = 1/bet + 0.25;
  const cl_double x_min = -0.5;
  const cl_double x_max = 10;
  const cl_int Nx = 2000;
  const cl_int Nx_loc = 128;
  // const cl_int Nx = 9;
  // const cl_int Nx_loc = 8;
  const cl_int Nx_pad = Nx + (Nx-2)/(Nx_loc-1);
  const cl_int Nx_tot = Nx_loc*((Nx_pad-1)/Nx_loc + 1);
  const cl_int Nx_blks = (Nx-1)/Nx_loc + 1;
  const cl_int Nq = 1000;
  const cl_int Nz = 2;
  const cl_int Ne = 2;
  const cl_int Ns = Nz*Ne;
  const cl_int Npar = 8;
  const cl_int Nsim = 5120;
  const cl_int Nsim_loc = 256;
  const cl_int Ngrps_sim = (Nsim - 1)/Nsim_loc + 1;
  const cl_int Nt = 1200;
  const cl_int Nburn = 200;

  cl_double* x_grid = poly_grid(x_min, x_max, k, Nx);
  cl_double* q_grid = poly_grid(q_min, q_max, 1.0, Nq); // 1.0 for even grid

  const cl_double z_grid[2] = {0.99, 1.01};
  const cl_double e_grid[2] = {0.3, 1.0};

  const cl_double u_b = 0.1;
  const cl_double u_g = 0.04;
  const cl_double dur_b = 8.0;
  const cl_double dur_g = 8.0;
  const cl_double udur_b = 2.5;
  const cl_double udur_g = 1.5;
  const cl_double rat_bg = 0.75;
  const cl_double rat_gb = 1.25;

  cl_double *P = getP(u_b, u_g, dur_b, dur_g, udur_b, udur_g, rat_bg, rat_gb);

  cl_double Pz[2][2];
  Pz[0][0] = 1 - 1/dur_b;
  Pz[0][1] = 1/dur_b;
  Pz[1][1] = 1 - 1/dur_g;
  Pz[1][0] = 1/dur_g;

  /* // Outputting P matrix
     for (cl_int ii = 0; ii < Ns; ++ii)
     {
     for (cl_int jj = 0; jj < Ns; ++jj)
     printf("P[%d, %d] = %g  ", ii, jj, P[4*jj + ii]);
     printf("\n");
     }
  */

  cl_double w_grid[2];
  w_grid[0] = z_grid[0]*pow(1 - u_b, alp);
  w_grid[1] = z_grid[1]*pow(1 - u_g, alp);

  // Allocate host solution memory
  cl_double *c_all = malloc(sizeof(cl_double) * Nx * Nq * Ns);
  if (!c_all) { perror("alloc c_all"); abort(); }

  cl_double *c_init = malloc(sizeof(cl_double) * Nx * Nq * Ns);
  if (!c_init) { perror("alloc c_init"); abort(); }

  /*
    cl_double *V_all = malloc(sizeof(cl_double) * Nx * Nq * Ns);
    if (!V_all) { perror("alloc V_all"); abort(); }

    cl_double *V_init = malloc(sizeof(cl_double) * Nx * Nq * Ns);
    if (!V_init) { perror("alloc V_init"); abort(); }
  */

  cl_double *y_grid = malloc(sizeof(cl_double) * Ns);
  if (!y_grid) { perror("alloc y_grid"); abort(); }

  cl_double *q_bar = malloc(sizeof(cl_double) * Ns);
  if (!q_bar) { perror("alloc q_bar"); abort(); }

  cl_double *q_bar_old = malloc(sizeof(cl_double) * Ns);
  if (!q_bar_old) { perror("alloc q_bar_old"); abort(); }

  cl_double* params = malloc(Npar*sizeof(cl_double));
  if (!params) { perror("alloc params"); abort(); }

  cl_int *done_start = malloc(sizeof(cl_int));
  if (!done_start) { perror("alloc done_start"); abort(); }

  cl_int *done_end = malloc(sizeof(cl_int));
  if (!done_end) { perror("alloc done_end"); abort(); }

  // Initialize solution matrices
  for (cl_int ix = 0; ix < Nx; ++ix)
    for (cl_int iq = 0; iq < Nq; ++iq)
      for (cl_int iz = 0; iz < Nz; ++iz)
        for (cl_int ie = 0; ie < Ne; ++ie)
          {
            c_all[Ne*(Nz*(Nq*ix + iq) + iz) + ie] = x_grid[ix] + w_grid[iz]*e_grid[ie] - x_min; // Zero bond solution
            c_init[Ne*(Nz*(Nq*ix + iq) + iz) + ie] = -1.0; // (bad) initial comparison
            /*
              V_all[Ne*(Nz*(Nq*ix + iq) + iz) + ie] = pow(c_all[Ne*(Nz*(Nq*ix + iq) + iz) + ie], 1-gam)/(1-gam);
              V_init[Ne*(Nz*(Nq*ix + iq) + iz) + ie] = -1e+10;
            */
          }

  q_bar[0] = 1/z_grid[0];
  q_bar[1] = 1/z_grid[1];

  q_bar_old[0] = -1e+10;
  q_bar_old[1] = -1e+10;

  for (cl_int iz = 0; iz < Nz; ++iz)
    for (cl_int ie = 0; ie < Ne; ++ie)
      {
        y_grid[Ne*iz + ie] = w_grid[iz]*e_grid[ie];
        // printf("y_grid[%d] = %g \n", Ne*iz + ie, y_grid[Ne*iz + ie]);
      }

  params[0] = bet;
  params[1] = gam;
  params[2] = x_min;
  params[3] = x_max;
  params[4] = q_min;
  params[5] = q_max;
  params[6] = k;
  params[7] = tol;

  *done_start = 1;
  *done_end = 0;

  /*
    printf("before kernel \n");
    for (cl_int ii = 0; ii < Nx*Nq*Ns; ++ii)
    {
    printf("%d: c = %g, V = %g \n", ii, c_all[ii], V_all[ii]);
    }
  */

  // Allocate device solution memory
  cl_mem c_buf = alloc_dbuf(ctx, Nx*Nq*Ns, 1, 1);
  cl_mem c_old_buf = alloc_dbuf(ctx, Nx*Nq*Ns, 1, 1);
  /*
    cl_mem V_buf = alloc_dbuf(ctx, Nx*Nq*Ns, 1, 1);
    cl_mem V_old_buf = alloc_dbuf(ctx, Nx*Nq*Ns, 1, 1);
  */
  cl_mem x_buf = alloc_dbuf(ctx, Nx, 1, 0);
  cl_mem q_buf = alloc_dbuf(ctx, Nq, 1, 0);
  cl_mem y_buf = alloc_dbuf(ctx, Ns, 1, 0);
  cl_mem P_buf = alloc_dbuf(ctx, Ns*Ns, 1, 0);
  cl_mem q_bar_buf = alloc_dbuf(ctx, Nz, 1, 0);
  cl_mem params_buf = alloc_dbuf(ctx, Npar, 1, 0);
  cl_mem done_buf = alloc_ibuf(ctx, 1, 1, 1);

  // Transfer solution buffers to device
  write_dbuf(queue, c_buf, c_all, Nx*Nq*Ns);
  // write_dbuf(queue, V_buf, V_all, Nx*Nq*Ns);
  write_dbuf(queue, x_buf, x_grid, Nx);
  write_dbuf(queue, q_buf, q_grid, Nq);
  write_dbuf(queue, y_buf, y_grid, Ns);
  write_dbuf(queue, P_buf, P, Ns*Ns);
  write_dbuf(queue, q_bar_buf, q_bar, Nz);
  write_dbuf(queue, params_buf, params, Npar);


  // Allocate host simulation memory
  cl_double *x_sim = malloc(sizeof(cl_double) * Nsim * Nt);
  if (!x_sim) { perror("alloc x_sim"); abort(); }

  cl_double *y_sim = malloc(sizeof(cl_double) * Nsim * Nt);
  if (!y_sim) { perror("alloc y_sim"); abort(); }

  cl_int *z_sim = malloc(sizeof(cl_int) * Nt);
  if (!z_sim) { perror("alloc z_sim"); abort(); }

  cl_int *e_sim = malloc(sizeof(cl_int) * Nsim * Nt);
  if (!e_sim) { perror("alloc e_sim"); abort(); }

  cl_double *a_net = malloc(sizeof(cl_double));
  if (!a_net) { perror("alloc a_net"); abort(); }

  cl_double *q_sim = malloc(sizeof(cl_double) * Nt);
  if (!q_sim) { perror("alloc q_sim"); abort(); }

  // Allocate device simulation memory
  cl_mem x_sim_buf = alloc_dbuf(ctx, Nsim*Nt, 1, 1);
  cl_mem y_sim_buf = alloc_dbuf(ctx, Nsim*Nt, 1, 0);
  cl_mem a_psums_buf = alloc_dbuf(ctx, Ngrps_sim, 1, 1);
  cl_mem a_net_buf = alloc_dbuf(ctx, 1, 1, 1);

  cl_mem z_sim_buf = alloc_ibuf(ctx, Nt, 1, 0);
  cl_mem e_sim_buf = alloc_ibuf(ctx, Nsim*Nt, 1, 0);

  // Initialize simulation arrays
  cl_double draw;
  // Draw recession/expansion states
  for (cl_int tt = 0; tt < Nt; ++tt)
    {
      if (tt > 0)
        {
          draw = ((cl_double) rand())/((cl_double) RAND_MAX);
          if (draw <= Pz[z_sim[tt-1]][0])
            z_sim[tt] = 0;
          else
            z_sim[tt] = 1;
          // printf("z_sim[%d] = %d \n", tt, z_sim[tt]);
          // printf("t = %d, draw = %g, P[.][0] = %g, z_sim = %d \n", tt, draw, Pz[z_sim[tt-1]][0], z_sim[tt]);
        }
      else
        {
          z_sim[tt] = 1;
        }
    }

  // Draw employment states for each individual
  cl_double Pe[Ne][Ne][Nz][Nz];
  cl_double Pe_sum;
  for (cl_int iz = 0; iz < Nz; ++iz)
    for (cl_int jz = 0; jz < Nz; ++jz)
      for (cl_int ie = 0; ie < Ne; ++ie)
        {
          Pe_sum = 0.0;
          for (cl_int je = 0; je < Ne; ++je)
            {
              Pe[ie][je][iz][jz] = P[Ns*(Ne*jz + je) + Ne*iz + ie]/Pz[iz][jz];
              Pe_sum += Pe[ie][je][iz][jz];
            }
          if (Pe_sum != 1.0)
            { perror("bad probability matrix \n"); abort(); }
        }

  for (cl_int tt = 0; tt < Nt; ++tt)
    {
      if (tt > 0)
        {
          for (cl_int ii = 0; ii < Nsim; ++ii)
            {
              draw = ((cl_double) rand())/((cl_double) RAND_MAX);
              if (draw <= Pe[e_sim[Nsim*(tt-1) + ii]][0][z_sim[tt-1]][z_sim[tt]])
                {
                  e_sim[Nsim*tt + ii] = 0;
                }
              else
                {
                  e_sim[Nsim*tt + ii] = 1;
                }
              y_sim[Nsim*tt + ii] = y_grid[Ne*z_sim[tt] + e_sim[Nsim*tt + ii]];
              /*
                printf("t = %d, i = %d, draw = %g, e_sim = %d, y_sim = %g \n",
                tt, ii, draw, e_sim[Nsim*tt + ii], y_sim[Nsim*tt + ii]);
              */
            }
        }
      else
        {
          for (cl_int ii = 0; ii < Nsim; ++ii)
            {
              e_sim[Nsim*tt + ii] = 1;
              y_sim[Nsim*tt + ii] = y_grid[Ne*z_sim[tt] + e_sim[Nsim*tt + ii]];
              /*
                printf("t = %d, i = %d, draw = %g, e_sim = %d, y_sim = %g \n",
                tt, ii, draw, e_sim[Nsim*tt + ii], y_sim[Nsim*tt + ii]);
              */
            }
        }
    }

  for (cl_int ii = 0; ii < Nsim; ++ii)
    {
      x_sim[ii] = 0.0;
    }

  // Transfer simulation buffers to device
  write_dbuf(queue, x_sim_buf, x_sim, Nsim*Nt);
  write_dbuf(queue, y_sim_buf, y_sim, Nsim*Nt);
  write_ibuf(queue, z_sim_buf, z_sim, Nt);
  write_ibuf(queue, e_sim_buf, e_sim, Nsim*Nt);

  // CALL_CL_GUARDED(clFinish, (queue));

  // Solution setup
  cl_int iter;
  char* knl_text = read_file("solve.cl");
  char buildOptions[200];
  sprintf(buildOptions, "-DNX=%d -DNX_LOC=%d -DNX_PAD=%d -DNX_TOT=%d -DNX_BLKS=%d -DNQ=%d -DNZ=%d -DNE=%d -DNS=%d"
          " -DNSIM=%d -DNSIM_LOC=%d -DNT=%d -DNGRPS_SIM=%d -DBET_TEST=%f, -DGAM_TEST=%f",
          Nx, Nx_loc, Nx_pad, Nx_tot, Nx_blks, Nq, Nz, Ne, Ns, Nsim, Nsim_loc, Nt, Ngrps_sim, bet, gam);

  // knl = kernel_from_string(ctx, knl_text, "solve", buildOptions);
  cl_program prg = program_from_string(ctx, knl_text, buildOptions);
  free(knl_text);

  cl_kernel solve_iter_knl = clCreateKernel(prg, "solve_iter", &status);
  CHECK_CL_ERROR(status, "clCreateKernel");

  size_t ldim[3] = {Nx_loc, 1, Ns};
  // size_t gdim[3] = {ldim[0]*((Nx-1)/(ldim[0]-1) + 1), Nq, Ns};
  size_t gdim[3] = {Nx_tot, Nq, Ns};

  printf("Nx = %d, Nx_pad = %d \n", Nx, Nx_pad);
  printf("ldim = (%d, %d, %d) \n", ldim[0], ldim[1], ldim[2]);
  printf("gdim = (%d, %d, %d) \n", gdim[0], gdim[1], gdim[2]);

  // Simulation setup
  cl_int cleared;
  cl_double q_lb, q_ub, q_mid;

  // sim_psums kernel
  size_t ldim_sim[] = {Nsim_loc};
  size_t gdim_sim[] = {Nsim_loc*((Nsim-1)/Nsim_loc + 1)};

  cl_kernel sim_psums_knl = clCreateKernel(prg, "sim_psums", &status);
  CHECK_CL_ERROR(status, "clCreateKernel");

  // add_psums kernel
  size_t ldim_add[] = {Nsim_loc};
  size_t gdim_add[] = {Nsim_loc};

  cl_kernel add_psums_knl = clCreateKernel(prg, "add_psums", &status);
  CHECK_CL_ERROR(status, "clCreateKernel");

  // sim_psums kernel

  cl_kernel sim_update_knl = clCreateKernel(prg, "sim_update", &status);
  CHECK_CL_ERROR(status, "clCreateKernel");

  // Iterate to convergence over q_bar
  cl_int qbar_done = 0;
  cl_int qbar_iter = 0;
  while (qbar_done == 0)
    {
      ++qbar_iter;

      printf("\nITERATION %d: \n", qbar_iter);
      printf("q_bar = {%g, %g}, q_bar_old = {%g, %g} \n",
             q_bar[0], q_bar[1], q_bar_old[0], q_bar_old[1]);

      // Initialize c_old
      write_dbuf(queue, c_old_buf, c_init, Nx*Nq*Ns);
      // write_dbuf(queue, V_old_buf, V_init, Nx*Nq*Ns);

      // Solve agent's problem
      get_timestamp(&time1);
      iter = 0;
      *done_end = 0;

      while (*done_end == 0)
        {
          ++iter;
          // printf("ITERATION %d: \n", iter);
          // initialize with done = 1
          write_ibuf(queue, done_buf, done_start, 1);

          SET_9_KERNEL_ARGS(solve_iter_knl, c_buf, c_old_buf, x_buf, q_buf, y_buf,
                            P_buf, q_bar_buf, params_buf, done_buf);
          // Add local arguments
          for (cl_int ii = 9; ii < 9 + 4; ++ii)
            {
              SET_LOCAL_ARG(solve_iter_knl, ii, Nx_loc*Ns*sizeof(cl_double));
            }

          CALL_CL_GUARDED(clFinish, (queue));

          CALL_CL_GUARDED(clEnqueueNDRangeKernel,
                          (queue, solve_iter_knl, /*dimension*/ 3,
                           NULL, gdim, ldim, 0, NULL, NULL));

          CALL_CL_GUARDED(clFinish, (queue));

          // printf("HOST: exited kernel \n");

          // Transfer from device
          read_ibuf(queue, done_buf, done_end, 1);
          // printf("iteration %d complete \n", iter);
        }

      get_timestamp(&time2);
      elapsed = timestamp_diff_in_seconds(time1,time2);
      printf("Solution routine, time elapsed: %f s\n", elapsed);
      printf("%d iterations to convergence \n", iter);

      /*
        read_dbuf(queue, c_buf, c_all, Nx*Nq*Ns);
        read_dbuf(queue, c_old_buf, c_init, Nx*Nq*Ns);
      */
      /*
        read_dbuf(queue, V_buf, V_all, Nx*Nq*Ns);
        read_dbuf(queue, V_old_buf, V_old, Nx*Nq*Ns);
      */

      CALL_CL_GUARDED(clFinish, (queue));

      /*
        printf("after kernel \n");
        for (cl_int ix = 0; ix < Nx; ++ix)
        for (cl_int iq = 0; iq < Nq; ++iq)
        for (cl_int is = 0; is < Ns; ++is)
        if (iq == 0 && is == 0)
        printf("(%d, %d, %d): x = %g, c = %g c_old = %g \n",
        ix, iq, is, x_grid[ix], c_all[Ns*(Nq*ix + iq) + is], c_init[Ns*(Nq*ix + iq)]);
      */

      // Loop to convergence over q_bar
      get_timestamp(&time1);

      for (cl_int tt = 0; tt < Nt; ++tt)
        {
          cleared = 0;
          iter = 0;

          q_lb = q_min;
          q_ub = q_max;
          // printf("\n iteration %d: q_lb = %g, q_ub = %g \n", 0, q_lb, q_ub);

          while (cleared == 0)
            {
              ++iter;
              // printf("ITERATION %d: \n", iter);

              q_mid = (cl_double) 0.5*(q_lb + q_ub);

              SET_11_KERNEL_ARGS(sim_psums_knl, x_sim_buf, y_sim_buf, z_sim_buf, e_sim_buf,
                                 c_buf, params_buf, x_buf, q_buf, a_psums_buf, q_mid, tt);

              /*
                CALL_CL_GUARDED(clSetKernelArg, (sim_psums_knl, 0, sizeof(x_sim_buf), &x_sim_buf));
                CALL_CL_GUARDED(clSetKernelArg, (sim_psums_knl, 1, sizeof(y_sim_buf), &y_sim_buf));
                CALL_CL_GUARDED(clSetKernelArg, (sim_psums_knl, 2, sizeof(z_sim_buf), &z_sim_buf));
                CALL_CL_GUARDED(clSetKernelArg, (sim_psums_knl, 3, sizeof(e_sim_buf), &e_sim_buf));
                CALL_CL_GUARDED(clSetKernelArg, (sim_psums_knl, 4, sizeof(c_buf), &c_buf));
                CALL_CL_GUARDED(clSetKernelArg, (sim_psums_knl, 5, sizeof(params_buf), &params_buf));
                CALL_CL_GUARDED(clSetKernelArg, (sim_psums_knl, 6, sizeof(x_buf), &x_buf));
                CALL_CL_GUARDED(clSetKernelArg, (sim_psums_knl, 7, sizeof(q_buf), &q_buf));
                CALL_CL_GUARDED(clSetKernelArg, (sim_psums_knl, 8, sizeof(a_psums_buf), &a_psums_buf));
                CALL_CL_GUARDED(clSetKernelArg, (sim_psums_knl, 9, sizeof(q_mid), &q_mid));
                CALL_CL_GUARDED(clSetKernelArg, (sim_psums_knl, 10, sizeof(tt), &tt));
              */

              SET_LOCAL_ARG(sim_psums_knl, 11, Nsim_loc*sizeof(cl_double));

              CALL_CL_GUARDED(clFinish, (queue));

              CALL_CL_GUARDED(clEnqueueNDRangeKernel,
                              (queue, sim_psums_knl, /*dimension*/ 1,
                               NULL, gdim_sim, ldim_sim, 0, NULL, NULL));

              CALL_CL_GUARDED(clFinish, (queue));
              // printf("finished sim_psums \n");

              SET_2_KERNEL_ARGS(add_psums_knl, a_psums_buf, a_net_buf);
              SET_LOCAL_ARG(add_psums_knl, 2, Nsim_loc*sizeof(cl_double));

              CALL_CL_GUARDED(clEnqueueNDRangeKernel,
                              (queue, add_psums_knl, /*dimension*/ 1,
                               NULL, gdim_sim, ldim_sim, 0, NULL, NULL));

              CALL_CL_GUARDED(clFinish, (queue));
              read_dbuf(queue, a_net_buf, a_net, 1);
              CALL_CL_GUARDED(clFinish, (queue));
              // printf("finished add_psums \n");

              if (fabs(*a_net) > tol)
                {
                  if (*a_net > 0)
                    q_lb = q_mid;
                  else
                    q_ub = q_mid;
                }
              else
                cleared = 1;

              // printf("tt = %d, iteration %d: q_lb = %g, q_ub = %g, a_net = %g \n", tt, iter, q_lb, q_ub, *a_net);

            }

          q_sim[tt] = q_mid;

          if (tt < Nt-1)
            {
              SET_10_KERNEL_ARGS(sim_update_knl, x_sim_buf, y_sim_buf, z_sim_buf, e_sim_buf,
                                 c_buf, params_buf, x_buf, q_buf, q_mid, tt);

              CALL_CL_GUARDED(clEnqueueNDRangeKernel,
                              (queue, sim_update_knl, /*dimension*/ 1,
                               NULL, gdim_sim, ldim_sim, 0, NULL, NULL));
              CALL_CL_GUARDED(clFinish, (queue));
              // printf("finished sim_update \n");
            }

        }

      CALL_CL_GUARDED(clFinish, (queue));

      get_timestamp(&time2);
      elapsed = timestamp_diff_in_seconds(time1,time2);
      printf("Simulation routine, time elapsed: %f s\n", elapsed);

      qbar_done = update_qbar(q_bar, q_bar_old, q_sim, z_sim, Nt, Nburn, tol);
    }

  // OUTPUT RESULTS
#if OUTPUT
  read_dbuf(queue, c_buf, c_all, Nx*Nq*Ns);
  read_dbuf(queue, x_sim_buf, x_sim, Nsim*Nt);

  CALL_CL_GUARDED(clFinish, (queue));

  FILE *cfile, *xfile, *yfile, *zfile, *efile;
  cfile = fopen("cfile.out", "wb");
  xfile = fopen("xfile.out", "wb");
  yfile = fopen("yfile.out", "wb");
  zfile = fopen("zfile.out", "wb");
  efile = fopen("efile.out", "wb");

  printf("size of cl_double: %d, size of cl_int: %d \n", sizeof(cl_double), sizeof(cl_int));

  fwrite(c_all, sizeof(cl_double), Nx*Nq*Ns, cfile);
  fwrite(x_sim, sizeof(cl_double), Nsim*Nt, xfile);
  fwrite(y_sim, sizeof(cl_double), Nsim*Nt, yfile);
  fwrite(z_sim, sizeof(cl_int), Nt, zfile);
  fwrite(e_sim, sizeof(cl_int), Nsim*Nt, efile);
  printf("wrote all files \n");

  fclose(cfile);
  fclose(xfile);
  fclose(yfile);
  fclose(zfile);
  fclose(efile);

  printf("closed all files \n");
#endif

  // CLEAN UP
  CALL_CL_GUARDED(clReleaseMemObject, (x_sim_buf));
  CALL_CL_GUARDED(clReleaseMemObject, (y_sim_buf));
  CALL_CL_GUARDED(clReleaseMemObject, (a_psums_buf));
  CALL_CL_GUARDED(clReleaseMemObject, (a_net_buf));
  CALL_CL_GUARDED(clReleaseMemObject, (z_sim_buf));
  CALL_CL_GUARDED(clReleaseMemObject, (e_sim_buf));
  printf("released sim buffers \n");

  CALL_CL_GUARDED(clReleaseKernel, (sim_psums_knl));
  CALL_CL_GUARDED(clReleaseKernel, (add_psums_knl));
  CALL_CL_GUARDED(clReleaseKernel, (sim_update_knl));
  printf("released sim kernels \n");

  free(x_sim);
  free(y_sim);
  free(z_sim);
  free(e_sim);
  free(a_net);
  printf("released sim arrays \n");

  CALL_CL_GUARDED(clReleaseKernel, (solve_iter_knl));
  CALL_CL_GUARDED(clReleaseMemObject, (c_old_buf));
  free(c_init);

  CALL_CL_GUARDED(clReleaseMemObject, (done_buf));
  free(done_start);
  free(done_end);
  printf("released solution only objects \n");

  CALL_CL_GUARDED(clReleaseMemObject, (c_buf));
  CALL_CL_GUARDED(clReleaseMemObject, (x_buf));
  CALL_CL_GUARDED(clReleaseMemObject, (q_buf));
  CALL_CL_GUARDED(clReleaseMemObject, (y_buf));
  CALL_CL_GUARDED(clReleaseMemObject, (P_buf));
  CALL_CL_GUARDED(clReleaseMemObject, (q_bar_buf));
  printf("released solution buffers \n");
  CALL_CL_GUARDED(clReleaseProgram, (prg));
  CALL_CL_GUARDED(clReleaseCommandQueue, (queue));
  CALL_CL_GUARDED(clReleaseContext, (ctx));
  printf("released other solution cl objects \n");

  free(c_all);
  // free(V_all);
  free(x_grid);
  free(q_grid);
  free(y_grid);
  free(q_bar);
  free(q_bar_old);
  free(P);
  printf("released other solution arrays \n");

  return 0;

}
