#include "timing.h"
#include "cl-helper.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

cl_double* poly_grid(double f_min, double f_max, double k, int N)
{
  cl_double *f = malloc(sizeof(cl_double)*N);
  if (!f) { perror("alloc error in poly_grid"); abort(); }

  for (int ii = 1; ii < N; ++ii)
    {
      f[ii] = f_min + (f_max - f_min) * pow(ii/(N-1), 1/k);
    }
}

cl_double* getP(double u_b, double u_g, double dur_b, double dur_g, double udur_b, double udur_g,
                double rat_bg, double rat_gb)
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
  for (int ii = 0; ii < 4; ++ii)
    {
      Psum = 0;
      for (int jj = 0; jj < 4; ++jj)
        {
          if (P[4*jj + ii] < 0) { perror("P < 0"); abort(); }
          Psum += P[4*jj + ii];
        }
      if (Psum != 1) { perror("Psum != 1"); abort(); }
    }

  return P;

}

int main(int argc, char **argv)
{

  // Timing Setup

  timestamp_type time1, time2;
  double elapsed;

  // OpenCL Setup

  print_platforms_devices();

  cl_context ctx;
  cl_command_queue queue;
  cl_int status;

  char* knl_text;
  cl_kernel knl;

  create_context_on("NVIDIA", NULL, 0, &ctx, &queue, 0);
  // create_context_on("Intel", NULL, 0, &ctx, &queue, 0);

  // Define parameters

  const cl_double freq = 4;
  const cl_double tol = 1e-6;
  const cl_double k = 0.4;
  const cl_double alp = 0.36;
  const cl_double gam = 2.0;
  const cl_double bet = pow(0.95, 1/freq);
  const cl_double q_min = pow(0.8, 1/freq);
  const cl_double q_max = pow(1.25, 1/freq);
  const cl_double x_min = 0;
  const cl_double x_max = 100;
  const cl_int Nx = 200;
  const cl_int Nq = 50;
  const cl_int Nz = 2;
  const cl_int Ne = 2;
  const cl_int Ns = Nz*Ne;

  cl_double* x_grid = poly_grid(x_min, x_max, k, Nx);
  cl_double* q_grid = poly_grid(q_min, q_max, 1.0, Nq); // 1.0 for even grid

  cl_double z_grid[2];
  z_grid[0] = 0.99;
  z_grid[1] = 1.01;
  cl_double e_grid[2];
  e_grid[0] = 0.3;
  e_grid[1] = 1.0;

  const cl_double u_b = 0.1;
  const cl_double u_g = 0.04;
  const cl_double dur_b = 8.0;
  const cl_double dur_g = 8.0;
  const cl_double udur_b = 2.5;
  const cl_double udur_g = 1.5;
  const cl_double rat_bg = 0.75;
  const cl_double rat_gb = 1.25;

  cl_double *P = getP(u_b, u_g, dur_b, dur_g, udur_b, udur_g, rat_bg, rat_gb);

  cl_double *w_grid;
  w_grid[0] = z_grid[0]*pow(1 - u_b, alp);
  w_grid[1] = z_grid[1]*pow(1 - u_g, alp);

  // Allocate CPU memory
  cl_double *c_all = malloc(sizeof(cl_double) * Nx * Nq * Ns);
  if (!c_all) { perror("alloc x_endog"); abort(); }

  cl_double *V_all = malloc(sizeof(cl_double) * Nx * Nq * Ns);
  if (!V_all) { perror("alloc c_endog"); abort(); }

  // Initialize Matrices

  for (int ix = 0; ix < Nx; ++ix)
    for (int iq = 0; iq < Nq; ++iq)
      for (int iz = 0; iz < Nz; ++iz)
        for (int ie = 0; ie < Ne; ++ie)
          {
            c_all[Ne*(Nz*(Nq*ix + iq) + iz) + ie] = x_grid[ix] + w_grid[iz]*e_grid[ie]; // Zero bond solution
            V_all[Ne*(Nz*(Nq*ix + iq) + iz) + ie] = pow(c_all[Ne*(Nz*(Nq*ix + iq) + iz) + ie], 1-gam)/(1-gam); 
          }

  // Allocate device buffers

  cl_mem c_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_double) * Nx * Nq * Ns, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem V_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_double) * Nx * Nq * Ns, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem x_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(cl_double) * Nx, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem q_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(cl_double) * Nq, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem w_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(cl_double) * Nz, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem e_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(cl_double) * Ne, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  // Transfer to device

  CALL_CL_GUARDED(clEnqueueWriteBuffer,
                  (queue, c_buf, /*blocking*/ CL_FALSE, /*offset*/ 0,
                   Nx * Nq * Ns * sizeof(cl_double), c_all,
                   0, NULL, NULL));

  CALL_CL_GUARDED(clEnqueueWriteBuffer,
                  (queue, V_buf, /*blocking*/ CL_FALSE, /*offset*/ 0,
                   Nx * Nq * Ns * sizeof(cl_double), V_all,
                   0, NULL, NULL));

  CALL_CL_GUARDED(clEnqueueWriteBuffer,
                  (queue, x_buf, /*blocking*/ CL_FALSE, /*offset*/ 0,
                   Nx * sizeof(cl_double), x_grid,
                   0, NULL, NULL));

  CALL_CL_GUARDED(clEnqueueWriteBuffer,
                  (queue, q_buf, /*blocking*/ CL_FALSE, /*offset*/ 0,
                   Nq * sizeof(cl_double), q_grid,
                   0, NULL, NULL));

  CALL_CL_GUARDED(clEnqueueWriteBuffer,
                  (queue, w_buf, /*blocking*/ CL_FALSE, /*offset*/ 0,
                   Nz * sizeof(cl_double), w_grid,
                   0, NULL, NULL));

  CALL_CL_GUARDED(clEnqueueWriteBuffer,
                  (queue, e_buf, /*blocking*/ CL_FALSE, /*offset*/ 0,
                   Ne * sizeof(cl_double), e_grid,
                   0, NULL, NULL));

  // Run solve.cl on device

  knl_text = read_file("solve.cl");
  knl = kernel_from_string(ctx, knl_text, "solve", NULL);
  free(knl_text);

  get_timestamp(&time1);

  CALL_CL_GUARDED(clFinish, (queue));
  SET_11_KERNEL_ARGS(knl, c_buf, V_buf, x_buf, q_buf, w_buf, e_buf, Nq, Nz, Ne, bet, gam);

  size_t ldim[] = {Nx, 1};
  size_t gdim[] = {Nx, Nq};

  CALL_CL_GUARDED(clEnqueueNDRangeKernel,
                  (queue, knl,
                   /*dimensions*/ 2, NULL, gdim, ldim,
                   0, NULL, NULL));

  CALL_CL_GUARDED(clFinish, (queue));

  clReleaseKernel(knl); // Release solve_endog kernel

  get_timestamp(&time2);
  elapsed = timestamp_diff_in_seconds(time1,time2);
  printf("%f s\n", elapsed);

  // Transfer from device
  CALL_CL_GUARDED(clEnqueueReadBuffer,
                  (queue, c_buf, /*blocking*/ CL_FALSE, /*offset*/ 0,
                   Nx * Nq * Ns * sizeof(cl_double), c_all,
                   0, NULL, NULL));

  CALL_CL_GUARDED(clEnqueueReadBuffer,
                  (queue, V_buf, /*blocking*/ CL_FALSE, /*offset*/ 0,
                   Nx * Nq * Ns * sizeof(cl_double), V_all,
                   0, NULL, NULL));

  // Clean up
  CALL_CL_GUARDED(clFinish, (queue));
  CALL_CL_GUARDED(clReleaseMemObject, (c_buf));
  CALL_CL_GUARDED(clReleaseMemObject, (V_buf));
  CALL_CL_GUARDED(clReleaseMemObject, (x_buf));
  CALL_CL_GUARDED(clReleaseMemObject, (q_buf));
  CALL_CL_GUARDED(clReleaseMemObject, (w_buf));
  CALL_CL_GUARDED(clReleaseMemObject, (e_buf));

  free(c_all);
  free(V_all);
  free(x_grid);
  free(q_grid);
  free(P);

}
