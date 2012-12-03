#include "timing.h"
#include "cl-helper.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_interp.h>

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
  cl_kernel solve_endog_knl;
  cl_kernel eval_v_knl;

  size_t origin, region, ldim, gdim;

  create_context_on("NVIDIA", NULL, 0, &ctx, &queue, 0);

  // Define parameters

  const cl_float tol, k, gam, bet, q_min, q_max, x_min, x_max, s_min, s_max;
  const cl_int Nx, Nq, Nz, Ny, Nw, Ns;
  const cl_float* Pz, Py, Pw, z_grid, y_grid, w_grid;

  tol = 1e-6;  // Max change in function for termination
  k = 0.4;  // Polynomial grid curvature

  Nx = 100;  // No. of points in total wealth grid
  Nq = 100;  // No. of points in bond price grid
  Nz = 2;  // No. of aggregate productivity states
  Ny = 4;  // No. of idiosyncratic productivity states
  Nw = Nz*Ny;  // No. of total productivity states
  Ns = Nx;  // Number of points in savings grid

  gam = 2;
  bet = 0.95;

  Pz = {0.8, 0.2, 0.2, 0.8};
  z_grid = {0.96, 1.04};

  /*****
        Need code for mapprox_r and kron
  *****/

  // Allocate CPU memory
  cl_float ***c_endog = malloc(sizeof(cl_float) * Nw * Nq * Ns);
  if (!c_endog) { perror("alloc c_endog"); abort(); }

  cl_float ***x_endog = malloc(sizeof(cl_float) * Nw * Nq * Ns);
  if (!x_endog) { perror("alloc x_endog"); abort(); }

  cl_float ***c_array = malloc(sizeof(cl_float) * Nw * Nq * Nx);
  if (!c) { perror("alloc c"); abort(); }

  cl_float ***V_array = malloc(sizeof(cl_float) * Nw * Nq * Nx);
  if (!x) { perror("alloc x"); abort(); }

  cl_float ***EV_array = malloc(sizeof(cl_float) * Nw * Nq * Ns);
  if (!x) { perror("alloc x"); abort(); }

  cl_float* x_grid = malloc(sizeof(cl_float) * Nx);
  if (!x_grid) { perror("alloc x_grid"); abort(); }
  cl_float* s_grid = malloc(sizeof(cl_float) * Ns);
  if (!s_grid) { perror("alloc s_grid"); abort(); }
  cl_float* q_grid = malloc(sizeof(cl_float) * Nq);
  if (!q_grid) { perror("alloc q_grid"); abort(); }
  cl_float* w_grid = malloc(sizeof(cl_float) * Nw);
  if (!w_grid) { perror("alloc w_grid"); abort(); }

  // Allocate device buffers

  cl_mem c_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_float) * Nq * Nw * Nx, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem x_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_float) * Nq * Nw * Nx, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem c_endog_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_float) * Nq * Nw * Ns, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem x_endog_buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_float) * Nq * Nw * Ns, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  // Allocate device images

  cl_mem_flags img_flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;

  cl_image_format img_format;
  img_format.image_channel_order = CL_A;
  img_format.image_channel_data_type = CL_FLOAT;

  cl_mem c_img = clCreateImage3D(ctx, img_flags, img_format, Nw, Nq, Nx,
                                 0, 0, c_array, status);
  CHECK_CL_ERROR(status, "clCreateImage3D");

  cl_mem V_img = clCreateImage3D(ctx, img_flags, img_format, Nw, Nq, Nx,
                                 0, 0, V_array, status);
  CHECK_CL_ERROR(status, "clCreateImage3D");

  cl_mem EV_img = clCreateImage3D(ctx, img_flags, img_format, Nw, Nq, Nx,
                                  0, 0, EV_array, status);
  CHECK_CL_ERROR(status, "clCreateImage3D");

  // Transfer to device

  CALL_CL_GUARDED(clEnqueueWriteBuffer,
                  (queue, s_buf, /*blocking*/ CL_FALSE, /*offset*/ 0,
                   Ns * sizeof(cl_float), s_grid,
                   0, NULL, NULL));

  CALL_CL_GUARDED(clEnqueueWriteBuffer,
                  (queue, q_buf, /*blocking*/ CL_FALSE, /*offset*/ 0,
                   Nq * sizeof(cl_float), q_grid,
                   0, NULL, NULL));

  CALL_CL_GUARDED(clEnqueueWriteBuffer,
                  (queue, w_buf, /*blocking*/ CL_FALSE, /*offset*/ 0,
                   Mw * sizeof(cl_float), w_grid,
                   0, NULL, NULL));

  origin = {0, 0, 0};
  region = {Nw, Nq, Ns};

  CALL_CL_GUARDED(clEnqueueWriteImage,
                  (queue, c_img, /*blocking*/ CL_FALSE, origin, region,
                   0, 0, c_array, 0, NULL, NULL));

  CALL_CL_GUARDED(clEnqueueWriteImage,
                  (queue, V_img, /*blocking*/ CL_FALSE, origin, region,
                   0, 0, V_array, 0, NULL, NULL));

  // Run solve_endog code on device

  knl_text = read_file("solve_endog.cl");
  knl = kernel_from_string(ctx, knl_text, "fill", NULL);
  free(knl_text);

  get_timestamp(&time1);

  CALL_CL_GUARDED(clFinish, (queue));
  SET_12_KERNEL_ARGS(knl, c_img, V_img, EV_img, c_endog, x_endog, s_grid, q_grid, w_grid,
                     a_next, q_bar, gam, bet);
  ldim = {1, 1, 1};
  gdim = {Nw, Nq, Ns};

  CALL_CL_GUARDED(clEnqueueNDRangeKernel,
                  (queue, knl,
                   /*dimensions*/ 3, NULL, gdim, ldim,
                   0, NULL, NULL));

  CALL_CL_GUARDED(clFinish, (queue));

  clReleaseKernel(knl); // Release solve_endog kernel

  get_timestamp(&time2);
  elapsed = timestamp_diff_in_seconds(time1,time2);
  printf("%f s\n", elapsed);

  // Transfer from device
  CALL_CL_GUARDED(clEnqueueReadBuffer,
                  (queue, c_endog_buf, /*blocking*/ CL_FALSE, /*offset*/ 0,
                   Ns * sizeof(cl_float), c_endog,
                   0, NULL, NULL));

  CALL_CL_GUARDED(clEnqueueReadBuffer,
                  (queue, x_endog_buf, /*blocking*/ CL_FALSE, /*offset*/ 0,
                   Ns * sizeof(cl_float), x_endog,
                   0, NULL, NULL));

  /* CALL_CL_GUARDED(clEnqueueReadImage, */
  /*                 (queue, EV_img, /\*blocking*\/ CL_FALSE, origin, region, */
  /*                  0, 0, EV_array, 0, NULL, NULL)); */

  // Interpolate to return from c_endog and x_endog to x_grid, fill in c_array, V_array

  gsl_interp_accel *acc;
  gsl_interp *interp1;  

  for (int iw = 0; iw < Nw; ++iw)
    {
      for (int iq = 0; iq < Nq; ++iq)
	{
	  acc = gsl_interp_accel_alloc(); // Need to allocate/free each time?
	  interp1 = gsl_interp_alloc(gsl_interp_linear, Ns); // Need to allocate/free each time?
	  
	  for (int ix = 0; ix < Nx; ++ix)
	    {
	      c_array[iw][iq][ix] = gsl_interp_eval(x_endog[iw][iq], c_endog[iw][iq], x_grid[ix], acc);
	    }

	  gsl_interp_free(acc); // Need to allocate/free each time?
	  gsl_interp_free(interp1); // Need to allocate/free each time?
	}
    }

  // Clean up
  CALL_CL_GUARDED(clFinish, (queue));
  CALL_CL_GUARDED(clReleaseMemObject, (x_buf));
  CALL_CL_GUARDED(clReleaseMemObject, (s_buf));
  CALL_CL_GUARDED(clReleaseMemObject, (q_buf));
  CALL_CL_GUARDED(clReleaseMemObject, (w_buf));
  CALL_CL_GUARDED(clReleaseMemObject, (c_endog_buf));
  CALL_CL_GUARDED(clReleaseMemObject, (x_endog_buf));
  CALL_CL_GUARDED(clReleaseMemObject, (c_img));
  CALL_CL_GUARDED(clReleaseMemObject, (V_img));
  CALL_CL_GUARDED(clReleaseMemObject, (EV_img));

  free(c_endog);
  free(x_endog);
  free(c_array);
  free(V_array);
  free(EV_array);
  free(x_grid);
  free(s_grid);
  free(q_grid);
  free(w_grid);

}
