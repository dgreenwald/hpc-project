function xgrid = poly_grid(xmin,xmax,nx,k)

zgrid = linspace(0,1,nx)'.^(1/k);
xgrid = xmin + (xmax - xmin)*zgrid;