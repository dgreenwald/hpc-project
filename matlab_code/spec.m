close all;
clear;

tol = 1e-6;  % Max change in function for termination

Nx = 100;    % No. of points in total wealth grid
Nq = 10;     % No. of points in bond price grid
Nz = 2;      % No. of aggregate productivity states
Ny = 4;      % No. of idiosyncratic productivity states
Nw = Nz*Ny;  % No. of total productivity states
Ns = Nx;     % Number of points in savings grid

k = 0.4;     % Polynomial grid curvature parameter

%% Preferences

gam = 2;
bet = 0.95;

U = @(c) (c.^(1-gam))/(1-gam);

%% Wage/Productivity Process

Pz = [0.8, 0.2; 0.2, 0.8];  % Transition matrix for aggregate state
z_grid = [0.96; 1.04];      % Productivity under aggregate state

rho_y = 0.95;  % Persistence of log idiosyncratic productivity
sig_y = 0.05;  % Volatility of log idiosyncratic productivity
[Py, y_til] = mcapprox_r(rho_y, sig_y, Ny, 0);  % Discretize log idiosyncratic productivity process
y_grid = exp(y_til)';  % Values for idiosyncratic productivity process

w_grid = kron(z_grid, y_grid);
Pw = kron(Pz, Py);

%% Bond Prices

q_min = 0.9;
q_max = 1.1;

q_grid = linspace(q_min, q_max, 10)';

q_bar = kron(1./z_grid, ones(Ny, 1));

%% Total wealth

x_min = min(w_grid);
x_max = 100;

% This function generates a polynomially spaced grid, so that there are more points near x = 0
% where the value function should have less curvature
x_grid = poly_grid(x_min, x_max, Nx, k);

% Required for interp2
[x_mat, q_mat] = meshgrid(x_grid, q_grid);

%% Savings

s_min = 0;
s_max = (x_max - max(w_grid))*min(q_grid);

s_grid = poly_grid(s_min, s_max, Ns, k);