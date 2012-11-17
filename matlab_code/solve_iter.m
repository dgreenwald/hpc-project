% solve_iter.m

% This program solves for the next iteration of functions (i.e. for the previous period), given a set of solutions,
% represented by Omega. The solution uses the "endogenous grid method" in which the nonlinear equation is solved by

% 1. Starting with a grid of savings values 

% 2. Applying earlier results to get next period's expected marginal utility given savings

% 3. Inverting the nonlinear Euler equation to obtain implied consumption values this period

% 4. Using the budget constraint (savings plus consumption equals starting cash on hand) to get implied starting cash
% on hand

% 5. Interpolating to get back to the original cash on hand grid

% The inputs are:

% Omega: expected marginal utility (i.e. E U'(c)), calculated from the previous set of solutions.

% x_grid, q_grid: vectors of values for x and q on which we define our functions

% P: the transition matrix for the stochastic processes

% bet, gam: preference paramters (discount factor/impatiance, and risk aversion, respectively).

function c = solve_iter(Omega, x_grid, q_grid, P, bet, gam)

% Get sizes of various grids
Np = length(P);
Nx = length(x_grid);
Nq = length(q_grid);

% Preallocate arrays
V = zeros(Nx, Nq, Np);
c = zeros(Nx, Nq, Np);

% Stack grids
s_stack = repmat(s_grid, 1, Nq);    % Stack 1 x Nq columns of s_grid
q_stack = repmat(q_grid', Ns, 1);   % Stack Ns x 1 rows of q_grid

% Loop through current state
for is = 1:Np

    % Calculate expected marginal return on savings.
    % Note, this will only work if Omega is defined on the same grids.
    euler_rhs = bet*Omega(:,:,is)./q_stack;

    % Calculate implied current consumption.
    c_endog = euler_rhs.^(-1/gam);

    % Calculate implied starting cash on hand
    x_endog = c_endog + s_stack;

    % Interpolate to get optimal consumption policy to original grid
    % Note additional points added to bottom of grid for constrained region in which agent consumes all income
    for iq = 1:Nq
        c(:,iq,is) = interp1([xmin; x_endog], [0; c_endog], x_grid, 'linear');
    end

end
