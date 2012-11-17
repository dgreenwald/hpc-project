% get_Omega.m

% This program calculates the expected marginal utility (E U') given a savings choice.
% The inputs are:

% c_next: next period's policy function describing how asset holdings and prices translate 
%         into consumption.

% gamma: the risk aversion parameter

% R: the gross interest rate

% x_grid: a vector of values for total cash on hand on which we have defined our functions

% q_grid: a vector of value for bond prices (inverse of the interest rate) on which we have defined our functions

% s_grid: a vector of savings choices that we want to consider

% q_bar: average value of bond prices conditional on next period's state

% y_bar: value of income conditional on next period's state

% [x_mat, q_mat]: the output of meshgrid(a_grid, q_grid) used for interpolation

function Omega = get_Omega(c_next, gamma, R, a_grid, q_grid, s_grid, q_bar, y_bar, a_mat, q_mat, P)


Np = length(P);

Ns = length(s_grid);  % Number of points in s_grid
Nq = length(q_grid);  % Number of points in q_grid
Omega = zeros(Ns, Nq, Np);  % Preallocate output matrix

s_stack = repmat(s_grid, 1, Nq);    % Stack 1 x Nq columns of s_grid
q_stack = repmat(q_grid', Ns, 1);   % Stack Ns x 1 rows of q_grid

a_next = s_stack./q_stack;  % Next period's assets (savings divided by current bond price, i.e. savings times current
                            % interest rate).

for is = 1:Np
    for js = 1:Np
        y_next = y_bar(js);  % Next period's income under this state
        x_next = a_next + y_next;  % Cash on hand = assets plus income
        q_next = q_bar(js)*ones(Ns,1);  % Need to know next period's bond prices -- assume average value
        dU_next = interp2(a_mat, q_mat, c_next(:,:,js), x_next, q_next, 'linear').^(-gam);  % Interpolate to get E U'
        Omega(:,:,is) = Omega(:,:,is) + P(is,js)*dU_next; % Weight by transition probability and add to running total
    end
end