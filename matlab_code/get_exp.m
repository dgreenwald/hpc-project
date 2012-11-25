% get_Omega.m

% This program calculates the expected marginal utility (E U') given a savings choice.
% The inputs are:

% c_next: next period's policy function describing how asset holdings and prices translate 
%         into consumption.

% gamma: the risk aversion parameter

% x_grid, q_grid: vectors of values for x and q on which we define our functions

% s_grid: a vector of savings choices that we want to consider

% q_bar: average value of bond prices conditional on next period's state

% w_grid: value of income conditional on next period's state

% [x_mat, q_mat]: the output of meshgrid(x_grid, q_grid) used for interpolation

% P: the transition probability matrix for the stochastic processes

function [EV, EdU] = get_exp(c_next, V_next, gam, s_grid, q_grid, w_grid, q_bar, x_mat, q_mat, ...
                                Pw, Nx, Nq, Nw, Ns)

EV = zeros(Ns, Nq, Nw);  % Preallocate matrix
EdU = zeros(Ns, Nq, Nw);  % Preallocate matrix

s_stack = repmat(s_grid, 1, Nq);    % Stack 1 x Nq columns of s_grid
q_stack = repmat(q_grid', Ns, 1);   % Stack Ns x 1 rows of q_grid

a_next = s_stack./q_stack;  % Next period's assets (savings divided by current bond price, i.e. savings times current
                            % interest rate).

for is = 1:Nw
    for js = 1:Nw
        x_next = a_next + w_grid(js); % Cash on hand = assets plus income
        q_next = q_bar(js)*ones(Ns, Nq);  % Need to know next period's bond prices -- assume average value
        
        V_w = interp2(x_mat, q_mat, V_next(:,:,js)', x_next, q_next, 'linear');  % Interpolate to get V given w'
        dU_w = interp2(x_mat, q_mat, c_next(:,:,js)', x_next, q_next, 'linear').^(-gam);  % Interpolate to get dU given w'
        
        if any(any(isnan(V_w))) || any(any(isnan(dU_w)))
            keyboard;
        end
        
        EV(:,:,is) = EV(:,:,is) + Pw(is,js)*V_w; % Weight by transition probability and add to total expectation
        EdU(:,:,is) = EdU(:,:,is) + Pw(is,js)*dU_w; 
    end
end