function [V,c] = solve(x_grid, q_grid, s_grid, w_grid, q_bar, x_mat, q_mat, ...
                        Pw, bet, gam, U, x_min, x_max, Nx, Nq, Nw, Ns, tol)


%% Apply terminal condition to start iterations

% Preallocating the output matrices with 3-D arrays of zeros
c = zeros(Nx, Nq, Nw); 
V = zeros(Nx, Nq, Nw);
for ii = 1:Nq  % Start with a guess.
    c(:,ii,:) = repmat(x_grid, 1, Nw) + repmat(w_grid', Nx, 1); % repmat tiles matrices
    V(:,ii,:) = U(c(:,ii,:));
end

%% Iterate to convergence

V_old = -Inf*ones(Nx, Nq, Nw);

it = 0;
tic
while max(max(max(abs(V - V_old)))) >= tol
    
    it = it + 1;
    V_old = V;
    
    % Calculates expected value function and marginal utility for next period
    [EV, EdU] = get_exp(c, V, gam, s_grid, q_grid, w_grid, q_bar, x_mat, q_mat, Pw, Nx, Nq, Nw, Ns);
    % Solves for optimal policy (c) and value function (V)
    [V,c] = solve_iter(EV, EdU, x_grid, q_grid, s_grid, bet, gam, U, x_min, x_max, Nx, Nq, Nw, Ns);
    
    if it == 10*floor(it/10)
        fprintf('Iteration %d complete, diff: %g, performance: %g it/s \n', it, max(max(max(abs(V - V_old)))), it/toc);
    end
    
end
