% Rouwenhorst method of finite-state Markov chain approximation to AR(1)

function [P,ybar] = mcapprox_r(rho,sig_e,N,cons)

q = 0.5*(1+rho);
sig_z = sqrt((sig_e^2)/(1-rho^2));
psi = sqrt(N-1)*sig_z;

ybar = linspace(-psi,psi,N) + cons;

P = [q,1-q;1-q,q];

for i = 3:N
    P = q*[P,zeros(i-1,1); zeros(1,i-1),0] + (1-q)*[zeros(i-1,1),P; 0,zeros(1,i-1)] ...
        + (1-q)*[zeros(1,i-1),0;P,zeros(i-1,1)] + q*[0,zeros(1,i-1);zeros(i-1,1),P];
    P(2:end-1,:) = 0.5*P(2:end-1,:);
end

for i = 1:N
	P(i,:) = P(i,:)/sum(P(i,:));
end