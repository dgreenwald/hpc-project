fclose('all');
close all;
clear;

Nx = 1000;
Nq = 1000;
Ns = 4;
Nsim = 256;
Nt = 1200;
Nburn = 200;

% fid_c = fopen('cfile.out','r');
% fid_e = fopen('efile.out','r');
fid_x = fopen('xfile.out','r');
fid_y = fopen('yfile.out','r');
fid_q = fopen('qfile.out','r');
% fid_z = fopen('zfile.out','r');

% c_all = reshape(fread(fid_c, Nx*Nq*Ns, 'double'), [Nx, Nq, Ns]);
% e_sim = reshape(fread(fid_e, Nsim*Nt, 'int'), [Nsim, Nt]);
x_sim = reshape(fread(fid_x, Nsim*Nt, 'double'), [Nsim, Nt]);
y_sim = reshape(fread(fid_y, Nsim*Nt, 'double'), [Nsim, Nt]);
q_sim = fread(fid_q, Nt, 'double');
% z_sim = reshape(fread(fid_z, Nt, 'int'), [Nsim, Nt]);

b_sim = x_sim(:,2:end)./repmat(q_sim(1:end-1)', Nsim, 1);
c_sim = x_sim(:,1:end-1) + y_sim(:,1:end-1) - b_sim;
x_sim(:,end) = [];
y_sim(:,end) = [];

Nplot = 100;
figure(1);
for ii = 1:4
subplot(2,2,ii);
P = plot(1:Nplot, x_sim(ii,end-Nplot+1:end), 'r-', 1:Nplot, y_sim(ii,end-Nplot+1:end), 'b-', 1:Nplot, ...
         c_sim(ii,end-Nplot+1:end), 'g-', 1:Nplot, q_sim(end-Nplot:end-1), 'k-');
set(P,'LineWidth',2);
set(gca,'YLim',[-0.5, 2]);
title(['Sample path ' num2str(ii)]);
legend('Wealth','Income','Consumption', 'Bond Price');
end
orient landscape;
saveas(figure(1), 'samplepaths.pdf');

%% Vary Nx Plot
Nx = [1000; 2000; 3000; 4000; 8000];
tx = [3.834364; 9.790992; 18.261715; 29.712004; 98.547297];

%% Vary Nq Plot
Nq = [500; 1000; 2000; 4000];
tq = [2.053042,  4.075932, 7.317281, 16.060123];

%% Vary Nsim Plot
Nsim = [512, 1024, 2048, 4096];
tsim = [10.992711, 11.104238, 11.583534, 12.090612];

%% Vary Nt Plot
Nt = [600, 1200, 2400];
tt = [5.633019, 11.104238, 22.348857];

figure(2);
subplot(2,2,1);
plot(Nx, tx,'LineWidth',2);
xlabel('Nx');
ylabel('Solution (s)');
title('Timings: Varying Nx');

subplot(2,2,2);
plot(Nq, tq,'LineWidth',2);
xlabel('Nq');
ylabel('Solution (s)');
title('Timings: Varying Nq');

subplot(2,2,3);
plot(Nt, tt,'LineWidth',2);
xlabel('Nt');
ylabel('Simulation (s)');
title('Timings: Varying Nt');

subplot(2,2,4);
plot(Nsim, tsim,'LineWidth',2);
xlabel('Nsim');
ylabel('Simulation (s)');
title('Timings: Varying Nsim');

orient landscape;
saveas(figure(2), 'timings.pdf');

fclose('all');