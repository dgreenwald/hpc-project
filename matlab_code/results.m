fclose('all');
close all;
clear;

Nx = 1000;
Nq = 1000;
Ns = 4;
Nsim = 256;
Nt = 1200;
Nburn = 200;

% % fid_c = fopen('cfile.out','r');
% % fid_e = fopen('efile.out','r');
% fid_x = fopen('xfile.out','r');
% fid_y = fopen('yfile.out','r');
% fid_q = fopen('qfile.out','r');
% % fid_z = fopen('zfile.out','r');

% % c_all = reshape(fread(fid_c, Nx*Nq*Ns, 'double'), [Nx, Nq, Ns]);
% % e_sim = reshape(fread(fid_e, Nsim*Nt, 'int'), [Nsim, Nt]);
% x_sim = reshape(fread(fid_x, Nsim*Nt, 'double'), [Nsim, Nt]);
% y_sim = reshape(fread(fid_y, Nsim*Nt, 'double'), [Nsim, Nt]);
% q_sim = fread(fid_q, Nt, 'double');
% % z_sim = reshape(fread(fid_z, Nt, 'int'), [Nsim, Nt]);

% b_sim = x_sim(:,2:end)./repmat(q_sim(1:end-1)', Nsim, 1);
% c_sim = x_sim(:,1:end-1) + y_sim(:,1:end-1) - b_sim;
% x_sim(:,end) = [];
% y_sim(:,end) = [];

% Nplot = 100;
% figure(1);
% for ii = 1:4
% subplot(2,2,ii);
% P = plot(1:Nplot, x_sim(ii,end-Nplot+1:end), 'r-', 1:Nplot, y_sim(ii,end-Nplot+1:end), 'b-', 1:Nplot, ...
%          c_sim(ii,end-Nplot+1:end), 'g-', 1:Nplot, q_sim(end-Nplot:end-1), 'k-');
% set(P,'LineWidth',2);
% set(gca,'YLim',[-0.5, 2]);
% title(['Sample path ' num2str(ii)]);
% legend('Wealth','Income','Consumption', 'Bond Price');
% end
% orient landscape;
% saveas(figure(1), 'samplepaths.pdf');

fclose('all');

%% GPU Plots

%% Vary Nx Plot
Nx = [125, 250, 500, 1000, 2000, 4000, 6000, 8000, 12000, 16000];
tx = [0.286988/125, 0.591280/124, 1.343251/124, 3.652484/122, 11.508120/121, 40.819724/122, 88.305932/122, ...
      152.642161/122, NaN, NaN];
tx_rel = tx./(Nx*1000);

%% Vary Nq Plot
Nq = [125, 250, 500, 1000, 2000, 4000, 6000, 8000, 12000, 16000];
tq = [0.486255/119, 0.954185/122, 1.836934/122, 3.641683/122, 6.731026/113, 14.414950/122, 21.555534/122, 28.825352/ ...
      122, NaN, NaN];
tq_rel = tq./(Nq*1000);

%% Vary Nsim Plot
Nsim = [256, 256, 512, 1024, 2048, 4096, 6144, 8192, 12032, 16128];
tsim = [NaN, 8.069625, 8.503930, 8.770216, 9.135744, 9.613263, 9.797882, 10.130390, 10.525847, 10.881414];
tsim_rel = tsim./(Nsim*1200);

%% Vary Nt Plot
Nt = [150, 300, 600, 1200, 2400, 4800, 7200, 9600, 14400, 19200];
tt = [1.218232, 2.429021, 4.895382, 9.702204, 19.488761, 38.860215, 58.452754, 77.713590, 117.383882, 155.566182];
tt_rel = tt./(Nt*5120);

%% CPU Plots

%% Vary Nx Plot
tx_cpu = [3.200975/109, 6.107784/103, 12.465084/101, 26.080841/102, 53.359727/98, 112.885596, 174.475109, 247.291272, NaN, NaN];
tx_cpu_rel = tx_cpu./(Nx*1000);

%% Vary Nq Plot
tq_cpu = [3.562676/112, 6.579685/103, 15.018404/117, 26.390958/103, 55.558352/108, 116.237298, 152.637958, 231.218435, NaN, NaN];
tq_cpu_rel = tq_cpu./(Nq*1000);

%% Vary Nsim Plot
tsim_cpu = [NaN, 9.583732, 11.962748, 13.025792, 13.691103, 25.045285, 31.847182, 40.148719, 54.819946, 75.594829];
tsim_cpu_rel = tsim_cpu./(Nsim*1200);

%% Vary Nt Plot
tt_cpu = [3.342499, 6.761099, 13.854158, 28.707261, 54.403391, 108.646633, 161.419926, NaN, NaN, NaN];
tt_cpu_rel = tt_cpu./(Nt*5120);

figure(2);
subplot(2,2,1);
P = plot(Nx, tx, 'b-');
set(P, 'LineWidth', 2);
xlabel('Nx');
ylabel('sec');
title('Varying Nx');

subplot(2,2,2);
P = plot(Nx, tq, 'b-');
set(P, 'LineWidth', 2);
xlabel('Nq');
ylabel('sec');
title('Varying Nq');

subplot(2,2,3);
P = plot(Nx, tt, 'b-');
set(P, 'LineWidth', 2);
xlabel('Nt');
ylabel('sec');
title('Varying Nt');

subplot(2,2,4);

P = plot(Nx, tsim, 'b-');
set(P, 'LineWidth', 2);
xlabel('Nsim');
ylabel('sec');
title('Varying Nsim');

orient landscape;
saveas(figure(2), 'timings.pdf');

figure(3);

subplot(2,2,1);
P = plot(Nx, tx_rel, 'b-');
set(P, 'LineWidth', 2);
xlabel('Nx');
ylabel('sec');
title('Varying Nx');

subplot(2,2,2);
P = plot(Nx, tq_rel, 'b-');
set(P, 'LineWidth', 2);
xlabel('Nq');
ylabel('sec');
title('Varying Nq');

subplot(2,2,3);
P = plot(Nx, tt_rel, 'b-');
set(P, 'LineWidth', 2);
xlabel('Nt');
ylabel('sec');
title('Varying Nt');

subplot(2,2,4);
P = plot(Nx, tsim_rel, 'b-');
set(P, 'LineWidth', 2);
xlabel('Nsim');
ylabel('sec');
title('Varying Nsim');

orient landscape;
saveas(figure(3), 'timings_pergrid.pdf');

figure(4);
subplot(2,2,1);
P = plot(Nx, tx_cpu, 'b-');
set(P, 'LineWidth', 2);
xlabel('Nx');
ylabel('sec');
title('Varying Nx');

subplot(2,2,2);
P = plot(Nx, tq_cpu, 'b-');
set(P, 'LineWidth', 2);
xlabel('Nq');
ylabel('sec');
title('Varying Nq');

subplot(2,2,3);
P = plot(Nx, tt_cpu, 'b-');
set(P, 'LineWidth', 2);
xlabel('Nt');
ylabel('sec');
title('Varying Nt');

subplot(2,2,4);

P = plot(Nx, tsim_cpu, 'b-');
set(P, 'LineWidth', 2);
xlabel('Nsim');
ylabel('sec');
title('Varying Nsim');

orient landscape;
saveas(figure(4), 'timings_cpu.pdf');

figure(5);

subplot(2,2,1);
P = plot(Nx, tx_cpu_rel, 'b-');
set(P, 'LineWidth', 2);
xlabel('Nx');
ylabel('sec');
title('Varying Nx');

subplot(2,2,2);
P = plot(Nx, tq_cpu_rel, 'b-');
set(P, 'LineWidth', 2);
xlabel('Nq');
ylabel('sec');
title('Varying Nq');

subplot(2,2,3);
P = plot(Nx, tt_cpu_rel, 'b-');
set(P, 'LineWidth', 2);
xlabel('Nt');
ylabel('sec');
title('Varying Nt');

subplot(2,2,4);
P = plot(Nx, tsim_cpu_rel, 'b-');
set(P, 'LineWidth', 2);
xlabel('Nsim');
ylabel('sec');
title('Varying Nsim');

orient landscape;
saveas(figure(5), 'timings_pergrid_cpu.pdf');

figure(6);

subplot(2,2,1);
P = plot(Nx, tx_cpu./tx, 'b-');
set(P, 'LineWidth', 2);
xlabel('Nx');
ylabel('sec');
title('Varying Nx');

subplot(2,2,2);
P = plot(Nx, tq_cpu./tq, 'b-');
set(P, 'LineWidth', 2);
xlabel('Nq');
ylabel('sec');
title('Varying Nq');

subplot(2,2,3);
P = plot(Nx, tt_cpu./tt, 'b-');
set(P, 'LineWidth', 2);
xlabel('Nt');
ylabel('sec');
title('Varying Nt');

subplot(2,2,4);
P = plot(Nx, tsim_cpu./tsim, 'b-');
set(P, 'LineWidth', 2);
xlabel('Nsim');
ylabel('sec');
title('Varying Nsim');

orient landscape;
saveas(figure(6), 'speedup.pdf');

%% Produce GPU Tables

fprintf('\n \n Timings');
fprintf('\n & $N_x$ & $N_q$ & $N_{sim}$ & $N_t$ \\\\ \n');
for ii = 1:length(Nx)
    fprintf('%d & %4.3f & %4.3f & %4.3f & %4.3f \\\\ \n', Nx(ii), tx(ii), tq(ii), tsim(ii), tt(ii));
end

fprintf('\n Timings per gridpoint');
fprintf('\n & $N_x$ & $N_q$ & $N_{sim}$ & $N_t$ \\\\ \n');
for ii = 1:length(Nx)
    fprintf('%d & %4.3e & %4.3e & %4.3e & %4.3e \\\\ \n', Nx(ii), tx_rel(ii), tq_rel(ii), tsim_rel(ii), tt_rel(ii));
end

%% Produce CPU Tables

fprintf('\n \n CPU Timings');
fprintf('\n & $N_x$ & $N_q$ & $N_{sim}$ & $N_t$ \\\\ \n');
for ii = 1:length(Nx)
    fprintf('%d & %4.3f & %4.3f & %4.3f & %4.3f \\\\ \n', Nx(ii), tx_cpu(ii), tq_cpu(ii), tsim_cpu(ii), tt_cpu(ii));
end

fprintf('\n CPU Timings per gridpoint');
fprintf('\n & $N_x$ & $N_q$ & $N_{sim}$ & $N_t$ \\\\ \n');
for ii = 1:length(Nx)
    fprintf('%d & %4.3e & %4.3e & %4.3e & %4.3e \\\\ \n', Nx(ii), tx_cpu_rel(ii), tq_cpu_rel(ii), tsim_cpu_rel(ii), tt_cpu_rel(ii));
end

%% Speedup

fprintf('\n \n Speedup');
fprintf('\n & $N_x$ & $N_q$ & $N_{sim}$ & $N_t$ \\\\ \n');
for ii = 1:length(Nx)
    fprintf('%d & %4.3f & %4.3f & %4.3f & %4.3f \\\\ \n', Nx(ii), tx_cpu(ii)/tx(ii), tq_cpu(ii)/tq(ii), tsim_cpu(ii)/tsim(ii), tt_cpu(ii)/tt(ii));
end