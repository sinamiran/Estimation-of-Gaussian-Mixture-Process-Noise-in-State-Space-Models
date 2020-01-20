close all; clear variables; clc;
%% READ ME:

% This file computes dynamic TRF estimates for the simulation example in 
% paper using Recursive Least Squares (RLS), Linear Gaussian SSM (LG_SSM),
% Gaussian Mixture SSM (GM_SSM). The file reproduces Fig. 3 of the paper, 
% and displays the dynamic TRF estimates using the mentioned approaches.

% ATTN: # of EM iterations in the proposed method has been set to 30 which
% suffices to show the convergence of mixture probabilities and means. To
% also observe the convergence of variance components and exactly reproduce
% Fig. 3 of the paper, please set the variable below to 200 instead of 30

EM_iter_GMSSM = 30;

%% loaded variables

load simulation_data;

% env:      speech envelope (filtered (1Hz-8Hz), log, standardized)
% TRF1:     synthetic dynamic TRF for speaker 1
% TRF2:     synthetic dynamic TRF for speaker 2
% y:        noisy one-dimensional observations (auditory neural response)
% y_c:      clean observations
% p_init:   initialization for mixture probabilities in GM_SSM
% mu_init:  initialization for mixture means in GM_SSM
% var_init: initialization for mixture variances in GM_SSM

%% parameters

FS        = 100;  % sampling frequency
TRF_len   = 0.25; % TRF length
win_len   = 0.3;  % window length for each TRF sample
                  % (piece-wise constant approximation to a dynamic TRF) 
z_win_len = 1.5;  % window length in LG_SSM over which TRF dynamics are the same
obs_var   = 1;    % observation noise variance
SNR = 10*log10(mean(y_c.^2)/obs_var);

W = win_len*FS;
W_z = z_win_len*FS;
[T,~] = size(env);
K = T/W;
K_z = T/W_z;

% RLS estimation parameters
t_eff = 2;
gamma = 1e1;                   % L2 regularization parameter
lambda = 1 - W/(t_eff*FS);     % forgetting factor

% LG_SSM estimation parameters
EM_iter_LGSSM = 40;            % iteration number
alpha = 0.999;                 % alpha in state equation 

% GM_SSM estimation parameters
M = 5;                         % number of mixture components
                               % (determined using AIC model selection)
LF = M;                        % # of components in filtering density
LB = M;                        % # of components in backward filter
LS = M;                        % # of components in joint smoothing density

% construct Gaussian kernel for representing TRFs
std_ker = 18e-3;
Sep = 0.05;
D_out = TRF_len*FS;
D = TRF_len/Sep;
t1 = (0:D_out-1)/FS;
mu1 = (0:D-1)*Sep;
G = zeros(D_out,D);
for i = 1:D
    G(:,i) = exp(-((t1-mu1(i)).^2)/(2*std_ker^2));
end

Cov1 = toeplitz([env(1,1) zeros(1,D_out-1)],env(:,1));
Cov2 = toeplitz([env(1,2) zeros(1,D_out-1)],env(:,2));
C = [Cov1'*G Cov2'*G]; % set of envelope vectors s_t for obs equation in SSM

%% computing the true underlying multimodal process noise representation

samp1 = TRF1(:,2:end) - alpha*TRF1(:,1:end-1);
samp2 = TRF2(:,2:end) - alpha*TRF2(:,1:end-1);
samp = [samp1; samp2];

GMM_true = fitgmdist(samp',M,'CovarianceType','diagonal');
p_true = GMM_true.ComponentProportion;
mu_true = GMM_true.mu';
var_true = squeeze( GMM_true.Sigma );

%% compute dynamic TRF estimates

% RLS estimate
TRF_RLS = RLS_TRFest(y,C,W,K,lambda,gamma);

% LGSSM estimate
y_pred_RLS = sum( repelem(TRF_RLS',W,1).*C , 2);
obs_var_RLS = mean((y - y_pred_RLS).^2);
inc = TRF_RLS(:,2:end) - alpha*TRF_RLS(:,1:end-1);
Q0_RLS = mean(inc.^2,2);
m0_RLS = TRF_RLS(:,1);
% m0_RLS = zeros(2*D,1);
Cov0_RLS = eye(2*D)*1e-6;
[TRF_LGSSM,Q_LGSSM,obs_var_LGSSM] = LinearGaussianSSM_TRFest(y,C,W,K,EM_iter_LGSSM,alpha,obs_var_RLS,Q0_RLS,m0_RLS,Cov0_RLS);

% GMSSM estimate
[p_inf,mu_inf,var_inf,obs_var_GMM,TRF_GMSSM] = GaussianMixtureSSM_TRFest(y,C,W,K,W_z,K_z,EM_iter_GMSSM,M,LF,LB,LS,alpha,obs_var_LGSSM(end),st_init,Cov0_RLS,p_init,mu_init,var_init);

%% estimate normalized RMSE wrt the ground truth TRF

RMSE_RLS = sqrt(mean(mean((TRF_RLS-[TRF1; TRF2]).^2)))/sqrt(mean(mean([TRF1; TRF2].^2)));
RMSE_LGSSM = sqrt(mean(mean((TRF_LGSSM-[TRF1; TRF2]).^2)))/sqrt(mean(mean([TRF1; TRF2].^2)));
RMSE_GMSSM = sqrt(mean(mean((TRF_GMSSM-[TRF1; TRF2]).^2)))/sqrt(mean(mean([TRF1; TRF2].^2)));

disp(['SNR:          ',num2str(SNR),' dB'])
disp(['RMSE RLS:     ',num2str(RMSE_RLS)])
disp(['RMSE LGSSM:   ',num2str(RMSE_LGSSM)])
disp(['RMSE GMSSM:   ',num2str(RMSE_GMSSM)])

%% plot results

% plotting the true TRFs as well as the estimated TRFs
lim1 = 0.8*max(max(abs([G*TRF1; G*TRF2])));
figure('Color','W');
subplot(4,2,1)
imagesc((1:K)*W/FS,(0:D-1)*Sep,G*TRF1,[-lim1,lim1]);
colormap('redblue')
title('Speaker 1 True TRF')
subplot(4,2,2)
imagesc((1:K)*W/FS,(0:D-1)*Sep,G*TRF2,[-lim1,lim1]);
colormap('redblue')
title('Speaker 2 True TRF')

subplot(4,2,3)
imagesc((1:K)*W/FS,(0:D-1)*Sep,G*TRF_RLS(1:D,:),[-lim1,lim1]);
colormap('redblue')
title('Speaker 1 Est. TRF (RLS)')
subplot(4,2,4)
imagesc((1:K)*W/FS,(0:D-1)*Sep,G*TRF_RLS(D+1:end,:),[-lim1,lim1]);
colormap('redblue')
title('Speaker 2 Est. TRF (RLS)')

subplot(4,2,5)
imagesc((1:K)*W/FS,(0:D-1)*Sep,G*TRF_LGSSM(1:D,:),[-lim1,lim1]);
colormap('redblue')
title('Speaker 1 Est. TRF (LGSSM)')
subplot(4,2,6)
imagesc((1:K)*W/FS,(0:D-1)*Sep,G*TRF_LGSSM(D+1:end,:),[-lim1,lim1]);
colormap('redblue')
title('Speaker 2 Est. TRF (LGSSM)')

subplot(4,2,7)
imagesc((1:K)*W/FS,(0:D-1)*Sep,G*TRF_GMSSM(1:D,:),[-lim1,lim1]);
colormap('redblue')
title('Speaker 1 Est. TRF (GMSSM)')
xlabel('time (s)')
subplot(4,2,8)
imagesc((1:K)*W/FS,(0:D-1)*Sep,G*TRF_GMSSM(D+1:end,:),[-lim1,lim1]);
colormap('redblue')
title('Speaker 2 Est. TRF (GMSSM)')
xlabel('time (s)')

% plotting the convergence of mixture parameters 
% (for M100 component of speaker 2 as an example)
figure('Color','W')
subplot(3,2,1)
hold on
for m = 1:M
    plot(0:EM_iter_GMSSM,p_inf(m,1:EM_iter_GMSSM+1),'LineWidth',1.13)
end
for m = 1:M
    plot(0:EM_iter_GMSSM,p_true(m)*ones(1,EM_iter_GMSSM+1),'--k')
end
hold off;
title('Conv. of Mixture Probabilities')

subplot(3,2,2)
hold on;
for m = 1:M
    plot(0:EM_iter_GMSSM,squeeze(mu_inf(8,m,1:EM_iter_GMSSM+1)),'LineWidth',1.13);
end
for m = 1:M
    plot(0:EM_iter_GMSSM,mu_true(8,m)*ones(1,EM_iter_GMSSM+1),'--k');
end
hold off;
title('Conv. of Mixture Means (M100 of Sp. 2)')

subplot(3,2,[3 4])
semilogy(0:EM_iter_GMSSM,squeeze(var_inf(8,1,1:EM_iter_GMSSM+1)),'LineWidth',1.13);
hold on;
for m = 2:M
    semilogy(0:EM_iter_GMSSM,squeeze(var_inf(8,m,1:EM_iter_GMSSM+1)),'LineWidth',1.13);
end
for m = 1:M
    semilogy(0:EM_iter_GMSSM,var_true(8,m)*ones(1,EM_iter_GMSSM+1),'--k');
end
hold off;
legend('Comp. 1','Comp. 2','Comp. 3','Comp. 4','Comp. 5','Location','best')
title('Conv. of Mixture Variances (M100 of Sp. 2)')
grid on;

subplot(3,2,[5 6])
ind = linspace(-0.05,0.05,300);
pdf_oracle = zeros(size(ind));
for m = 1:M
    pdf_oracle = pdf_oracle + p_true(m)*exp(-(ind-mu_true(8,m)).^2/(2*var_true(8,m)))/sqrt(2*pi*var_true(8,m));
end
pdf_est = zeros(size(ind));
for m = 1:M
    pdf_est = pdf_est + p_inf(m,EM_iter_GMSSM+1)*exp(-(ind-mu_inf(8,m,EM_iter_GMSSM+1)).^2/(2*var_inf(8,m,EM_iter_GMSSM+1)))/sqrt(2*pi*var_inf(8,m,EM_iter_GMSSM+1));
end
pdf_init = zeros(size(ind));
for m = 1:M
    pdf_init = pdf_init + p_inf(m,1)*exp(-(ind-mu_inf(8,m,1)).^2/(2*var_inf(8,m,1)))/sqrt(2*pi*var_inf(8,m,1));
end

hold on;
plot(ind,pdf_est,'LineWidth',1.13)
plot(ind,pdf_oracle,'LineWidth',1.13)
plot(ind,pdf_init,'LineWidth',1.13)
hold off;
title('Process Noise Density')
legend('Est. PDF (GMSSM)','oracle PDF','init PDF (from LGSSM)','Location','best')
