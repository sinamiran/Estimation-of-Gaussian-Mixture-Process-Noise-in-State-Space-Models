%% TRF estimation using SSM with Gaussian mixture process noise (GMSSM)

% inputs:
% y: observation vector
% C: concatination of vectors in the observation equation of SSM
% W: window length in samples for piece-wise constant approximation of TRF
% K: number of windows
% EM_iter: number of EM iterations
% alpha: constant in the state equation
% obs_var0: initialization for the observation noise variance
% Q0: initialization of Gaussian process noise covariance
% m0: TRF initialization at time 0
% Cov0: covariance initialization of TRF at time 0

% outputs:
% TRF_est: estimated TRF at the final iteration as the smoothed states
% Q: estimated process noise covariance at each iteration
% obs_var: estimated observation noise variance in each iteration

%%

function [TRF_est,Q,obs_var] = LinearGaussianSSM_TRFest(y,C,W,K,EM_iter,alpha,obs_var0,Q0,m0,Cov0)

[T,D] = size(C);

Q = zeros(D,EM_iter+1);
Q(:,1) = Q0;

x_init = m0;
V_init = Cov0;
A = alpha*eye(D);

obs_var = zeros(1,EM_iter+1);
obs_var(1) = obs_var0;

for i = 1:EM_iter
    
    x_filt = zeros(D,K);
    x_pred = zeros(D,K);
    V_filt = zeros(D,D,K);
    V_pred = zeros(D,D,K);
    
    for k = 1:K
        
        cov = C((k-1)*W+1:k*W,:);
        if k == 1
            x_pred(:,k) = x_init;
            V_pred(:,:,k) = V_init;
        else
            x_pred(:,k) = A*x_filt(:,k-1);
            V_pred(:,:,k) = A*V_filt(:,:,k-1)*A' + diag(Q(:,i));
        end
        gain = (V_pred(:,:,k)*cov')/(cov*V_pred(:,:,k)*cov' + obs_var(i)*eye(W));
        x_filt(:,k) = x_pred(:,k) + gain*(y((k-1)*W+1:k*W)-cov*x_pred(:,k));
        V_filt(:,:,k) = (eye(D) - gain*cov)*V_pred(:,:,k);
        
    end
    
    x_smooth = zeros(D,K);
    x_smooth(:,K) = x_filt(:,K);
    V_smooth = zeros(D,D,K);
    V_smooth(:,:,K) = V_filt(:,:,K);
    P_smooth = zeros(D,D,K);
    P_smooth(:,:,K) = V_smooth(:,:,K) + x_smooth(:,K)*x_smooth(:,K)';
    V_cov = zeros(D,D,K-1);
    P_cov = zeros(D,D,K-1);
    J = zeros(D,D);
    
    for k = K-1:-1:1
        
        J1 = J;
        J = (V_filt(:,:,k)*A')/V_pred(:,:,k+1);
        x_smooth(:,k) = x_filt(:,k) + J*(x_smooth(:,k+1)-A*x_filt(:,k));
        V_smooth(:,:,k) = V_filt(:,:,k) + J*( V_smooth(:,:,k+1) - V_pred(:,:,k+1) )*J';
        P_smooth(:,:,k) = V_smooth(:,:,k) + x_smooth(:,k)*x_smooth(:,k)';
        if k == K-1
            V_cov(:,:,K-1) = (eye(D) - gain*cov)*A*V_filt(:,:,K-1);
        else
            V_cov(:,:,k) = V_filt(:,:,k+1)*J' + J1*( V_cov(:,:,k+1) - A*V_filt(:,:,k+1) )*J';
        end
        P_cov(:,:,k) = V_cov(:,:,k) + x_smooth(:,k+1)*x_smooth(:,k)';
        
    end
    
    x_init = x_smooth(:,1);
    V_init = V_smooth(:,:,1);
    
    tmp = diag( sum(P_smooth(:,:,2:end),3) - A*sum(P_cov,3)' - sum(P_cov,3)*A' + A*sum(P_smooth(:,:,1:end-1),3)*A' );
    Q(:,i+1) = tmp./(K-1);
    
    x_temp = repelem(x_smooth,1,W);
    tempp1 = sum( (y - 2*sum(C.*x_temp',2)).*y );
    tempp2 = 0;
    for t = 1:T
        tt = ceil(t/W);
        tempp2 = tempp2 + C(t,:)*P_smooth(:,:,tt)*C(t,:)';
    end
    obs_var(i+1) = (tempp1+tempp2)/T;
    
end

TRF_est = x_smooth;


end