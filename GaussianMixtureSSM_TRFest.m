%% TRF estimation using SSM with Gaussian mixture process noise (GMSSM)

% inputs:
% y: observation vector
% C: concatination of vectors in the observation equation of SSM
% W: window length in samples for piece-wise constant approximation of TRF
% K: number of windows
% W1: window length in samples where TRF dynamics are similar
% K1: number of windows in which TRF dynamics are similar
% EM_iter: number of EM iterations
% M: number of mixture components
% LF: number of mixture components kept in the filtering density
% LB: number of mixture components kept in the backward filter
% LS: number of mixture components in the joint smoothing densities
% alpha: constant in the state equation
% obs_var0: initialization for the observation noise variance
% m0: TRF initialization at time 0
% Cov0: covariance initialization of TRF at time 0
% p_init: initialization for mixture probabilities
% mu_init: initialization for mixture means
% var_init: initialization for mixture variances

% outputs:
% p_inf: estimated misture probabilities in each iteration
% mu_inf: estimated mixture means in each iteration
% var_inf: estimated mixture variances in each iteration
% obs_var: estimated observation noise variance in each iteration
% TRF_est: estimated TRF at the final iteration as the smoothed states


%%

function [p_inf,mu_inf,var_inf,obs_var,TRF_est] = GaussianMixtureSSM_TRFest(y,C,W,K,W1,K1,EM_iter,M,LF,LB,LS,alpha,obs_var0,m0,Cov0,p_init,mu_init,var_init)

[T,D] = size(C);
Win = W1/W;
A = alpha*eye(D);

p_inf = zeros(M,EM_iter+1);
mu_inf = zeros(D,M,EM_iter+1);
var_inf = zeros(D,M,EM_iter+1);

obs_var = zeros(1,EM_iter+1);
obs_var(1) = obs_var0;

% initialization
p_inf(:,1) = p_init;
mu_inf(:,:,1) = mu_init;
var_inf(:,:,1) = var_init;

% EM inference

epi = 1e-4;
p1 = zeros(LF,1);
p1(1) = log(1 - (LF-1)*epi);
p1(2:LF) = log(epi);
init1 = repmat(m0,1,LF);
cov1 = repmat(Cov0,1,1,LF);

for i1 = 1:EM_iter
    
    P_smooth = zeros(D,D,K);
    
    p_M = p_inf(:,i1);
    mu_M = mu_inf(:,:,i1);
    var_M = var_inf(:,:,i1);
    
    % filtering densities
    
    p_f_init_log = zeros(LF,K1);
    mu_f_init = zeros(D,LF,K1);
    cov_f_init = zeros(D,D,LF,K1);

    p_f_init_log(:,1) = p1;
    mu_f_init(:,:,1) = init1;
    cov_f_init(:,:,:,1) = cov1;
    
    p_f_log = zeros(LF*M,Win+1,K1);
    p_f = zeros(LF*M,Win+1,K1);
    mu_f = zeros(D,LF*M,Win+1,K1);
    cov_f = zeros(D,D,LF*M,Win+1,K1);
    
    for k = 1:K1
        
        for lf = 1:LF
            
            for m = 1:M
                
                p1_log = zeros(1,Win+1);
                mu1 = zeros(D,Win+1);
                cov1 = zeros(D,D,Win+1);
                
                i5 = (lf-1)*M+m;
                p1_log(1) = log(p_M(m)) + p_f_init_log(lf,k);
                mu1(:,1) = mu_f_init(:,lf,k);
                cov1(:,:,1) = cov_f_init(:,:,lf,k);
                
                for w = 1:Win
                    
                    inds = (k-1)*Win*W + (w-1)*W;
                    Covv1 = C(inds+1:inds+W,:);
                    
                    temp1 = A*mu1(:,w) + mu_M(:,m);
                    temp2 = A*cov1(:,:,w)*A' + diag(var_M(:,m));
                    temp3 = Covv1*temp2*Covv1'+obs_var(i1)*eye(W);
                    temp4 = y(inds+1:inds+W)-Covv1*temp1;
                    gain = (temp2*Covv1')/temp3;
                    mu1(:,w+1) = temp1 + gain*temp4;
                    cov1(:,:,w+1) = temp2 - gain*Covv1*temp2;
                    p1_log(w+1) = p1_log(w) -0.5*(temp4'/temp3)*temp4 -0.5*myLogDet(temp3,1);

                end
                
                p_f_log(i5,:,k) = p1_log;
                mu_f(:,i5,:,k) = mu1;
                cov_f(:,:,i5,:,k) = cov1;

            end
            
        end
        
        p_f_log(:,:,k) = p_f_log(:,:,k) - repmat(max(p_f_log(:,:,k),[],1),LF*M,1);
        temp = exp(p_f_log(:,:,k));
        p_f(:,:,k) = temp ./ repmat( sum(temp,1),LF*M,1 );
        
        if k<K1
            [val,ind] = sort(p_f_log(:,Win+1,k),'descend');
            p_f_init_log(:,k+1) = val(1:LF);
            mu_f_init(:,:,k+1) = mu_f(:,ind(1:LF),Win+1,k);
            cov_f_init(:,:,:,k+1) = cov_f(:,:,ind(1:LF),Win+1,k);
        end
        
        
    end
    
    % backward propagation and smoothing

    log_alp = zeros(LB*M,Win+1,K1);
    B = zeros(D,D,LB*M,Win+1,K1);
    b = zeros(D,LB*M,Win+1,K1);
    
    log_p_s_all = zeros(LB*LF*M,K);
    mu_s_all = zeros(2*D,LB*LF*M,K);
    cov_s_all = zeros(2*D,2*D,LB*LF*M,K);
    
    p_s = zeros(LS,K);
    mu_s = zeros(2*D,LS,K);
    cov_s = zeros(2*D,2*D,LS,K);
    
    TRF_est = zeros(D,K);
    
    u = zeros(D,K);
    U = zeros(D,D,K);
    U_diag = zeros(D,K);

    log_alp_init = zeros(LB,K1);
    B_init = zeros(D,D,LB,K1);
    b_init = zeros(D,LB,K1);
    
    B_init(:,:,1:LB,K1) = repmat(C(T-W+1:T,:)'*C(T-W+1:T,:)/obs_var(i1),1,1,LB);
    b_init(:,1:LB,K1) = repmat(C(T-W+1:T,:)'*y(T-W+1:T)/obs_var(i1),1,LB);
    log_alp_init(1,K1) = log(1-epi*(LB-1));
    log_alp_init(2:LB,K1) = log(epi);
    
    for k = K1:-1:1

        % backward filters
        for l1 = 1:LB

            for m = 1:M

                V = diag(var_M(:,m));
                VI = diag(1./var_M(:,m));
                B1 = zeros(D,D,Win+1);
                b1 = zeros(D,Win+1);
                log_alp1 = zeros(1,Win+1);
                B1(:,:,Win+1) = B_init(:,:,l1,k);
                b1(:,Win+1) = b_init(:,l1,k);
                log_alp1(Win+1) = log_alp_init(l1,k);

                for w = Win:-1:1

                    i = (k-1)*Win+w-1;
                    
                    if i == 0
                        break;
                    end
                    
                    Covv1 = C((i-1)*W+1:i*W,:);
                    
                    B1(:,:,w) = Covv1'*Covv1/obs_var(i1) + A'*VI*A - (A'/(V+V*B1(:,:,w+1)*V))*A;
                    b1(:,w) = Covv1'*y((i-1)*W+1:i*W)/obs_var(i1) - A'*VI*mu_M(:,m) + A'*((V+V*B1(:,:,w+1)*V)\(V*b1(:,w+1)+mu_M(:,m)));
                    log_alp1(w) = log_alp1(w+1) - 0.5*myLogDet(eye(D)+V*B1(:,:,w+1),2) - 0.5*mu_M(:,m)'*VI*mu_M(:,m) + 0.5*((V*b1(:,w+1)+mu_M(:,m))'/(V*B1(:,:,w+1)*V+V))*(V*b1(:,w+1)+mu_M(:,m));
                    
                end

                if k>1
                    B(:,:,(l1-1)*M+m,:,k) = B1;
                    b(:,(l1-1)*M+m,:,k) = b1;
                    log_alp((l1-1)*M+m,:,k) = log_alp1;
                else
                    B(:,:,(l1-1)*M+m,2:Win+1,1) = B1(:,:,2:Win+1);
                    b(:,(l1-1)*M+m,2:Win+1,1) = b1(:,2:Win+1);
                    log_alp((l1-1)*M+m,2:Win+1,1) = log_alp1(2:Win+1);
                end

            end

        end

        % normalization
        log_alp(:,:,k) = log_alp(:,:,k) - repmat(max(log_alp(:,:,k),[],1),LB*M,1);
        
        % smoothing
        S12 = zeros(D,D,M);
        for m = 1:M
            S12(:,:,m) = - A'*diag(1./var_M(:,m));
        end

        tmp1 = zeros(D,M);
        for m = 1:M
            tmp1(:,m) = diag(1./var_M(:,m))*mu_M(:,m);
        end
    
        for w = Win:-1:1
        
            i = (k-1)*Win + w;
            log_p_s = zeros(LF*LB*M,1);
            
            for lf = 1:LF
                
                for lb = 1:LB                
                
                    for m = 1:M
                        
                        ilf = (lf-1)*M+m;
                        cov_f_inv = cov_f(:,:,ilf,w,k)\eye(D);
                        tmp2 = cov_f_inv * mu_f(:,ilf,w,k);
                        
                        ilb = (lb-1)*M+m;
                        
                        i2 = (lf-1)*LB*M + (lb-1)*M + m;
                        
                        S = zeros(2*D,2*D);
                        S(1:D,1:D) = cov_f_inv - S12(:,:,m)*A;
                        S(1:D,D+1:2*D) = S12(:,:,m);
                        S(D+1:2*D,1:D) = S12(:,:,m)';
                        S(D+1:2*D,D+1:2*D) = B(:,:,ilb,w+1,k) + diag(1./var_M(:,m));
                        
                        Sigma = S\eye(2*D);
                        vec = zeros(2*D,1);
                        vec(1:D) = tmp2 - A'*tmp1(:,m);
                        vec(D+1:end) = tmp1(:,m) + b(:,ilb,w+1,k); 
                        mu = Sigma*vec;
                        log_p_s(i2) = - 0.5*mu_f(:,ilf,w,k)'*tmp2 - 0.5*mu_M(:,m)'*tmp1(:,m) + 0.5*mu'*S*mu + log_alp(ilb,w+1,k) + log(p_f(ilf,w,k)) - 0.5*myLogDet(cov_f(:,:,ilf,w,k),1) - 0.5*sum(log(var_M(:,m))) + 0.5*myLogDet(Sigma,1);
                        
                        mu_s_all(:,i2,i) = mu;
                        cov_s_all(:,:,i2,i) = Sigma;
                        
                    end
                
                end
            
            end
            
            log_p_s = log_p_s - max(log_p_s);
            
            [vals,inds] = sort(log_p_s,'descend');
            p_s(:,i) = exp(vals(1:LS))/sum(exp(vals(1:LS)));
            mu_s(:,:,i) = mu_s_all(:,inds(1:LS),i);
            cov_s(:,:,:,i) = cov_s_all(:,:,inds(1:LS),i);
            
            TRF_est(:,i) = mu_s(D+1:2*D,:,i)*p_s(:,i);
            
            for ls = 1:LS
                
                U(:,:,i) = U(:,:,i) + p_s(ls,i)*[-A, eye(D)]*( cov_s(:,:,ls,i) + mu_s(:,ls,i)*mu_s(:,ls,i)' )*[-A'; eye(D)];
                u(:,i) = u(:,i) + p_s(ls,i)*[-A, eye(D)]*mu_s(:,ls,i);
                
                P_smooth(:,:,i) = P_smooth(:,:,i) + p_s(ls,i)*( cov_s(D+1:2*D,D+1:2*D,ls,i) + mu_s(D+1:2*D,ls,i)*mu_s(D+1:2*D,ls,i)' );
                
            end
            
            U_diag(:,i) = diag(U(:,:,i));
        
        end

        indss = inds - floor((inds-1)/(LB*M))*LB*M;
        log_alp(:,1,k) = log_alp(:,1,k) + repmat(log(p_M),LB,1);
        if k>1 
            for lb = 1:LB
                
                log_alp_init(lb,k-1) = log_alp(indss(1),1,k);
                B_init(:,:,lb,k-1) = B(:,:,indss(1),1,k);
                b_init(:,lb,k-1) = b(:,indss(1),1,k);
                
                indss(indss==indss(1)) = [];
            end
            log_alp_init(:,k-1) = log_alp_init(:,k-1) - max(log_alp_init(:,k-1));
        end

    end
    
    if LF<=LS
        p1 = log(p_s(1:LF,1));
        init1 = mu_s(D+1:2*D,1:LF,1);
        cov1 = cov_s(D+1:2*D,D+1:2*D,1:LF,1);
    else
        p1(1:LS) = log(p_s(:,1));
        init1(:,1:LS) = mu_s(D+1:2*D,:,1);
        cov1(:,:,1:LS) = cov_s(D+1:2*D,D+1:2*D,:,1);
        p1(LS+1:LF) = log(epi);
        init1(:,LS+1:LF) = mu_s(D+1:2*D,1,1);
        cov1(:,:,LS+1:LF) = cov_s(D+1:2*D,D+1:2*D,1,1);
        p1(1) = log(p_s(1,1) - (LF-LS)*epi);
    end
    
    
    % M-step
    
    % approximating epsilons
    
    diff2 = u;
    
    epsi_log = zeros(K1,M);
    temp = log(p_M)' - 0.5*Win*sum(log(var_M),1);

    p_log1 = zeros(M,K);
    for m = 1:M
        diff3 = diff2 - repmat(mu_M(:,m),1,K);
        p_log1(m,:) = - 0.5*sum((diff3.^2)./repmat(var_M(:,m),1,K),1);
    end
    for k = 1:K1
        epsi_log(k,:) = temp + sum(p_log1(:,(k-1)*Win+1:k*Win),2)';
    end
    epsi_log = epsi_log - repmat(max(epsi_log,[],2),1,M);
    epsii = exp(epsi_log)./repmat(sum(exp(epsi_log),2),1,M);
    epsii = epsii';
    
    p_inf(:,i1+1) = sum(epsii,2)/K1;
    
    temp1 = repelem(epsii,1,Win);
    for m = 1:M
        mu_inf(:,m,i1+1) = (u*temp1(m,:)')/(Win*K1*p_inf(m,i1+1));
    end
    
    for m = 1:M
        var_inf(:,m,i1+1) = (U_diag*temp1(m,:)' - mu_inf(:,m,i1+1).^2*Win*K1*p_inf(m,i1+1) )./(Win*K1*p_inf(m,i1+1));
    end
    
    
    x_temp = repelem(TRF_est,1,W);
    tempp1 = sum( (y - 2*sum(C.*x_temp',2)).*y );
    tempp2 = 0;
    for t = 1:T
        tt = ceil(t/W);
        tempp2 = tempp2 + C(t,:)*P_smooth(:,:,tt)*C(t,:)';
    end
    obs_var(i1+1) = (tempp1+tempp2)/T;
    
    iteration_number = i1;
    iteration_number
    
end

end