%% RLS estimation of TRFs

% inputs:
% y: observation vector
% C: concatination of vectors in the observation equation of SSM
% W: window length in samples for piece-wise constant approximation of TRF
% K: number of windows
% lambda: forgetting factor
% gamma: L2 regularization penalty

% outputs:
% TRF: estimated TRF using RLS

%%

function TRF = RLS_TRFest(y,C,W,K,lambda,gamma)

[~,D] = size(C);
TRF = zeros(D,K);
A = zeros(D,D);
b = zeros(1,D);
for k = 1:K

    A = lambda*A + C((k-1)*W+1:k*W,:)'*C((k-1)*W+1:k*W,:);
    b = lambda*b + y((k-1)*W+1:k*W)'*C((k-1)*W+1:k*W,:);
    TRF(:,k) = (A + gamma*eye(D))\b';
   
end

end