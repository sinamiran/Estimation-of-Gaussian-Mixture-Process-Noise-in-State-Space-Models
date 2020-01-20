function s = myLogDet(A,i)

    if i == 1
        s = 2 * sum(log(diag(chol(A))));
    else
        [~, U, P] = lu(A);
        du = diag(U);
        c = det(P) * prod(sign(du));
        s = log(c) + sum(log(abs(du)));
    end

end