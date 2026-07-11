function [W1, W2, W3, b1, b2, b3, vel_W1, vel_W2, vel_W3, vel_b1, vel_b2, vel_b3] = ...
    InitializeWeights(numHidden1, numHidden2, N, M)

    W1 = randn(numHidden1,N) * sqrt(2/N);
    W2 = randn(numHidden2,numHidden1) * sqrt(2/numHidden1);
    W3 = randn(M,numHidden2) * sqrt(2/numHidden2);

    b1 = zeros(numHidden1,1);
    b2 = zeros(numHidden2,1);
    b3 = zeros(M,1);
    
    vel_W1 = zeros(size(W1));
    vel_W2 = zeros(size(W2));
    vel_W3 = zeros(size(W3));
    
    vel_b1 = zeros(size(b1));
    vel_b2 = zeros(size(b2));
    vel_b3 = zeros(size(b3));

end