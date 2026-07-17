function [W1, W2, W3, b1, b2, b3] = SubgradientUpdateWeights(W1, W2, W3, b1, b2, b3, alpha, d_curr)
    curr = 0;

    nElem = numel(W1);
    W1 = W1 - alpha * reshape(d_curr(curr + (1:nElem)), size(W1));
    curr = curr + nElem;
            
    nElem = numel(b1);
    b1 = b1 - alpha * reshape(d_curr(curr + (1:nElem)), size(b1));
    curr = curr + nElem;
            
    nElem = numel(W2);
    W2 = W2 - alpha * reshape(d_curr(curr + (1:nElem)), size(W2));
    curr = curr + nElem;
            
    nElem = numel(b2);
    b2 = b2 - alpha * reshape(d_curr(curr + (1:nElem)), size(b2));
    curr = curr + nElem;
            
    nElem = numel(W3);
    W3 = W3 - alpha * reshape(d_curr(curr + (1:nElem)), size(W3));
    curr = curr + nElem;
            
    nElem = numel(b3);
    b3 = b3 - alpha * reshape(d_curr(curr + (1:nElem)), size(b3));
end