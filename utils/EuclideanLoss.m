function error = EuclideanLoss (Input, Output, Weights, Function)

% Function used for the regressiont task to compute the euclidean distance 
% between the target and the predicted values

[~, Net] = feedForward(Input, Weights, Function);
diff = abs(Net{end} - Output);
error =mean( sqrt(sum(diff.^2,2)));

