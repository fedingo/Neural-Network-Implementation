function error = Loss (Input, Output, Weights, Function)

% Function to compute the MSE

[Out, ~] = feedForward(Input, Weights, Function);
diff = abs(Out{end} - Output);
error = mean(diff.^2);
% error = sum(diff.^2) / size(Input,1);
