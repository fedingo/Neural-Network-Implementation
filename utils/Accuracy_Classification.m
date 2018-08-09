function R = Accuracy_Classification(Input, Output, Weights, Function)

% Function used to calculate the accuracy of the model

[Out, ~] = feedForward(Input, Weights, Function);
Classified = Out{end} > 0;
Classified = Classified*2 -1;
R = sum((Classified == Output)) / size(Input,1);

