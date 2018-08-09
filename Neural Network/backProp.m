function [nabla] = backProp (Input_Sample, Output_Sample, Weights_Array, ...
                                    Activation_function, Differential, Regression)

% Function used to compute the Delta of the weights for the NN
% on the current samples

% Input_Sample: Matrix holding the Input of the sample considered
% Output_Sample: Matrix holding the Output of the sample considered
% Weights_array: Current weights of all the NN we are using
% Activation_function: Activation Function that we are using
% Differential: Differential of the Activation function
% Regression: Boolean used to specify if we are doing regression

Size = size(Weights_Array,1);

[Out, Net] = feedForward(Input_Sample, Weights_Array, Activation_function);

nabla = cell(Size,1);

% Nabla on the Output Layer
% We need to eliminate the First differential for the Regression
if Regression == false
    nabla{Size} = (Output_Sample - Out{Size}) .* Differential(Net{Size});
else
    nabla{Size} = (Output_Sample - Net{Size});
end

% Nabla on the Hidden Layers
for i = Size-1:-1:1
    nabla{i} =  nabla{i+1} * Weights_Array{i+1}(1:end-1,:)' .* Differential(Net{i});
end

for i = Size:-1:2
    nabla{i} = Out{i-1}' * nabla{i};
end

nabla{1} = [Input_Sample ones(size(Input_Sample,1),1)]' * nabla{1};



