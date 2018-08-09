function [Out, Net] = feedForward(Input_Sample, Weights_Array, Activation_function)

% function to perform the Feed foorward step

% Input_Sample: Matrix holding the Input of the sample considered
% Weights_array: Current weights of all the NN we are using
% Activation_function: Activation Function that we are using

Size = size(Weights_Array,1);

Net = cell(Size,1);
Out = cell(Size,1);

Input_Sample = [ Input_Sample ones(size(Input_Sample,1),1) ];
Net{1} = Input_Sample * Weights_Array{1};
Out{1} = Activation_function(Net{1});

for i = 2 : Size
    
    Out{i-1} = [ Out{i-1} ones(size(Out{i-1},1),1) ];
    Net{i} = Out{i-1} * Weights_Array{i};
    Out{i} = Activation_function(Net{i});
end