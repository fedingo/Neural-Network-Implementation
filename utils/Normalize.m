function [Input, Output] = Normalize( Matrix )

% Function to normalize the input with a 1-of-k encoding

samples = size(Matrix(:,1));
Input = [];
Output = Matrix(:,1);
for j = 2:size(Matrix,2)
   
    temp = unique(Matrix(:,j));
    values = size(temp,1);
    T = zeros(samples);
    for k = 1: values
        T(:,k) = Matrix(:,j) == temp(k)*ones(samples);
    end
    
    Input = [Input T];
end