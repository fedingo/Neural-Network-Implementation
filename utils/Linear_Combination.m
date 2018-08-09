function result = Linear_Combination (A, coeffA, B, coeffB)
    
% Function used to compute a linear combination between two arrays of
% matrices, combined with two coefficients

    if nargin < 3 ||  size(B,1) == 0
        result = cell(size(A,1),1);
        for i = 1: size(A)
            result{i} = A{i} * coeffA; 
        end
    else
        
        result = cell(size(A,1),1);
        for i = 1: size(A)
            result{i} = A{i} * coeffA + B{i} * coeffB; 
        end
        
    end