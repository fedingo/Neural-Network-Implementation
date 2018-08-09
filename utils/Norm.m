function [normNabla] = Norm(nabla)

% Function used to calculate the norm of a set of matrices

    normNabla = 0;
    Size = size(nabla,1);
    
    for i = 1 : Size
        normNabla = normNabla + norm(nabla{i});
    end