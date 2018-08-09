function [result] = AutoGrid_Model_Selection (Input, Output, K, Ranges, regression)

% K = number of folds for Cross-validation
% Ranges = ranges where to search for each hyper-parameter
% Regression = boolean variable

if nargin < 5
    regression = false;
end

BestGuyError = 100;    % best error found up to now
BestGuy = cell(0,0);   % best configuration of parameters up to now


fprintf("Starting... \n");
tic;

keep_iterating = true;
iteration = 0;

while (keep_iterating)
    
    iteration = iteration +1;
    bestCurrentError = Inf;
    Tot_Errors = 0;

    for hidden = 1:2
    for eta = 1:2
    for lambda = 1:2
    for alpha = 1:2
        
        currentGuy = {Ranges(1,hidden), ...
                        Ranges(2,eta), ...
                        Ranges(3,lambda), ...
                        Ranges(4,alpha)};
                             
        My_Error = CrossValidation(Input, Output, K, currentGuy, regression);
        
        Tot_Errors = Tot_Errors + My_Error/16;
        
        if bestCurrentError > My_Error
            
           bestCurrentError = My_Error;
           guy = currentGuy;
        end

    end
    end
    end
    end
    
    fprintf( "Iteration " + iteration + ": "+ ...
             "Min: " + bestCurrentError + ", "+...
             "Mean: " + Tot_Errors + ", " +...
             "Time Taken: " + toc + "\n" ...
         );
    
    if guy{1} == Ranges(1,2) 
        Ranges(1,1) = floor( (Ranges(1,1) + Ranges (1,2) )/2) ;
    else
        Ranges(1,2) = floor( (Ranges(1,1) + Ranges (1,2) )/2) ;
    end
    
    % Learning Rate Range Update
    % newEta = exp( (log(Ranges(2,1)) + log(Ranges (2,2)))/2 );
    newEta = (Ranges(2,1) + Ranges (2,2) )/2;
    
    if guy{2} == Ranges(2,2) 
        Ranges(2,1) = newEta;
    else
        Ranges(2,2) = newEta;
    end
    
    % Regularization Range Update
    if guy{3} == Ranges(3,2) 
        Ranges(3,1) = (Ranges(3,1) + Ranges (3,2) )/2;
    else
        Ranges(3,2) = (Ranges(3,1) + Ranges (3,2) )/2;
    end
    
    % Momentum Range Update
    if guy{4} == Ranges(4,2) 
        Ranges(4,1) = (Ranges(4,1) + Ranges (4,2) )/2;
    else
        Ranges(4,2) = (Ranges(4,1) + Ranges (4,2) )/2;
    end
    
    if BestGuyError > bestCurrentError
       BestGuyError = bestCurrentError;
       BestGuy = guy;
    end
    
    if iteration == 10
        keep_iterating = false;
    end
end

Random_print(BestGuy,BestGuyError);

result = BestGuy;

