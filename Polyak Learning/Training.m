function [LC, TC, AC, Output] = Training (Input, Output, TestInput, TestOutput, ...
                   parameters, SkipLoss, Regression)

% parameters = Hidden units, Beta, lambda
% SkipLoss =   Boolean value to skip the Loss calculus on the 
%               middle epochs
% Regression = Boolean value to switch from Classification to Regression


HiddenSizes = parameters{1};  % array containing the width of every hidden layer size
Beta = parameters{2};
lambda = parameters{3};
delta = parameters{4};

BetaMax = Beta;
BetaMin = 0.01 * Beta;  % Minimum Learning Rate (since is decreasing)
Tau = 250;              % Number of iteration where Beta decrease

if nargin < 6
    SkipLoss = false;
end

if nargin < 7
    Regression = false;
end

target_Epochs = 500;
batch_Size = size(Input,1);
Epoch_To_Step_Ratio = 2;
miniBatch_Size = floor(batch_Size/Epoch_To_Step_Ratio); %batch
Full_Batch_Rate = 5;
Sample_Size = size(Input,1);
Test_Size = size(TestInput,1);
Random_Scaling = 1;

useIncremental = true;

% Load the Activation Function
Load_Activation_Functions;

% LC = Learning Curve
% TC = Test Curve
% AC = Accuracy Curve

if Regression == false
    LC = ones(target_Epochs,1);
    TC = ones(target_Epochs,1);
    AC = ones(target_Epochs,1);
else
    LC = 1000 .* ones(target_Epochs,1);
    TC = 1000 .* ones(target_Epochs,1);
    AC = 1000 .* ones(target_Epochs,1);
end

Weights_Size = size(HiddenSizes,1) + 1;
% We need 1 matrix for each hidden layer and 1 for the output

Weights_Array = cell(Weights_Size,1);
old_dir = cell(0,0);

Weights_Array{1} = (randn(size(Input,2)+1,HiddenSizes(1))) * Random_Scaling ;

for i = 1 : Weights_Size-2
    Weights_Array{i+1} = (randn(HiddenSizes(i)+1,HiddenSizes(i+1))) * Random_Scaling ;
end

Weights_Array{Weights_Size} = (randn(HiddenSizes(Weights_Size-1)+1,size(Output,2))) * Random_Scaling;

recordF = + Inf;

for epoch = 1: target_Epochs
    
    % Incremental Approach
    if useIncremental == true && ...
        mod(epoch, Epoch_To_Step_Ratio*Full_Batch_Rate) == 2
        selected = randperm(Sample_Size,batch_Size);
        epoch = epoch + Epoch_To_Step_Ratio - 1;
    else
        selected = randperm(Sample_Size,miniBatch_Size);
    end
    
    
    total_nabla = backProp(Input(selected,:), Output(selected,:), ...
                   Weights_Array, Activation, Diff_Activation, Regression);
        
    total_nabla = Linear_Combination(total_nabla, 1/miniBatch_Size);
    
    if size(old_dir,1) == 0
        old_dir = total_nabla;
    end
         
    % Regularization
    nabla_reg = Regularize(Weights_Array,lambda/BetaMax);
    
    total_nabla = Linear_Combination(total_nabla,1, nabla_reg,1);
    
    % Polyak Step:  diff  = old_nabla - total_nabla;
    %               gamma = old_nabla^T * (diff) / norm(diff)
    %               dir   = gamma* total_nabla + (1-gamma)*old_nabla
    diff   = Linear_Combination( old_dir, 1, total_nabla, -1);
    norm   = Norm(diff);
    if norm > 1e-12
        temp  = ApplyFunction(@(a,b) a.*b, old_dir, diff);
        
        dot = 0;
        for m = temp
           dot = dot + sum(sum(m{1}));
        end
        
        gamma  = dot / norm;
        
        if gamma > 1
            dir = total_nabla;
        elseif gamma < 0
            dir = old_dir;
        else
            dir = Linear_Combination( total_nabla, gamma, old_dir, 1 - gamma);
        end
        
    else
        dir = total_nabla;
    end
    
    if Regression
        LC(epoch) =  EuclideanLoss(Input(selected,:), Output(selected,:), Weights_Array, Activation);
    else
        LC(epoch) =  Loss(Input(selected,:), Output(selected,:), Weights_Array, Activation);
    end
    
    % Target Level Estimation
    currentF = LC(epoch) + lambda * Norm(Weights_Array);
    recordF  = min ( recordF, currentF);
    
    % step = beta * (Loss - Best + Delta) / Norm(dir)
    step = Beta * (currentF - recordF + delta)/ Norm(dir);
    
    c = epoch/Tau;
    Beta = max((1-c)*BetaMax + c*BetaMin, BetaMin);
    
    Weights_Array = Linear_Combination( Weights_Array, 1, ...
                                      dir, step);  
    
    old_dir = dir;
        
    if SkipLoss == false
        
        if Regression
            %selectedTest = randperm(Test_Size,floor(Test_Size/3));
            
            LC(epoch) =  EuclideanLoss(Input, Output, Weights_Array, Activation);
            TC(epoch) =  EuclideanLoss(TestInput, TestOutput, Weights_Array, Activation);
        else
            TC(epoch) =  Loss(TestInput, TestOutput, Weights_Array, Activation);
            AC(epoch) = Accuracy_Classification(TestInput, TestOutput,...
                                                Weights_Array, Activation);
        end
    end
end

if SkipLoss
    if Regression
        LC(epoch) =  EuclideanLoss(Input, Output, Weights_Array, Activation);
        TC(epoch) =  EuclideanLoss(TestInput, TestOutput, Weights_Array, Activation);
    else
        LC(epoch) =  Loss(Input, Output, Weights_Array, Activation);
        TC(epoch) =  Loss(TestInput, TestOutput, Weights_Array, Activation);
        AC(epoch) = Accuracy_Classification(TestInput, TestOutput,...
                                                Weights_Array, Activation);
    end
end

if Regression
    [~, Output] = feedForward(TestInput, Weights_Array, Activation);
else
    Output = feedForward(TestInput, Weights_Array, Activation);
end


    
    
    
    
    