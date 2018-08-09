function [LC, TC, AC, Output] = Training (Input, Output, TestInput, TestOutput, ...
                   parameters, SkipLoss, Regression)

% parameters = HiddenSizes, eta, lambda, alpha
% SkipLoss =   Boolean value to skip the Loss calculus on the 
%               middle epochs
% Regression = Boolean value to switch from Classification to Regression

HiddenSizes = parameters{1}; % array containing the width of every hidden layer size
eta = parameters{2};         % Learning Rate
lambda = parameters{3};      % Regularization
alpha = parameters{4};       % Momentum

eta_max = eta;
eta_min = 0.01*eta;
tau = 500;

if nargin < 6
    SkipLoss = false;
end
if nargin < 7
    Regression = false;
end
%tic;

target_Epochs = 500;                     % number of epochs
miniBatch_Size = floor(size(Input,1));   % number of sample for the minibatch
Sample_Size = size(Input,1);
Test_Size = size(TestInput,1);

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

Random_Scaling = .7;

Load_Activation_Functions;

Weights_Size = size(HiddenSizes,1) + 1;
% We need 1 matrix for each hidden layer and 1 for the output

Weights_Array = cell(Weights_Size,1);
old_nabla = cell(0,0);

Weights_Array{1} = (rand(size(Input,2)+1,HiddenSizes(1))-0.5) * Random_Scaling ;

for i = 1 : Weights_Size-2
    Weights_Array{i+1} = (rand(HiddenSizes(i)+1,HiddenSizes(i+1))-0.5) * Random_Scaling ;
end

Weights_Array{Weights_Size} = (rand(HiddenSizes(Weights_Size-1)+1,size(Output,2))-0.5) * Random_Scaling ;

nWeights = 0;
for index = 1 : Weights_Size
    el = Weights_Array{index};
    nWeights = nWeights + size(el,1)*size(el,2);
end


for epoch = 1 : target_Epochs
    selected = randperm(Sample_Size,miniBatch_Size);
    
    total_nabla = backProp(Input(selected,:), Output(selected,:), ...
                   Weights_Array, Activation, Diff_Activation, Regression);
               
    normNabla = 0;
    for i = 1 : Weights_Size
        normNabla = normNabla + norm(total_nabla{i});
    end
    
    % Regularization
    nabla_reg = Regularize(Weights_Array,lambda*(eta/eta_max));
    
    Weights_Array = Linear_Combination (Weights_Array, 1, nabla_reg, 1);
    
    % Momentum: nabla = eta* nabla + alpha*old_nabla
    total_nabla = Linear_Combination( total_nabla, eta/miniBatch_Size, ...
                                      old_nabla, alpha);  
    
    % Gradient Descent Step
    Weights_Array = Linear_Combination( Weights_Array, 1, total_nabla, 1); 
    
    old_nabla = total_nabla;
    
    if 1 == 1
        a = epoch / tau;
        eta = eta_max * (1-a) + eta_min * a;
        eta = max( eta, eta_min);
    end
    
    if SkipLoss == false
        if Regression
            selected = randperm(Sample_Size);
            selectedTest = randperm(Test_Size,floor(Test_Size));
            
            LC(epoch) =  EuclideanLoss(Input(selected,:), Output(selected,:), Weights_Array, Activation);
            TC(epoch) =  EuclideanLoss(TestInput(selectedTest,:), TestOutput(selectedTest,:), Weights_Array, Activation);
        else
            LC(epoch) =  Loss(Input, Output, Weights_Array, Activation);
            TC(epoch) =  Loss(TestInput, TestOutput, Weights_Array, Activation);
            AC(epoch) =  Accuracy_Classification(TestInput, TestOutput,...
                                                Weights_Array, Activation);
        end
    end
   
end

if Regression
    LC(epoch) =  EuclideanLoss(Input, Output , Weights_Array, Activation);
    TC(epoch) =  EuclideanLoss(TestInput, TestOutput, Weights_Array, Activation);
else
    LC(epoch) =  Loss(Input, Output, Weights_Array, Activation);
    TC(epoch) =  Loss(TestInput, TestOutput, Weights_Array, Activation);
    AC(epoch) =  Accuracy_Classification(TestInput, TestOutput,...
                                        Weights_Array, Activation);
end

if Regression
    [~, Output] = feedForward(TestInput, Weights_Array, Activation);
else
    Output = feedForward(TestInput, Weights_Array, Activation);
end


    
    
    
    
    