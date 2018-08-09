function meanError = CrossValidation( Input, Output, K_Folds, Parameters, regression)

% K-Folds = number of folds
% Parameters = parameters for the training
% regression = boolean variable

if nargin < 5 
    regression = false;
end

NumberOfSamples = size(Input,1);
FoldSize = floor(NumberOfSamples/K_Folds);

shuffle = randperm(NumberOfSamples);

LCurves = zeros(K_Folds,1);
VCurves = zeros(K_Folds,1);

for index = 0:K_Folds-1
    % We define the Training fold as the difference between the all
    % dataset shuffled and the TestFold

    TestFold = shuffle(FoldSize*index+1:FoldSize*(index+1));
    TrainingFold = setdiff(shuffle, TestFold);
   
    [LC, VC] = Training  (Input(TrainingFold,:), ...
                                Output(TrainingFold,:), ...
                                Input(TestFold,:), ...
                                Output(TestFold,:), ...
                                Parameters,  ...
                                true,  ... % SkipLoss
                                regression ... % Regression
                             );
    LCurves(index+1) = LC(end);
    VCurves(index+1) = VC(end);
end

meanError = mean(VCurves,1);

if isnan(meanError)
   meanError(end) = 1000; 
end



