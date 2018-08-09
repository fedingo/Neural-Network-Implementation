
% Script to execute a Hyper-Parameter Search on the Cup dataset

%% Here we load the data
load('Data\CupData.mat');

NumberOfSamples = size(Input,1);
TestSize = 0.25 * NumberOfSamples;
shuffle = randperm(NumberOfSamples);

TestFold = shuffle(1:TestSize);
TrainingFold = setdiff(shuffle, TestFold);

TestInput  = Input (TestFold,:);
TestOutput = Output(TestFold,:);

TrainingInput  = Input (TrainingFold,:);
TrainingOutput = Output(TrainingFold,:);

%% Default Imports
addpath('utils');
addpath('Neural Network');

%% Here we choose the type of regularization
addpath('Regularization L1');

%% Here we can choose the type of learning
addpath('Polyak Learning');

%% Insert here any kind of Model Selection loading and algorithm

range = [   2       50;     % N° of Hidden Units
            0.0001  0.5;    % Learning Rate
            0       0.01;   % Regularization
            0       2       % Momentum
        ];

randomTries = 1000;

fprintf("Type 1 to use the Random Search\n");
fprintf("Type 2 to use the Genetic Search\n");
fprintf("Type 3 to use the Automatic Grid Search\n");
fprintf("Or insert your own parameters\n");
x = input('Input: ');

if iscell(x)
    result = x;
elseif x == 3
    result = AutoGrid_Model_Selection(Input,Output,3,range, true);
elseif x == 2
    result = Genetic_Model_Selection(Input,Output,3,range, true);
elseif x == 1
    result = Random_Model_Selection(Input,Output,3,range,randomTries, true);
end

%% Perform the Training or Model_Selection Command
tic;
N = 1; % Number of Training
clf;   % Clear the current graphs

TrainErrors = zeros(N,1);
TestErrors  = zeros(N,1);

for i= 1:N
    
    % Test
    [LC, TC, AC] = Training(TrainingInput, TrainingOutput,...
                            TestInput, TestOutput, result, ...
                            false, ...
                            true);

    TrainErrors(i) = LC(end);
    TestErrors(i)  = TC(end);
        
    hold on; % utile per il plot, non cancello quello che ho fatto prima
    
    step = 2;
    plot(   LC(1:step:end), "blue-");
    plot(   TC(1:step:end), "red--");
        
    legend( 'TR Error',...
            'VL2 Error');
        
    xlabel('Epoch') % x-axis label
    ylabel('Error') % y-axis label
    
end
toc;

fprintf ("Mean Train Error: " + mean(TrainErrors) + "\n");
fprintf ("Mean Val2 Error: "  + mean(TestErrors) + "\n");


%% In the end we clean the path loading and the workspace
