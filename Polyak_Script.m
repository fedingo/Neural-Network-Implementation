
%% Here we load the data
clear;
load('Data/Monks3.mat');

Output = Output*2 -1;
TestOutput = TestOutput*2 -1;

%% Default Imports
addpath('utils');
addpath('Neural Network');
addpath('Regularization L1');

%% Here we can choose the type of learning
addpath('Polyak Learning');

%% Insert here any kind of Model Selection loading and algorithm

range = [   1       30;     % N� of Hidden Units
            0.01    0.1;      % Learning Rate (Beta)
            0       0.001       % Regularization
        ];

randomTry = 250;

% Training
result = Random_Model_Selection(Input,Output,3,range, randomTry);

%result = {29, 0.01, 0};

%% Perform the Training or Model_Selection Command
tic;
N = 1; % Number of Training
clf;   % Clear the current graphs


%figure;

for i= 1:N
    
    % Test
    [LC, TC, AC] = Training(Input, Output, TestInput, TestOutput, result);

    target_Epochs = 1000;
    fprintf ("Last Train Error: " + LC(target_Epochs)+ "\n");
    fprintf ("Last Test Error: "  + TC(target_Epochs)+ "\n");
    fprintf ("Accuracy: "         + AC(target_Epochs)+ "\n");
    
    subplot(1,2,1);
    
    hold on; % utile per il plot, non cancello quello che ho fatto prima
    visualization = 1:target_Epochs;
    plot(   visualization, LC(visualization), "blue-", ...
            visualization, TC(visualization), "red--");
        
    subplot(1,2,2);
    hold on;
    plot(AC);
    
end
toc;

%% In the end we clean the path loading and the workspace
matlabrc;
clear;