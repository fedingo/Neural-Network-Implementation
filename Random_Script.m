
% Script for execute a Random Search on the Monk's dataset

%% Load the data
clear;
load('Data/Monks1.mat');

Output = Output*2 -1;
TestOutput = TestOutput*2 -1;

%% Default Imports
addpath('utils');
addpath('Neural Network');
addpath('Regularization L2');

%% Here we can choose the type of learning
addpath('Gradient Descent Learning');

%% Insert here any kind of Model Selection loading and algorithm

range = [   2       32;     % N� of Hidden Units
            0.01    1;      % Learning Rate
            0       0.01;   % Regularization
            0       1       % Momentum
        ];

randomTry = 250;

% Training
result = Random_Model_Selection(Input,Output,3,range, randomTry);

%result = [12, 0.50472, 0.0057184, 0.069691];

%% Perform the Training or Model_Selection Command
tic;
N = 1; % Number of Training
clf;   % Clear the current graphs


%figure;

for i= 1:N
    
    % Test
    [LC, TC, AC] = Training(Input, Output, TestInput, TestOutput, result);

    fprintf ("Last Train Error: " + LC(end)+ "\n");
    fprintf ("Last Test Error: "  + TC(end)+ "\n");
    fprintf ("Accuracy: "         + AC(end)+ "\n");
    
    subplot(1,2,1);
    
    hold on; % utile per il plot, non cancello quello che ho fatto prima
    plot(   LC, "blue-");
    plot(   TC, "red--");
        
    legend( 'TR Error',...
            'TS Error');
        
    xlabel('Epoch') % x-axis label
    ylabel('Error') % y-axis label
        
    subplot(1,2,2);
    hold on;
    plot(AC);
    
    legend( 'Accuracy' );
        
    xlabel('Epoch') % x-axis label
    ylabel('Accuracy') % y-axis label
    
end
toc;

%% In the end we clean the path loading and the workspace
matlabrc;
clear;