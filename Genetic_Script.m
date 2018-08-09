
% Script for execute a Genetic Search on the Monk's dataset

%% Here we load the data
clear;
load('Data\Monks3.mat');

Output = Output*2 -1;
TestOutput = TestOutput*2 -1;

%% Default Imports
addpath('utils\');
addpath('Neural Network\');

addpath('Regularization L2');

%% Here we can choose the type of learning
addpath('Gradient Descent Learning');

%% Insert here any kind of Model Selection loading and algorithm

range = [   1       12;     % Max N� of Layers, Max N� of Hidden Units
            0.001   1;      % Learning Rate
            0       0.01;   % Regularization
            0       1       % Momentum
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
    result = AutoGrid_Model_Selection(Input,Output,5,range);
elseif x == 2
    result = Genetic_Model_Selection(Input,Output,3,range);
elseif x == 1
    result = Random_Model_Selection(Input,Output,3,range,randomTries);
end

%% Perform the Training or Model_Selection Command
tic;
N = 1;      % Number of Training
clf;        % Clear the current graphs
hold on;    % utile per il plot, non cancello quello che ho fatto prima

for i= 1:N
    
    [LC, TC, AC] = Training(Input, Output, TestInput, TestOutput,result);
    fprintf ("Last Train Error: " + LC(end)+ "\n");
    fprintf ("Last Test Error: "  + TC(end)+ "\n");
    fprintf ("Accuracy: "         + AC(end)+ "\n");
    
    subplot(1,2,1);
    
    hold on; % utile per il plot, non cancello quello che ho fatto prima
    plot(   LC(1:10:end), "blue-");
    plot(   TC(1:10:end), "red--");
        
    legend( 'TR Error',...
            'TS Error');
        
    xlabel('Epoch') % x-axis label
    ylabel('Error') % y-axis label
        
    subplot(1,2,2);
    hold on;
    plot(AC(1:10:end));
    
    legend( 'Accuracy' );
        
    xlabel('Epoch') % x-axis label
    ylabel('Accuracy') % y-axis label
    
end
hold off;
toc;

%% In the end we clean the path loading and the workspace
matlabrc;
clear;