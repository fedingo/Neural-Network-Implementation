
% Script for execute a Grid Search on the Monk's dataset

%% Here we load the data
clear;
load('Data\Monks3.mat');

Output = Output*2 -1;
TestOutput = TestOutput*2 -1;

%% Default Imports
addpath('utils');
addpath('Neural Network');
addpath('Regularization L2');

%% Here we can choose the type of learning
addpath('Gradient Descent Learning');

%% Insert here any kind of Model Selection loading and algorithm

range = [   4       32;       % N# of Hidden Units
            0.001    0.5;      % Learning Rate
            0       0.01;    % Regularization
            0       0.8       % Momentum
        ];

while true
    
    clf;   % Clear the current graphs
    Model_Selection(Input,Output,3,range,false);
    
    x = input('Insert best graph ID: ','s');
    
    if strcmp(x,'end')
        break; 
    end
    
    if strcmp(x,'redo')
        continue;
    end
    
    x = str2double(x);
    
    % N� units range update
    if  x >= 9 
        range(1,1) = floor( (range(1,1) + range (1,2) )/2) ;
    else
        range(1,2) = floor( (range(1,1) + range (1,2) )/2) ;
    end
    
    % Learning Rate Range Update
    newEta = exp( (log(range(2,1)) + log(range (2,2)))/2 );
    
    if mod(x,8) >= 5
        range(2,1) = newEta;
    else
        range(2,2) = newEta;
    end
    
    % Regularization Range Update
    if mod(x,4) >= 3
        range(3,1) = (range(3,1) + range (3,2) )/2;
    else
        range(3,2) = (range(3,1) + range (3,2) )/2;
    end
    
    % Momentum Range Update
    if mod(x,2) >= 1
        range(4,1) = (range(4,1) + range (4,2) )/2;
    else
        range(4,2) = (range(4,1) + range (4,2) )/2;
    end
        
end

result = mean(range,2);
result(1) = floor(result(1));
result = [12, 0.0355, 0.000625, 0.05];

fprintf( "Final Parameters: " + result(1) +...
                ", " + result(2) +...
                ", " + result(3) +...
                ", " + result(4) +...
                "\n");

%% Perform the Training or Model_Selection Command
tic;
N = 1; % Number of Training
clf;   % Clear the current graphs

for i= 1:N
    
    % Test
    [LC, TC, AC] = Training(Input, Output, TestInput, TestOutput,{result(1),...
                                                                  result(2),...
                                                                  result(3),...
                                                                  result(4)});

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