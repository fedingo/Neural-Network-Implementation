% Script to execute a Grid Search on the Cup dataset

%% Load the data
clear;
load('Data\CupData.mat');

NumberOfSamples = size(Input,1);
TestSize = 0.25 * NumberOfSamples;
shuffle = randperm(NumberOfSamples);

TestFold = shuffle(1:TestSize);
TrainingFold = setdiff(shuffle, TestFold);

TestInput  = Input (TestFold,:);                      
TestOutput = Output(TestFold,:);

Input  = Input (TrainingFold,:);
Output = Output(TrainingFold,:);


%% Default Imports
addpath('utils');
addpath('Neural Network');
addpath('Regularization L1');

%% Here we can choose the type of learning
addpath('Polyak Learning');

%% Insert here any kind of Model Selection loading and algorithm

range = [   2       32;       % N° of Hidden Units
            0.001   0.5;      % Learning Rate
            0       0.01;    % Regularization
            0       0.8       % Momentum
        ];

while true
    
    clf;   % Clear the current graphs
    Model_Selection(Input,Output,3,range,true);
    
    x = input('Insert best graph ID: ','s');
    
    if strcmp(x,'end')
        break; 
    end
    
    if strcmp(x,'redo')
        continue;
    end
    
    x = str2double(x);
    
    % Nï¿½ units range update
    if  x >= 9 
        range(1,1) = floor( (range(1,1) + range (1,2) )/2) ;
    else
        range(1,2) = floor( (range(1,1) + range (1,2) )/2) ;
    end
    
    % Learning Rate Range Update
    newEta = exp( (log(range(2,1)) + log(range (2,2)))/2 );
    
    if mod(x-1,8) > 3
        range(2,1) = newEta;
    else
        range(2,2) = newEta;
    end
    
    % Regularization Range Update
    if mod(x-1,4) > 1
        range(3,1) = (range(3,1) + range (3,2) )/2;
    else
        range(3,2) = (range(3,1) + range (3,2) )/2;
    end
    
    % Momentum Range Update
    if mod(x-1,2) > 0
        range(4,1) = (range(4,1) + range (4,2) )/2;
    else
        range(4,2) = (range(4,1) + range (4,2) )/2;
    end
        
end

result = mean(range,2);
result(1) = floor(result(1));
% result = [18, 0.016322, 0.000625, 0.05];

fprintf( "Final Parameters: " + result(1) +...
                ", " + result(2) +...
                ", " + result(3) +...
                ", " + result(4) +...
                "\n");

%% Perform the Training or Model_Selection Command
tic;
N = 10; % Number of Training
clf;   % Clear the current graphs

meanTrainError = 0;
meanTestError  = 0;

for i= 1:N
    
    % Test
    [LC, TC, AC] = Training(Input, Output, TestInput, TestOutput, {result(1),...
                                                                  result(2),...
                                                                  result(3),...
                                                                  result(4)},...
                                                                  false,...
                                                                  true);

    meanTrainError = meanTrainError + LC(end);
    meanTestError = meanTestError + TC(end);
    
    hold on; % utile per il plot, non cancello quello che ho fatto prima

    plot(   LC, "blue-");
    plot(   TC, "red--");
        
    legend( 'TR Error',...
            'TS Error');
        
    xlabel('Epoch') % x-axis label
    ylabel('Error') % y-axis label
    
end
toc;

fprintf ("Mean Train Error: " + meanTrainError/N + "\n");
fprintf ("Mean Test Error: "  + meanTestError/N + "\n");

%% In the end we clean the path loading and the workspace
matlabrc;
clear;